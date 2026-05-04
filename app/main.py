from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, HTTPException, Request, UploadFile

from app.auth import require_engine_auth
from app.config import get_settings
from app.schemas import DebugProcessResponse, DeliveryJobCreated, DeliveryJobStatus, DeliveryMetadata, DeliveryResult, HealthResponse, JobStatus, ModelsResponse
from pipeline.orchestrator import DeliveryProcessingPipeline, ProcessingOptions
from storage.local_store import LocalDeliveryStore

logger = logging.getLogger("fusiontrack-cloud")
logging.basicConfig(level=logging.INFO)

settings = get_settings()
store = LocalDeliveryStore(settings)
pipeline = DeliveryProcessingPipeline(settings=settings)

@asynccontextmanager
async def lifespan(_app: FastAPI):
    logger.info("Starting FusionTrack Cloud service")
    pipeline.load_models()
    if settings.warmup_on_startup:
        pipeline.warmup_models()
    logger.info("Loaded models: %s", pipeline.loaded_models)
    logger.info("GPU available: %s %s", pipeline.gpu_available, pipeline.gpu_name or "")
    yield


app = FastAPI(title="FusionTrack Cloud", version="0.1.0", lifespan=lifespan)


@app.post("/v1/runsync", dependencies=[Depends(require_engine_auth)])
async def speed_studio_runsync(request: Request) -> dict:
    """
    JSON for iOS Speed Studio: body ``{"input": { metadata, videoBase64, ... }}``,
    response ``{ "status": "COMPLETED", "output": { ... } }``.

    Point the app at your engine URL (Vast ``http://<ip>:<mapped-port>``, RunPod proxy, or SSH tunnel
    to ``127.0.0.1:8000``). Path must end with ``/v1/runsync`` so the client does not double-append.
    """
    try:
        body = await request.json()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Expected JSON body: {exc}") from exc
    inner = body.get("input")
    if not isinstance(inner, dict):
        raise HTTPException(status_code=400, detail="JSON body must include an object in key 'input'")
    from speed_studio_job import run_speed_studio_job

    out = run_speed_studio_job(inner, pipeline=pipeline)
    return {"status": "COMPLETED", "id": str(uuid.uuid4()), "output": out}


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    status = "ok" if pipeline.has_required_models and pipeline.warmup_status in {"ready", "not_started"} else "models_missing"
    return HealthResponse(
        status=status,
        service=settings.service_name,
        loadedModels=pipeline.loaded_models,
        missingModelFiles=pipeline.missing_model_files,
        gpuAvailable=pipeline.gpu_available,
        torchCudaAvailable=pipeline.torch_cuda_available,
        gpuName=pipeline.gpu_name,
        ultralyticsVersion=pipeline.ultralytics_version,
        opencvVersion=pipeline.opencv_version,
        device=settings.device,
        debug=settings.debug,
        warmupStatus=pipeline.warmup_status,
        warmupDevice=pipeline.warmup_device,
        warmupNotes=pipeline.warmup_notes,
    )


@app.get("/models", response_model=ModelsResponse)
async def models() -> ModelsResponse:
    roles, paths = pipeline.model_registry()
    loaded_set = set(pipeline.loaded_models)
    loaded_map = {alias: any(alias in model_name for model_name in loaded_set) for alias in roles}
    return ModelsResponse(
        defaultProductionConfig="ball_v2_plus_stumps",
        modelsUsedByDefault=["cricket_ball_v2", "cricket_stumps_v1"],
        loadedModels=pipeline.loaded_models,
        modelRoles=roles,
        modelPaths=paths,
        modelLoaded=loaded_map,
        warmupStatus=pipeline.warmup_status,
        warmupDevice=pipeline.warmup_device,
        warmupNotes=pipeline.warmup_notes,
    )


@app.post("/v1/deliveries", response_model=DeliveryJobCreated, dependencies=[Depends(require_engine_auth)])
async def create_delivery(
    background_tasks: BackgroundTasks,
    upload: UploadFile = File(...),
    metadata: str = Form(...),
) -> DeliveryJobCreated:
    parsed_metadata = _parse_metadata(metadata)
    _validate_upload(upload)

    job_id = str(uuid.uuid4())
    store.write_status(DeliveryJobStatus(jobId=job_id, status=JobStatus.queued))

    upload_path = store.save_upload(job_id, upload.file, upload.filename or "delivery.bin")
    metadata_path = store.save_metadata(job_id, parsed_metadata)

    background_tasks.add_task(process_delivery_job, job_id, upload_path, metadata_path)
    return DeliveryJobCreated(jobId=job_id, status=JobStatus.queued)


@app.post("/process-delivery", response_model=DeliveryResult, dependencies=[Depends(require_engine_auth)])
async def process_delivery(
    upload: UploadFile = File(...),
    metadata: str = Form(...),
    debug: bool = Form(False),
    renderAnnotatedVideo: bool = Form(False),
) -> DeliveryResult:
    parsed_metadata = _parse_metadata(metadata)
    _validate_upload(upload)
    job_id = str(uuid.uuid4())
    upload_started = time.perf_counter()
    upload_path = store.save_upload(job_id, upload.file, upload.filename or "delivery.bin")
    upload_ms = (time.perf_counter() - upload_started) * 1000
    artifact_dir = store.job_dir(job_id) if debug else None
    started = time.perf_counter()
    result = pipeline.process(
        upload_path=upload_path,
        metadata=parsed_metadata,
        artifact_dir=artifact_dir,
        options=ProcessingOptions(render_annotated_video=renderAnnotatedVideo, render_mode="debug" if debug else "product"),
    )
    elapsed = time.perf_counter() - started
    logger.info("request timing ms upload_read=%.1f process_total=%.1f", upload_ms, elapsed * 1000.0)
    if elapsed > settings.request_timeout_sec:
        raise HTTPException(status_code=504, detail=f"Processing exceeded timeout policy ({settings.request_timeout_sec:.1f}s).")
    return result


@app.post("/process-delivery-debug", response_model=DebugProcessResponse, dependencies=[Depends(require_engine_auth)])
async def process_delivery_debug(
    upload: UploadFile = File(...),
    metadata: str = Form(...),
) -> DebugProcessResponse:
    parsed_metadata = _parse_metadata(metadata)
    _validate_upload(upload)
    job_id = str(uuid.uuid4())
    upload_started = time.perf_counter()
    upload_path = store.save_upload(job_id, upload.file, upload.filename or "delivery.bin")
    upload_ms = (time.perf_counter() - upload_started) * 1000
    artifact_dir = store.job_dir(job_id)
    started = time.perf_counter()
    result = pipeline.process(
        upload_path=upload_path,
        metadata=parsed_metadata,
        artifact_dir=artifact_dir,
        options=ProcessingOptions(render_annotated_video=True, render_mode="debug"),
    )
    elapsed = time.perf_counter() - started
    logger.info("debug request timing ms upload_read=%.1f process_total=%.1f", upload_ms, elapsed * 1000.0)
    if elapsed > settings.request_timeout_sec:
        raise HTTPException(status_code=504, detail=f"Processing exceeded timeout policy ({settings.request_timeout_sec:.1f}s).")
    manifest = sorted(path.name for path in artifact_dir.iterdir())
    annotated_frames = artifact_dir / "annotated_frames"
    summary = artifact_dir / "summary.md"
    return DebugProcessResponse(
        result=result,
        debugFolder=str(artifact_dir),
        debugManifest=manifest,
        annotatedVideoPath=result.annotatedVideoPath,
        consumerAnnotatedVideoPath=result.consumerAnnotatedVideoPath,
        annotatedFramesPath=str(annotated_frames) if annotated_frames.exists() else None,
        summaryPath=str(summary) if summary.exists() else None,
    )


@app.get("/v1/deliveries/{job_id}", response_model=DeliveryJobStatus, dependencies=[Depends(require_engine_auth)])
async def get_delivery(job_id: str) -> DeliveryJobStatus:
    status = store.read_status(job_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Delivery job not found")
    return status


def process_delivery_job(job_id: str, upload_path: Path, metadata_path: Path) -> None:
    store.set_status(job_id, JobStatus.processing)
    artifact_dir = store.job_dir(job_id)
    try:
        metadata = DeliveryMetadata.model_validate_json(metadata_path.read_text(encoding="utf-8"))
        result = pipeline.process(upload_path=upload_path, metadata=metadata, artifact_dir=artifact_dir)
        store.write_status(DeliveryJobStatus(jobId=job_id, status=JobStatus.complete, result=result))
    except Exception as exc:
        logger.exception("Delivery processing failed for job %s", job_id)
        pipeline.write_failure_artifacts(artifact_dir=artifact_dir, reason=str(exc))
        store.write_status(DeliveryJobStatus(jobId=job_id, status=JobStatus.failed, error=str(exc)))


def _parse_metadata(metadata: str) -> DeliveryMetadata:
    if not pipeline.has_required_models:
        missing = ", ".join(pipeline.missing_model_files) if pipeline.missing_model_files else "no YOLO models loaded"
        raise HTTPException(
            status_code=503,
            detail=f"FusionTrack Cloud model files are missing; cannot process deliveries ({missing}).",
        )
    try:
        return DeliveryMetadata.model_validate(json.loads(metadata))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid metadata JSON: {exc}") from exc


def _validate_upload(upload: UploadFile) -> None:
    file_obj = upload.file
    current = file_obj.tell()
    file_obj.seek(0, 2)
    size_bytes = file_obj.tell()
    file_obj.seek(current, 0)
    max_bytes = settings.max_upload_mb * 1024 * 1024
    if size_bytes > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Upload too large ({size_bytes / (1024 * 1024):.1f}MB). Max allowed: {settings.max_upload_mb}MB.",
        )
