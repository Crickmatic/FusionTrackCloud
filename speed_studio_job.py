"""
Speed Studio job runner: metadata JSON + video bytes → full pipeline → metrics + overlay video (base64).

Used by FastAPI ``POST /v1/runsync`` (see ``app.main``). Same processing path as ``scripts/run_test3_reference.py``.
"""

from __future__ import annotations

import base64
import logging
import shutil
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

logger = logging.getLogger("fusiontrack.speed_studio_job")

_pipeline_singleton: Any = None


def _get_standalone_pipeline():
    """Lazy pipeline when no FastAPI lifespan has run (e.g. one-off scripts)."""
    global _pipeline_singleton
    if _pipeline_singleton is None:
        from app.config import get_settings
        from pipeline.orchestrator import DeliveryProcessingPipeline

        settings = get_settings()
        p = DeliveryProcessingPipeline(settings=settings)
        p.load_models()
        _pipeline_singleton = p
    return _pipeline_singleton


def run_speed_studio_job(inp: dict[str, Any], *, pipeline: Any | None = None) -> dict[str, Any]:
    """
    Input (inner ``input`` object from Speed Studio / JSON API):

    - ``metadata``: object (fusiontrack.captureMetadata.v1)
    - ``videoBase64``: base64-encoded clip bytes
    - ``debug``: bool — if true, first overlay render uses ``render_mode="debug"`` (like reference script)
    - ``renderConsumerOverlay``: bool — default true; must be true to produce overlay MP4s
    - ``filename``: optional hint for temp suffix
    """
    from app.schemas import DeliveryMetadata
    from pipeline.orchestrator import ProcessingOptions

    proc = pipeline if pipeline is not None else _get_standalone_pipeline()

    if not proc.has_required_models:
        return {
            "success": False,
            "schemaVersion": "fusiontrack.publicResult.v1",
            "debug": {
                "error": "models_missing",
                "missing": list(proc.missing_model_files),
                "hint": "Place YOLO weights under models/ (see models/README.md).",
            },
        }

    raw_meta = inp.get("metadata")
    if not isinstance(raw_meta, dict):
        return {
            "success": False,
            "schemaVersion": "fusiontrack.publicResult.v1",
            "error": "metadata must be a JSON object",
        }

    b64 = inp.get("videoBase64") or inp.get("video_base64")
    if not isinstance(b64, str) or not b64.strip():
        return {
            "success": False,
            "schemaVersion": "fusiontrack.publicResult.v1",
            "error": "videoBase64 is required",
        }

    try:
        video_bytes = base64.b64decode(b64, validate=False)
    except Exception as exc:  # noqa: BLE001
        return {
            "success": False,
            "schemaVersion": "fusiontrack.publicResult.v1",
            "error": f"Invalid base64 video: {exc}",
        }

    if len(video_bytes) < 256:
        return {
            "success": False,
            "schemaVersion": "fusiontrack.publicResult.v1",
            "error": "video payload too small",
        }

    debug = bool(inp.get("debug", False))
    want_render = bool(inp.get("renderConsumerOverlay", True)) or debug
    fname = inp.get("filename")
    suffix = Path(str(fname or "clip.mov")).suffix or ".mov"
    if suffix.lower() not in {".mov", ".mp4", ".m4v", ".avi", ".mkv"}:
        suffix = ".mov"

    upload_started = time.perf_counter()
    artifact_dir: Path | None = None
    upload_path: Path | None = None
    try:
        try:
            metadata = DeliveryMetadata.model_validate(raw_meta)
        except Exception as exc:  # noqa: BLE001
            return {
                "success": False,
                "schemaVersion": "fusiontrack.publicResult.v1",
                "error": f"Invalid metadata: {exc}",
            }

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(video_bytes)
            upload_path = Path(tmp.name)
        upload_ms = (time.perf_counter() - upload_started) * 1000.0

        artifact_dir = Path(tempfile.mkdtemp(prefix="fusiontrack-job-"))
        process_started = time.perf_counter()
        result = proc.process(
            upload_path=upload_path,
            metadata=metadata,
            artifact_dir=artifact_dir if want_render else None,
            options=ProcessingOptions(
                ground_truth=None,
                use_ground_truth_window=False,
                render_annotated_video=want_render,
                render_mode="debug" if debug else "product",
            ),
        )
        process_ms = (time.perf_counter() - process_started) * 1000.0

        out: dict[str, Any] = {
            "success": True,
            "schemaVersion": "fusiontrack.publicResult.v1",
            "uploadReadMs": upload_ms,
            "processMs": process_ms,
            "processingTimeMs": result.processingTimeMs,
        }

        thin = {
            "speedKph": result.speedKph,
            "speedMph": result.speedMph,
            "deliverySpeedKph": result.speedKph,
            "deliverySpeedMph": result.speedMph,
            "speedSource": result.speedSource,
            "releaseSec": result.releaseSec,
            "bounceSec": result.bounceSec,
            "stumpImpactSec": result.stumpImpactSec,
            "releasePoint": result.releasePoint.model_dump(),
            "bouncePoint": result.bouncePoint.model_dump(),
            "endPoint": result.endPoint.model_dump(),
            "line": result.line,
            "length": result.length,
            "lengthCategory": result.lengthCategory,
            "drsDecision": result.drsDecision,
            "confidence": result.confidence,
            "stumpsHitting": result.stumpsHitting,
            "annotatedVideoPath": result.annotatedVideoPath,
            "consumerAnnotatedVideoPath": result.consumerAnnotatedVideoPath,
            "consumerSyncAnnotatedVideoPath": result.consumerSyncAnnotatedVideoPath,
        }
        out.update(thin)
        out["result"] = thin

        # Primary return video: time-synced consumer overlay (smaller / no intro segments).
        overlay_path_str = result.consumerSyncAnnotatedVideoPath or result.consumerAnnotatedVideoPath
        if want_render and overlay_path_str:
            opath = Path(overlay_path_str)
            if opath.is_file():
                out["consumerOverlayVideoBase64"] = base64.standard_b64encode(opath.read_bytes()).decode("ascii")
                out["consumerOverlayKind"] = "sync" if result.consumerSyncAnnotatedVideoPath else "full"
            else:
                out["consumerOverlayVideoBase64"] = None
                out["debug"] = {"warning": "consumer_overlay_missing", "path": str(opath)}
        else:
            out["consumerOverlayVideoBase64"] = None

        return out
    except Exception as exc:  # noqa: BLE001
        logger.exception("run_speed_studio_job failed")
        return {
            "success": False,
            "schemaVersion": "fusiontrack.publicResult.v1",
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
    finally:
        if upload_path is not None:
            try:
                upload_path.unlink(missing_ok=True)
            except OSError:
                pass
        if artifact_dir is not None:
            try:
                shutil.rmtree(artifact_dir, ignore_errors=True)
            except OSError:
                pass
