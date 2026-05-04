from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import get_settings  # noqa: E402
from app.schemas import DeliveryMetadata  # noqa: E402
from pipeline.orchestrator import DeliveryProcessingPipeline, ProcessingOptions  # noqa: E402


def find_sample_video() -> Path:
    samples = ROOT / "samples"
    for suffix in ("*.mp4", "*.mov", "*.m4v", "*.avi", "*.zip"):
        matches = sorted(samples.glob(suffix))
        if matches:
            return matches[0]
    # Convenience fallback for root-level manual test clips.
    for suffix in ("*.MOV", "*.mov", "*.mp4", "*.m4v", "*.avi", "*.zip"):
        matches = sorted(ROOT.parent.glob(suffix))
        if matches:
            return matches[0]
    named = ROOT.parent / "test video.MOV"
    if named.exists():
        return named
    raise SystemExit("No sample clip found. Add a small .mp4/.mov or frame .zip under fusiontrack-cloud/samples/.")


def build_metadata(delivery_id: str) -> DeliveryMetadata:
    return DeliveryMetadata(
        schemaVersion="fusiontrack.captureMetadata.v1",
        sessionId="local-smoke-session",
        deliveryId=delivery_id,
        resolution={"width": 1080, "height": 1920},
        pitch={
            "lengthMeters": 20.12,
            "batterEndStumpsWorld": [0.0, 0.0, 0.0],
            "bowlerEndStumpsWorld": [0.0, 0.0, 20.12],
            "corridorWorld": [[-1.525, 0.0, 0.0], [1.525, 0.0, 0.0], [1.525, 0.0, 20.12], [-1.525, 0.0, 20.12]],
        },
        camera={
            "intrinsics": [[1200.0, 0.0, 540.0], [0.0, 1200.0, 960.0], [0.0, 0.0, 1.0]],
        },
        pitchCalibration={
            "pitchLengthMeters": 20.12,
            "pitchWidthMeters": 3.05,
        },
        corridorGeometry={
            "pitchPolygon": [
                {"x": 0.44, "y": 0.18},
                {"x": 0.56, "y": 0.18},
                {"x": 0.62, "y": 0.88},
                {"x": 0.38, "y": 0.88},
            ],
            "spine": [
                {"x": 0.50, "y": 0.18},
                {"x": 0.50, "y": 0.88},
            ],
        },
        cameraResolution={"width": 1080, "height": 1920},
        watchEvents={
            "releaseWindowLikely": None,
            "releaseDetected": None,
        },
        bowlingProfile={"hand": "unknown"},
    )


def load_ground_truth() -> dict | None:
    path = ROOT / "samples" / "test_video_ground_truth.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    settings = get_settings()
    pipeline = DeliveryProcessingPipeline(settings=settings)
    pipeline.load_models()
    if not pipeline.has_required_models:
        print("Missing required models:")
        for missing in pipeline.missing_model_files:
            print(f"  - {missing}")
        raise SystemExit(2)

    sample_path = find_sample_video()
    job_id = f"local-{uuid.uuid4()}"
    output_dir = ROOT / "outputs" / job_id
    metadata = build_metadata(delivery_id=job_id)
    print(f"Sample: {sample_path}")
    print(f"Output: {output_dir}")
    try:
        result = pipeline.process(
            upload_path=sample_path,
            metadata=metadata,
            artifact_dir=output_dir,
            options=ProcessingOptions(ground_truth=load_ground_truth(), use_ground_truth_window=False),
        )
    except Exception as exc:
        print(f"Processing failed: {exc}")
        print(f"Failure debug report: {output_dir / 'debug_report.html'}")
        raise SystemExit(1)

    print(f"Candidate counts: {result.debug.candidateCounts}")
    print(f"Trajectory points: {len(result.trajectory)}")
    print(f"Speed: {result.speedKph}")
    print(f"Speed mph: {result.speedMph}")
    print(f"Speed confidence: {result.speedConfidence}")
    print(f"Calibration confidence: {result.calibrationConfidence}")
    print(f"Confidence: {result.confidence:.3f}")
    print(json.dumps(result.model_dump(), indent=2))


if __name__ == "__main__":
    main()
