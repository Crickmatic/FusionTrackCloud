"""Regenerate canonical clip artifacts under outputs/<slug>-latest/ (default: test3-latest)."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import get_settings  # noqa: E402
from app.schemas import DeliveryMetadata  # noqa: E402
from pipeline.orchestrator import DeliveryProcessingPipeline, ProcessingOptions  # noqa: E402


def _slug(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", name.strip()).strip("-").lower()
    return s or "clip"


def _resolve_clip(path: Path) -> Path:
    downloads = Path.home() / "Downloads"
    for candidate in (path, ROOT / path, downloads / path.name):
        if candidate.is_file():
            return candidate.resolve()
    raise SystemExit(f"Clip not found: {path}")


def default_clip() -> Path:
    samples = ROOT / "samples"
    for name in ("test 3.MOV", "test 3.mov", "test video.MOV", "clip.mov"):
        candidate = samples / name
        if candidate.is_file():
            return candidate
    raise SystemExit(
        "No reference clip in samples/. Add a .mov under samples/ or pass --clip /path/to/video.mov "
        "(paths under ~/Downloads/ are also resolved by filename)."
    )


def build_metadata(delivery_id: str) -> DeliveryMetadata:
    return DeliveryMetadata(
        schemaVersion="fusiontrack.captureMetadata.v1",
        sessionId="test3-reference-run",
        deliveryId=delivery_id,
        resolution={"width": 1080, "height": 1920},
        pitch={
            "lengthMeters": 20.12,
            "batterEndStumpsWorld": [0.0, 0.0, 0.0],
            "bowlerEndStumpsWorld": [0.0, 0.0, 20.12],
            "corridorWorld": [[-1.525, 0.0, 0.0], [1.525, 0.0, 0.0], [1.525, 0.0, 20.12], [-1.525, 0.0, 20.12]],
        },
        camera={"intrinsics": [[1200.0, 0.0, 540.0], [0.0, 1200.0, 960.0], [0.0, 0.0, 1.0]]},
        pitchCalibration={"pitchLengthMeters": 20.12, "pitchWidthMeters": 3.05},
        corridorGeometry={
            "pitchPolygon": [
                {"x": 0.44, "y": 0.18},
                {"x": 0.56, "y": 0.18},
                {"x": 0.62, "y": 0.88},
                {"x": 0.38, "y": 0.88},
            ],
            "spine": [{"x": 0.50, "y": 0.18}, {"x": 0.50, "y": 0.88}],
        },
        cameraResolution={"width": 1080, "height": 1920},
        watchEvents={"releaseWindowLikely": None, "releaseDetected": None},
        bowlingProfile={"hand": "unknown"},
    )


def write_run_summary(out_dir: Path, clip: Path, result, elapsed_s: float) -> None:
    stats = dict(result.debug.reconstructionStats or {})
    keys = (
        "postBounceUseReflected",
        "postBounceObservationsUnreliable",
        "postBounceStrictObservations",
        "postBounceHypothesisChosen",
        "bounceNetConfidence",
        "bounceSource",
        "bounceNetUsedInReconstruction",
        "impactSource",
        "pitchedInLine",
        "pitchedInLineConfidence",
        "radarSpeedMethod",
        "radarInitialSpeedKph",
        "radarArcMeters",
        "radarWindowEndSecFromRelease",
        "radarSampleCount",
        "radarRegressionMpsRaw",
        "radarRegressionMpsAdjusted",
        "radarSpeedFailure",
    )
    recon_pick = {k: stats[k] for k in keys if k in stats}
    summary = {
        "schemaVersion": "fusiontrack.runSummary.v1",
        "generatedAtUtc": datetime.now(timezone.utc).isoformat(),
        "clipPath": str(clip.resolve()),
        "outputDir": str(out_dir.resolve()),
        "processingTimeMs": result.processingTimeMs,
        "scriptElapsedSec": round(elapsed_s, 3),
        "speedKph": result.speedKph,
        "speedMph": result.speedMph,
        "speedSource": result.speedSource,
        "lengthCategory": result.lengthCategory or result.length,
        "bounceDistanceFromBatterM": result.bounceDistanceFromBatterStumpsMeters,
        "confidence": result.confidence,
        "releasePoint": result.releasePoint.model_dump(),
        "bouncePoint": result.bouncePoint.model_dump(),
        "endPoint": result.endPoint.model_dump(),
        "annotatedVideoPath": result.annotatedVideoPath,
        "consumerAnnotatedVideoPath": result.consumerAnnotatedVideoPath,
        "consumerSyncAnnotatedVideoPath": result.consumerSyncAnnotatedVideoPath,
        "reconstructionStats": recon_pick,
        "drsFinalDecision": (result.drsDecision or {}).get("finalDecision"),
    }
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run pipeline + debug/consumer renders for a reference clip.")
    parser.add_argument(
        "--clip",
        type=Path,
        default=None,
        help="Video path (relative to repo root or ~/Downloads/<name>). Default: first clip found in samples/.",
    )
    args = parser.parse_args()
    if args.clip:
        clip = _resolve_clip(args.clip)
        out_dir = ROOT / "outputs" / f"{_slug(clip.stem)}-latest"
    else:
        clip = default_clip()
        out_dir = ROOT / "outputs" / "test3-latest"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    settings = get_settings()
    pipeline = DeliveryProcessingPipeline(settings=settings)
    pipeline.load_models()
    if not pipeline.has_required_models:
        print("Missing required models:")
        for missing in pipeline.missing_model_files:
            print(f"  - {missing}")
        raise SystemExit(2)

    delivery_id = out_dir.name
    metadata = build_metadata(delivery_id=delivery_id)
    (out_dir / "metadata.json").write_text(metadata.model_dump_json(indent=2), encoding="utf-8")

    print(f"Clip:   {clip}")
    print(f"Output: {out_dir}")
    t0 = time.perf_counter()
    result = pipeline.process(
        upload_path=clip,
        metadata=metadata,
        artifact_dir=out_dir,
        options=ProcessingOptions(
            ground_truth=None,
            use_ground_truth_window=False,
            render_annotated_video=True,
            render_mode="debug",
        ),
    )
    elapsed = time.perf_counter() - t0
    write_run_summary(out_dir, clip, result, elapsed)
    print(f"Done in {elapsed:.1f}s — debug: {result.annotatedVideoPath}")
    print(f"Consumer: {result.consumerAnnotatedVideoPath}")
    print(f"Consumer (sync only): {result.consumerSyncAnnotatedVideoPath}")
    print(f"Summary:  {out_dir / 'run_summary.json'}")


if __name__ == "__main__":
    main()
