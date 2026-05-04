"""Process a local MOV/MP4 with default corridor metadata; writes outputs/<name>-latest/."""

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


def slug(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", name.strip()).strip("-").lower()
    return s or "clip"


def build_metadata(delivery_id: str) -> DeliveryMetadata:
    return DeliveryMetadata(
        schemaVersion="fusiontrack.captureMetadata.v1",
        sessionId="local-clip",
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", type=Path, required=True)
    parser.add_argument("--out", type=Path, help="Output directory (default outputs/<slug>-latest)")
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()
    clip: Path = args.clip
    if not clip.is_file():
        alt = ROOT.parent / clip.name
        if alt.is_file():
            clip = alt
        else:
            raise SystemExit(f"Clip not found: {args.clip}")

    out = args.out or (ROOT / "outputs" / f"{slug(clip.stem)}-latest")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    meta = build_metadata(delivery_id=out.name)
    (out / "metadata.json").write_text(meta.model_dump_json(indent=2), encoding="utf-8")

    pipeline = DeliveryProcessingPipeline(get_settings())
    pipeline.load_models()
    if not pipeline.has_required_models:
        raise SystemExit("Missing models")
    t0 = time.perf_counter()
    result = pipeline.process(
        clip,
        meta,
        out,
        options=ProcessingOptions(
            render_annotated_video=not args.no_render,
            render_mode="debug",
        ),
    )
    summary = {
        "schemaVersion": "fusiontrack.runSummary.v1",
        "generatedAtUtc": datetime.now(timezone.utc).isoformat(),
        "clipPath": str(clip.resolve()),
        "elapsedSec": round(time.perf_counter() - t0, 3),
        "speedKph": result.speedKph,
        "speedMph": result.speedMph,
        "lengthCategory": result.lengthCategory or result.length,
        "bounceDistanceFromBatterM": result.bounceDistanceFromBatterStumpsMeters,
        "bounceDistanceFromBowlerM": result.bounceDistanceFromBowlerStumpsMeters,
        "releaseSec": result.releaseSec,
        "bounceSec": result.bounceSec,
        "stumpImpactSec": result.stumpImpactSec,
        "bounceImageY": result.bouncePoint.y,
    }
    (out / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
