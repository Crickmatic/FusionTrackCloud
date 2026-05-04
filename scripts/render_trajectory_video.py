from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import get_settings  # noqa: E402
from app.schemas import Candidate, DeliveryMetadata, DeliveryResult  # noqa: E402
from pipeline.orchestrator import DeliveryProcessingPipeline, ProcessingOptions  # noqa: E402
from pipeline.trajectory_renderer import TrajectoryRenderer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render DRS-style trajectory overlay video.")
    parser.add_argument("--clip", required=True)
    parser.add_argument("--result")
    parser.add_argument("--metadata")
    parser.add_argument("--config", default="ball_v2_plus_stumps")
    parser.add_argument("--process", action="store_true")
    parser.add_argument("--out", required=True)
    parser.add_argument("--mode", choices=["product", "debug", "consumer", "consumer_sync"], default="product")
    parser.add_argument("--archive", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    clip = Path(args.clip)
    if not clip.exists():
        clip = ROOT.parent / args.clip
    if not clip.exists():
        raise SystemExit(f"Clip not found: {args.clip}")
    output = Path(args.out)
    if args.archive:
        output = output.with_name(f"{output.stem}-{Path.cwd().name}{output.suffix}")
    output.parent.mkdir(parents=True, exist_ok=True)

    if args.process:
        if not args.metadata:
            raise SystemExit("--metadata is required with --process")
        metadata = DeliveryMetadata.model_validate_json(Path(args.metadata).read_text(encoding="utf-8"))
        settings = get_settings().model_copy(deep=True)
        settings.models = "cricket_ball_v2,cricket_stumps_v1" if args.config == "ball_v2_plus_stumps" else settings.models
        pipeline = DeliveryProcessingPipeline(settings)
        pipeline.load_models()
        run_dir = output.parent / "latest_process"
        result = pipeline.process(
            upload_path=clip,
            metadata=metadata,
            artifact_dir=run_dir,
            options=ProcessingOptions(render_annotated_video=True, render_mode=args.mode),
        )
        print(
            json.dumps(
                {
                    "annotatedVideoPath": result.annotatedVideoPath,
                    "consumerAnnotatedVideoPath": result.consumerAnnotatedVideoPath,
                    "consumerSyncAnnotatedVideoPath": result.consumerSyncAnnotatedVideoPath,
                    "renderFramesCount": result.renderFramesCount,
                    "outputVideoSizeMb": result.outputVideoSizeMb,
                },
                indent=2,
            )
        )
        return

    if not args.result:
        raise SystemExit("--result is required when --process is not set")
    result_payload = json.loads(Path(args.result).read_text(encoding="utf-8"))
    result = DeliveryResult.model_validate(result_payload)
    metadata_path = Path(args.metadata) if args.metadata else (Path(args.result).parent / "metadata.json")
    if not metadata_path.exists():
        raise SystemExit("Metadata not found. Provide --metadata explicitly.")
    metadata = DeliveryMetadata.model_validate_json(metadata_path.read_text(encoding="utf-8"))
    candidates_path = Path(args.result).parent / "candidates.json"
    selected_tracklet: list[Candidate] = []
    merged_candidates: list[Candidate] = []
    if candidates_path.exists():
        data = json.loads(candidates_path.read_text(encoding="utf-8"))
        selected_tracklet = [Candidate.model_validate(candidate) for candidate in data.get("candidates", {}).get("selected_tracklet", [])]
        merged_candidates = [Candidate.model_validate(candidate) for candidate in data.get("candidates", {}).get("active_window", [])]
    renderer = TrajectoryRenderer()
    artifacts = renderer.render(
        clip_path=clip,
        result=result,
        metadata=metadata,
        selected_tracklet=selected_tracklet,
        merged_candidates=merged_candidates,
        output_path=output,
        mode=args.mode,
    )
    print(
        json.dumps(
            {
                "annotatedVideoPath": str(artifacts.annotated_video_path),
                "renderTimeMs": artifacts.render_time_ms,
                "renderFramesCount": artifacts.frames_rendered,
                "outputVideoSizeMb": artifacts.output_video_size_mb,
                "manifest": str(artifacts.render_manifest_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
