from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
import shutil
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import Settings, get_settings  # noqa: E402
from app.schemas import DeliveryMetadata  # noqa: E402
from pipeline.orchestrator import DeliveryProcessingPipeline, ProcessingOptions  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark FusionTrack cloud model/source configs.")
    parser.add_argument("--clip", required=True)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--use-ground-truth-window", action="store_true")
    parser.add_argument("--metadata-mode", choices=["video-only", "guided-boxes", "metadata-assisted"], default="metadata-assisted")
    parser.add_argument("--configs", default=None, help="Optional comma-separated config names to run.")
    parser.add_argument("--archive", action="store_true", help="Copy latest output to timestamped archive.")
    parser.add_argument("--max-frames", type=int, default=96)
    parser.add_argument("--conf", type=float, default=0.20)
    return parser.parse_args()


def build_metadata(delivery_id: str, metadata_mode: str) -> DeliveryMetadata:
    payload: dict = dict(
        schemaVersion="fusiontrack.captureMetadata.v1",
        sessionId="benchmark-session",
        deliveryId=delivery_id,
        fps=None,
        resolution={"width": 1080, "height": 1920},
        calibrationMode="video_only",
        confidence=0.32,
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
        capture={"triggerTimestampSec": None, "preRollSec": 1.0, "clipStartWallTime": None},
    )
    if metadata_mode == "guided-boxes":
        payload.update(
            calibrationMode="guided_boxes",
            confidence=0.72,
            guidedBoxes={
                "nearStumpsBoxNorm": {"x": 0.455, "y": 0.73, "w": 0.09, "h": 0.17},
                "farStumpsBoxNorm": {"x": 0.485, "y": 0.18, "w": 0.03, "h": 0.07},
                "pitchCorridorNorm": [
                    {"x": 0.44, "y": 0.18},
                    {"x": 0.56, "y": 0.18},
                    {"x": 0.62, "y": 0.88},
                    {"x": 0.38, "y": 0.88},
                ],
                "setupQuality": 0.78,
                "nearStumpsDetected": True,
                "farStumpsDetected": True,
            },
        )
    elif metadata_mode == "metadata-assisted":
        payload.update(
            calibrationMode="ar_world",
            confidence=0.95,
            pitch={
                "lengthMeters": 20.12,
                "batterEndStumpsWorld": [0.0, 0.0, 0.0],
                "bowlerEndStumpsWorld": [0.0, 0.0, 20.12],
                "creasePositionsMeters": {"battingCreaseFromBatterStumps": 1.22, "bowlingCreaseFromBowlerStumps": 1.22},
                "corridorWorld": [[-1.525, 0.0, 0.0], [1.525, 0.0, 0.0], [1.525, 0.0, 20.12], [-1.525, 0.0, 20.12]],
            },
            camera={
                "intrinsics": [[1200.0, 0.0, 540.0], [0.0, 1200.0, 960.0], [0.0, 0.0, 1.0]],
                "staticPose": None,
            },
            watch={"releaseTimestampSec": None, "confidence": None},
        )
    return DeliveryMetadata(**payload)


def release_error_ms(predicted: float | None, gt_range: list[float]) -> float | None:
    if predicted is None:
        return None
    low, high = gt_range
    if low <= predicted <= high:
        return 0.0
    return abs((low if predicted < low else high) - predicted) * 1000.0


def point_error_ms(predicted: float | None, gt: float) -> float | None:
    if predicted is None:
        return None
    return abs(predicted - gt) * 1000.0


def model_exists(name: str, settings: Settings) -> bool:
    normalized = name if Path(name).suffix else f"{name}.pt"
    path = settings._resolve_model_path(normalized)
    if path.exists():
        return True
    return False


def _extract_zip_once(zip_path: Path, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    marker = destination / ".extracted"
    if marker.exists():
        return destination
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(destination)
    marker.write_text("ok", encoding="utf-8")
    return destination


def _find_supported_artifacts(root: Path) -> list[Path]:
    matches: list[Path] = []
    for suffix in ("*.pt", "*.onnx", "*.engine", "*.mlpackage", "*.mlmodel"):
        matches.extend(root.rglob(suffix))
    return sorted(set(matches))


def discover_cricket_model_aliases() -> tuple[dict[str, str], dict[str, str]]:
    repo_root = ROOT.parent
    extraction_root = ROOT / "models" / "extracted_cricket"
    model_dir = ROOT / "models"
    aliases: dict[str, str] = {}
    load_notes: dict[str, str] = {}
    direct_candidates = {
        "cricket_stumps_v1": [repo_root / "v1.pt", ROOT / "v1.pt", repo_root / "cricket_stumps_v1.pt", ROOT / "cricket_stumps_v1.pt"],
        "cricket_ball_v2": [repo_root / "v2.pt", ROOT / "v2.pt", repo_root / "cricket_ball_v2.pt", ROOT / "cricket_ball_v2.pt"],
    }

    for alias, paths in direct_candidates.items():
        direct_path = next((path for path in paths if path.exists()), None)
        if direct_path is None:
            continue
        link_name = f"{alias}.pt"
        linked_path = model_dir / link_name
        if linked_path.exists() or linked_path.is_symlink():
            linked_path.unlink()
        linked_path.symlink_to(direct_path.resolve())
        aliases[alias] = link_name
        load_notes[alias] = f"using direct weight {linked_path} -> {direct_path}"
    del extraction_root
    return aliases, load_notes


def benchmark_configs(settings: Settings, cricket_aliases: dict[str, str]) -> list[dict]:
    del settings
    stumps_model = cricket_aliases.get("cricket_stumps_v1", "cricket_stumps_v1")
    ball_model = cricket_aliases.get("cricket_ball_v2", "cricket_ball_v2")
    return [
        {"name": "ball_v2_only", "models": ball_model, "motion": False, "requires": [ball_model]},
        {"name": "ball_v2_plus_motion", "models": ball_model, "motion": True, "requires": [ball_model]},
        {"name": "ball_v2_plus_stumps", "models": ",".join([ball_model, stumps_model]), "motion": False, "requires": [ball_model, stumps_model]},
        {"name": "ball_v2_plus_stumps_plus_motion", "models": ",".join([ball_model, stumps_model]), "motion": True, "requires": [ball_model, stumps_model]},
        {"name": "yolo_context_only", "models": "yolo26n,yolo26s,yolov8s", "motion": False},
        {"name": "yolo_context_plus_ball_v2", "models": ",".join(["yolo26n", "yolo26s", "yolov8s", ball_model]), "motion": False, "requires": [ball_model]},
        {
            "name": "all_fusion",
            "models": ",".join(["yolo26n", "yolo26s", "yolov8s", ball_model, stumps_model]),
            "motion": True,
            "requires": [ball_model, stumps_model],
        },
    ]


def row_rank_key(row: dict) -> tuple:
    success_rank = 0 if row["success"] else 1
    return (
        success_rank,
        row.get("bounceTimeErrorMs") or 999999,
        row.get("releaseTimeErrorMs") or 999999,
        row.get("endTimeErrorMs") or 999999,
        -(row.get("confidence") or 0),
        row.get("processingTimeMs") or 999999,
    )


def main() -> None:
    args = parse_args()
    settings = get_settings().model_copy(deep=True)
    settings.debug = args.debug
    settings.yolo_confidence_threshold = args.conf

    clip_path = Path(args.clip)
    if not clip_path.exists():
        fallback = ROOT.parent / args.clip
        if fallback.exists():
            clip_path = fallback
    if not clip_path.exists():
        raise SystemExit(f"Clip not found: {args.clip}")

    gt_path = Path(args.ground_truth)
    if not gt_path.exists():
        fallback = ROOT.parent / args.ground_truth
        if fallback.exists():
            gt_path = fallback
    ground_truth = json.loads(gt_path.read_text(encoding="utf-8"))

    benchmark_name = (
        "latest_oracle"
        if args.use_ground_truth_window
        else (
            "latest_calibrated"
            if args.metadata_mode == "metadata-assisted"
            else ("latest_guided_boxes" if args.metadata_mode == "guided-boxes" else "latest_video_only")
        )
    )
    benchmark_root = ROOT / "outputs" / "benchmarks" / benchmark_name
    if benchmark_root.exists():
        for child in benchmark_root.iterdir():
            if child.is_dir():
                import shutil

                shutil.rmtree(child)
            else:
                child.unlink()
    benchmark_root.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []
    missing_models: set[str] = set()
    cricket_aliases, cricket_notes = discover_cricket_model_aliases()

    requested_configs = {name.strip() for name in args.configs.split(",")} if args.configs else None
    for config in benchmark_configs(settings, cricket_aliases):
        if requested_configs is not None and config["name"] not in requested_configs:
            continue
        required = config.get("requires", [])
        unmet = [name for name in required if not model_exists(name, settings)]
        if unmet:
            missing_models.update(unmet)
            summary_rows.append(
                {
                    "config": config["name"],
                    "success": False,
                    "failureReason": f"missing required model(s): {', '.join(unmet)}",
                }
            )
            continue

        run_settings = settings.model_copy(deep=True)
        run_settings.models = config["models"]
        pipeline = DeliveryProcessingPipeline(run_settings)
        pipeline.load_models()
        run_dir = benchmark_root / config["name"]
        run_dir.mkdir(parents=True, exist_ok=True)

        if not pipeline.has_required_models:
            missing_models.update(pipeline.missing_model_files)
            summary_rows.append(
                {
                    "config": config["name"],
                    "success": False,
                    "failureReason": "no models loaded",
                    "missingModels": pipeline.missing_model_files,
                }
            )
            continue

        metadata = build_metadata(delivery_id=config["name"], metadata_mode=args.metadata_mode)
        options = ProcessingOptions(
            use_ground_truth_window=args.use_ground_truth_window,
            ground_truth=ground_truth,
            enable_motion=config["motion"],
            max_frames=args.max_frames,
        )
        try:
            result = pipeline.process(upload_path=clip_path, metadata=metadata, artifact_dir=run_dir, options=options)
            row = {
                "config": config["name"],
                "success": True,
                "releaseSec": result.releasePoint.t,
                "bounceSec": result.bouncePoint.t,
                "endSec": result.endPoint.t,
                "releaseTimeErrorMs": release_error_ms(result.releasePoint.t, ground_truth["releaseSecRange"]),
                "bounceTimeErrorMs": point_error_ms(result.bouncePoint.t, ground_truth["bounceSec"]),
                "endTimeErrorMs": point_error_ms(result.endPoint.t, ground_truth["stumpImpactSec"]),
                "activeStartSec": result.activeStartSec,
                "activeEndSec": result.activeEndSec,
                "activeStartErrorMs": point_error_ms(result.activeStartSec, 3.2),
                "activeEndErrorMs": point_error_ms(result.activeEndSec, 4.3),
                "speedKph": result.speedKph,
                "speedMph": result.speedMph,
                "speedConfidence": result.speedConfidence,
                "calibrationConfidence": result.calibrationConfidence,
                "bounceDistanceFromBatterStumpsMeters": result.bounceDistanceFromBatterStumpsMeters,
                "bounceDistanceFromBowlerStumpsMeters": result.bounceDistanceFromBowlerStumpsMeters,
                "lineMetersFromCenter": result.lineMetersFromCenter,
                "lineMetersFromOffStump": result.lineMetersFromOffStump,
                "lengthCategory": result.lengthCategory,
                "confidence": result.confidence,
                "stumpImpactPredictedSec": result.stumpImpactPredictedSec,
                "stumpImpactConfidence": result.stumpImpactConfidence,
                "distanceToStumpRoiPx": result.distanceToStumpRoiPx,
                "endpointSource": result.endpointSource,
                "processingTimeMs": result.processingTimeMs,
                "decodeTimeMs": (result.debug.stageTimingsMs or {}).get("decode"),
                "frameExtractionTimeMs": (result.debug.stageTimingsMs or {}).get("frame_extraction"),
                "ballStumpInferenceTimeMs": (result.debug.stageTimingsMs or {}).get("ball_stump_inference"),
                "solverTimeMs": (result.debug.stageTimingsMs or {}).get("solver"),
                "debugWriteTimeMs": (result.debug.stageTimingsMs or {}).get("debug_artifacts"),
                "renderVideoTimeMs": (result.debug.stageTimingsMs or {}).get("render_video"),
                "annotatedVideoPath": result.annotatedVideoPath,
                "renderFramesCount": result.renderFramesCount,
                "outputVideoSizeMb": result.outputVideoSizeMb,
                "framesProcessed": result.debug.candidateCounts.get("frames_processed"),
                "fpsUsed": result.fps,
                "deviceUsed": run_settings.device,
                "candidateCounts": result.debug.candidateCounts,
                "modelsUsed": result.modelsUsed,
                "failureReason": "",
            }
        except Exception as exc:
            if not (run_dir / "result.json").exists():
                pipeline.write_failure_artifacts(run_dir, str(exc))
            row = {
                "config": config["name"],
                "success": False,
                "failureReason": str(exc),
            }
        summary_rows.append(row)

    ranked = sorted(summary_rows, key=row_rank_key)
    balanced_candidates = [
        row for row in ranked
        if row.get("success") and row.get("config") != "all_fusion"
    ]
    accuracy_candidates = [row for row in summary_rows if row.get("success")]
    summary_json = {
        "clip": str(clip_path),
        "groundTruth": ground_truth,
        "missingModels": sorted(missing_models),
        "cricketModelDiscovery": cricket_notes,
        "cricketModelAliases": cricket_aliases,
        "results": summary_rows,
        "recommendation": {
            "best_fast_config": min(
                [row for row in summary_rows if row.get("success")], key=lambda row: row.get("processingTimeMs", 999999), default=None
            ),
            "best_accuracy_config": min(
                accuracy_candidates,
                key=lambda row: (
                    round(row.get("endTimeErrorMs", 999999), 3),
                    round(row.get("bounceTimeErrorMs", 999999), 3),
                    round(row.get("releaseTimeErrorMs", 999999), 3),
                    row.get("processingTimeMs", 999999),
                ),
                default=None,
            ),
            "best_balanced_config": balanced_candidates[0] if balanced_candidates else None,
        },
    }
    (benchmark_root / "summary.json").write_text(json.dumps(summary_json, indent=2), encoding="utf-8")

    csv_fields = [
        "config",
        "success",
        "releaseSec",
        "bounceSec",
        "endSec",
        "releaseTimeErrorMs",
        "bounceTimeErrorMs",
        "endTimeErrorMs",
        "activeStartSec",
        "activeEndSec",
        "activeStartErrorMs",
        "activeEndErrorMs",
        "speedKph",
        "speedMph",
        "speedConfidence",
        "calibrationConfidence",
        "bounceDistanceFromBatterStumpsMeters",
        "bounceDistanceFromBowlerStumpsMeters",
        "lineMetersFromCenter",
        "lineMetersFromOffStump",
        "lengthCategory",
        "confidence",
        "stumpImpactPredictedSec",
        "stumpImpactConfidence",
        "distanceToStumpRoiPx",
        "endpointSource",
        "processingTimeMs",
        "decodeTimeMs",
        "frameExtractionTimeMs",
        "ballStumpInferenceTimeMs",
        "solverTimeMs",
        "debugWriteTimeMs",
        "renderVideoTimeMs",
        "annotatedVideoPath",
        "renderFramesCount",
        "outputVideoSizeMb",
        "framesProcessed",
        "fpsUsed",
        "deviceUsed",
        "failureReason",
    ]
    with (benchmark_root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({field: row.get(field) for field in csv_fields})

    lines = ["# FusionTrack Cloud Benchmark Summary", "", f"- Clip: `{clip_path}`", f"- Ground truth: `{gt_path}`", ""]
    lines.append("| Rank | Config | Success | Release Err (ms) | Bounce Err (ms) | End Err (ms) | Confidence | Time (ms) |")
    lines.append("|---:|---|:---:|---:|---:|---:|---:|---:|")
    for rank, row in enumerate(ranked, start=1):
        lines.append(
            f"| {rank} | {row.get('config')} | {'Y' if row.get('success') else 'N'} | "
            f"{row.get('releaseTimeErrorMs', 'n/a')} | {row.get('bounceTimeErrorMs', 'n/a')} | {row.get('endTimeErrorMs', 'n/a')} | "
            f"{row.get('confidence', 'n/a')} | {row.get('processingTimeMs', 'n/a')} |"
        )
    lines.append("")
    lines.append("## Recommendation")
    lines.append(f"- best_fast_config: `{(summary_json['recommendation']['best_fast_config'] or {}).get('config', 'n/a')}`")
    lines.append(f"- best_accuracy_config: `{(summary_json['recommendation']['best_accuracy_config'] or {}).get('config', 'n/a')}`")
    lines.append(f"- best_balanced_config: `{(summary_json['recommendation']['best_balanced_config'] or {}).get('config', 'n/a')}`")
    (benchmark_root / "summary.md").write_text("\n".join(lines), encoding="utf-8")

    if args.archive:
        archive_root = ROOT / "outputs" / "benchmarks" / "archive"
        archive_root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        archive_dir = archive_root / f"{benchmark_name}-{stamp}"
        if archive_dir.exists():
            shutil.rmtree(archive_dir)
        shutil.copytree(benchmark_root, archive_dir)

    print(f"Benchmark complete: {benchmark_root}")
    print(f"Missing models: {sorted(missing_models)}")


if __name__ == "__main__":
    main()
