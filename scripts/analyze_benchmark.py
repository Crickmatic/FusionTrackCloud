from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate detailed benchmark analysis files.")
    parser.add_argument("--benchmark-dir", required=True)
    return parser.parse_args()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe(value, default=0.0):
    return default if value is None else value


def _score_success_row(row: dict) -> float:
    timing_mae = (_safe(row.get("releaseTimeErrorMs")) + _safe(row.get("bounceTimeErrorMs")) + _safe(row.get("endTimeErrorMs"))) / 3.0
    active_mae = (_safe(row.get("activeStartErrorMs")) + _safe(row.get("activeEndErrorMs"))) / 2.0
    processing = _safe(row.get("processingTimeMs"), 999999.0)
    confidence = _safe(row.get("confidence"))
    counts = row.get("candidateCounts") or {}
    robust_used = _safe(counts.get("robust_fit_used_points"))
    outliers = _safe(counts.get("robust_fit_outliers_removed"))
    yolo_used = _safe(counts.get("yolo_points_used"))
    motion_used = _safe(counts.get("motion_points_used"))
    speed_penalty = 0.0 if row.get("speedKph") is not None else 800.0
    return timing_mae + 0.4 * active_mae + 0.002 * processing + speed_penalty - (300.0 * confidence) - (8.0 * robust_used) + (8.0 * outliers) - (4.0 * yolo_used) + (2.0 * motion_used)


def _fmt(x) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return f"{x:.3f}".rstrip("0").rstrip(".")
    return str(x)


def main() -> None:
    args = parse_args()
    benchmark_dir = Path(args.benchmark_dir)
    summary = _load_json(benchmark_dir / "summary.json")
    results = summary.get("results", [])
    successful = [row for row in results if row.get("success")]
    failed = [row for row in results if not row.get("success")]

    ranked_success = sorted(successful, key=_score_success_row)
    best_accuracy = min(
        successful,
        key=lambda r: (
            round(_safe(r.get("endTimeErrorMs"), 999999.0), 3),
            round(_safe(r.get("bounceTimeErrorMs"), 999999.0), 3),
            round(_safe(r.get("releaseTimeErrorMs"), 999999.0), 3),
            _safe(r.get("processingTimeMs"), 999999.0),
        ),
        default=None,
    )
    fastest = min(successful, key=lambda r: _safe(r.get("processingTimeMs"), 999999.0), default=None)
    balanced_success = [row for row in ranked_success if row.get("config") != "all_fusion"]
    best_balanced = balanced_success[0] if balanced_success else None
    most_yolo_supported = max(successful, key=lambda r: _safe((r.get("candidateCounts") or {}).get("yolo_points_used")), default=None)
    most_motion_dominated = max(successful, key=lambda r: _safe((r.get("candidateCounts") or {}).get("motion_points_used")), default=None)

    table_rows = []
    for row in results:
        run_dir = benchmark_dir / str(row.get("config"))
        result_payload = _load_json(run_dir / "result.json") if (run_dir / "result.json").exists() else {}
        candidates_payload = _load_json(run_dir / "candidates.json") if (run_dir / "candidates.json").exists() else {}
        counts = row.get("candidateCounts") or {}
        if not counts:
            counts = result_payload.get("debug", {}).get("candidateCounts", {}) or candidates_payload.get("candidateCounts", {})
        table_rows.append(
            {
                "config": row.get("config"),
                "success": row.get("success"),
                "release": {"predicted": row.get("releaseSec"), "errorMs": row.get("releaseTimeErrorMs")},
                "bounce": {"predicted": row.get("bounceSec"), "errorMs": row.get("bounceTimeErrorMs")},
                "end": {"predicted": row.get("endSec"), "errorMs": row.get("endTimeErrorMs")},
                "activeStart": {"predicted": row.get("activeStartSec"), "errorMs": row.get("activeStartErrorMs")},
                "activeEnd": {"predicted": row.get("activeEndSec"), "errorMs": row.get("activeEndErrorMs")},
                "speedKph": row.get("speedKph"),
                "speedMph": row.get("speedMph"),
                "speedConfidence": row.get("speedConfidence"),
                "calibrationConfidence": row.get("calibrationConfidence"),
                "bounceDistanceFromBatterStumpsMeters": row.get("bounceDistanceFromBatterStumpsMeters"),
                "lineMetersFromCenter": row.get("lineMetersFromCenter"),
                "lengthCategory": row.get("lengthCategory"),
                "confidence": row.get("confidence"),
                "endpointSource": row.get("endpointSource"),
                "stumpImpactPredictedSec": row.get("stumpImpactPredictedSec"),
                "stumpImpactConfidence": row.get("stumpImpactConfidence"),
                "distanceToStumpRoiPx": row.get("distanceToStumpRoiPx"),
                "processingTimeMs": row.get("processingTimeMs"),
                "decodeTimeMs": row.get("decodeTimeMs"),
                "ballStumpInferenceTimeMs": row.get("ballStumpInferenceTimeMs"),
                "solverTimeMs": row.get("solverTimeMs"),
                "debugWriteTimeMs": row.get("debugWriteTimeMs"),
                "renderVideoTimeMs": row.get("renderVideoTimeMs"),
                "outputVideoSizeMb": row.get("outputVideoSizeMb"),
                "framesProcessed": row.get("framesProcessed"),
                "fpsUsed": row.get("fpsUsed"),
                "deviceUsed": row.get("deviceUsed"),
                "candidateCountsBySource": {
                    "yolo": counts.get("yolo"),
                    "cricket": counts.get("cricket"),
                    "motion": counts.get("motion"),
                    "tracknet": counts.get("tracknet"),
                    "merged": counts.get("merged"),
                },
                "active_window_candidates": counts.get("active_window_candidates"),
                "robust_fit_used_points": counts.get("robust_fit_used_points"),
                "outliers_removed": counts.get("robust_fit_outliers_removed"),
                "yolo_points_used": counts.get("yolo_points_used"),
                "ball_points_used": counts.get("ball_points_used"),
                "motion_points_used": counts.get("motion_points_used"),
                "failureReason": row.get("failureReason", "") or result_payload.get("reason", ""),
            }
        )

    if successful:
        diagnosis = (
            "cricket_ball_v2 now forms detector-supported ball tracklets with real video timestamps. "
            "Rows that include cricket_stumps_v1 can extend the endpoint by projecting the terminal ball path into the stable stump ROI; motion remains support-only."
        )
    else:
        diagnosis = (
            "No valid detector-supported trajectory was produced. Inspect candidate_timeline.csv and tracklet_debug.md for whether ball detections are missing, too sparse, or rejected by tracklet constraints."
        )
    yolo_only_failure = (
        "YOLO context-only configs are expected to fail trajectory solving because YOLO is treated as context, not the primary ball detector. "
        "cricket_ball_v2 is the primary ball source; cricket_stumps_v1 contributes terminal stump geometry when present."
    )

    analysis_json = {
        "benchmarkDir": str(benchmark_dir),
        "successfulCount": len(successful),
        "failedCount": len(failed),
        "rankedSuccessfulConfigs": [row.get("config") for row in ranked_success],
        "bestAccuracyConfig": best_accuracy.get("config") if best_accuracy else None,
        "fastestSuccessfulConfig": fastest.get("config") if fastest else None,
        "bestBalancedConfig": best_balanced.get("config") if best_balanced else None,
        "mostYoloSupportedConfig": most_yolo_supported.get("config") if most_yolo_supported else None,
        "mostMotionDominatedConfig": most_motion_dominated.get("config") if most_motion_dominated else None,
        "diagnosis": diagnosis,
        "yoloOnlyFailureExplanation": yolo_only_failure,
        "rows": table_rows,
    }
    (benchmark_dir / "analysis.json").write_text(json.dumps(analysis_json, indent=2), encoding="utf-8")

    lines = [
        "# FusionTrack Benchmark Analysis",
        "",
        f"- Benchmark dir: `{benchmark_dir}`",
        f"- Successful configs: `{len(successful)}`",
        f"- Failed configs: `{len(failed)}`",
        "",
        "## Highlights",
        f"- Best accuracy config: `{analysis_json['bestAccuracyConfig']}`",
        f"- Fastest successful config: `{analysis_json['fastestSuccessfulConfig']}`",
        f"- Best balanced config: `{analysis_json['bestBalancedConfig']}`",
        f"- Most YOLO-supported config: `{analysis_json['mostYoloSupportedConfig']}`",
        f"- Most motion-dominated config: `{analysis_json['mostMotionDominatedConfig']}`",
        "",
        "## Key Diagnosis",
        f"- {diagnosis}",
        "",
        "## YOLO-only Failure Explanation",
        f"- {yolo_only_failure}",
        "",
        "## Per-config Table",
        "",
        "| Config | Success | Release pred / err(ms) | Bounce pred / err(ms) | End pred / err(ms) | Endpoint source | Speed kph/mph | Speed conf | Cal conf | Time ms | decode/infer/solver/debug/render ms | frames@fps | out MB | device | Failure reason |",
        "|---|:---:|---|---|---|---|---:|---:|---:|---:|---|---|---:|---|---|",
    ]
    for row in table_rows:
        cc = row["candidateCountsBySource"]
        lines.append(
            f"| {row['config']} | {'Y' if row['success'] else 'N'} | "
            f"{_fmt(row['release']['predicted'])} / {_fmt(row['release']['errorMs'])} | "
            f"{_fmt(row['bounce']['predicted'])} / {_fmt(row['bounce']['errorMs'])} | "
            f"{_fmt(row['end']['predicted'])} / {_fmt(row['end']['errorMs'])} | "
            f"{_fmt(row['endpointSource'])} | {_fmt(row['speedKph'])}/{_fmt(row['speedMph'])} | "
            f"{_fmt(row['speedConfidence'])} | {_fmt(row['calibrationConfidence'])} | "
            f"{_fmt(row['processingTimeMs'])} | "
            f"{_fmt(row['decodeTimeMs'])}/{_fmt(row['ballStumpInferenceTimeMs'])}/{_fmt(row['solverTimeMs'])}/{_fmt(row['debugWriteTimeMs'])}/{_fmt(row['renderVideoTimeMs'])} | "
            f"{_fmt(row['framesProcessed'])}@{_fmt(row['fpsUsed'])} | {_fmt(row['outputVideoSizeMb'])} | {_fmt(row['deviceUsed'])} | {_fmt(row['failureReason'])} |"
        )
    (benchmark_dir / "analysis.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
