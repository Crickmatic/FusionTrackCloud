from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Forensic detector pass over a full-rate GT window.")
    parser.add_argument("--clip", default="samples/test video.MOV")
    parser.add_argument("--start-sec", type=float, default=3.20)
    parser.add_argument("--end-sec", type=float, default=4.35)
    parser.add_argument("--output-dir", default="outputs/forensics/latest")
    return parser.parse_args()


def resolve_clip(path: str) -> Path:
    candidates = [Path(path), ROOT / path, REPO_ROOT / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise SystemExit(f"Clip not found: {path}")


def resolve_model_paths() -> dict[str, Path]:
    candidates = {
        "cricket_stumps_v1": [REPO_ROOT / "v1.pt", ROOT / "v1.pt", ROOT / "models" / "cricket_stumps_v1.pt"],
        "cricket_ball_v2": [REPO_ROOT / "v2.pt", ROOT / "v2.pt", ROOT / "models" / "cricket_ball_v2.pt"],
    }
    resolved: dict[str, Path] = {}
    for name, paths in candidates.items():
        path = next((candidate for candidate in paths if candidate.exists()), None)
        if path is None:
            raise SystemExit(f"Required model not found for {name}: {paths}")
        resolved[name] = path
    return resolved


def extract_frames(clip_path: Path, output_dir: Path, start_sec: float, end_sec: float) -> list[dict]:
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(clip_path))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    start_frame = int(math.floor(start_sec * fps))
    end_frame = int(math.ceil(end_sec * fps))
    frame_records: list[dict] = []
    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if start_frame <= frame_index <= end_frame:
            timestamp = frame_index / fps
            filename = f"frame_{frame_index:05d}_{timestamp:.3f}s.jpg"
            path = frames_dir / filename
            cv2.imwrite(str(path), frame)
            frame_records.append(
                {
                    "frameIndex": frame_index,
                    "timestamp": timestamp,
                    "path": path,
                    "width": frame.shape[1],
                    "height": frame.shape[0],
                }
            )
        if frame_index > end_frame:
            break
        frame_index += 1
    capture.release()
    (output_dir / "frames.json").write_text(json.dumps(frame_records, indent=2, default=str), encoding="utf-8")
    return frame_records


def inference_regions(width: int, height: int) -> dict[str, tuple[int, int, int, int]]:
    return {
        "full": (0, 0, width, height),
        "pitch_corridor_crop": (int(width * 0.18), int(height * 0.08), int(width * 0.82), int(height * 0.94)),
        "bounce_corridor_crop": (int(width * 0.25), int(height * 0.35), int(width * 0.75), int(height * 0.90)),
    }


def run_model(model_name: str, model_path: Path, threshold: float, frame_records: list[dict], output_dir: Path) -> list[dict]:
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    run_dir = output_dir / f"{model_name}_{format_conf(threshold)}"
    annotated_dir = run_dir / "annotated_frames"
    annotated_dir.mkdir(parents=True, exist_ok=True)
    detections: list[dict] = []
    class_names = getattr(model, "names", {}) or {}

    for record in frame_records:
        frame = cv2.imread(str(record["path"]))
        if frame is None:
            continue
        height, width = frame.shape[:2]
        annotated = frame.copy()
        for region_name, (x0, y0, x1, y1) in inference_regions(width, height).items():
            crop = frame[y0:y1, x0:x1]
            if crop.size == 0:
                continue
            results = model.predict(crop, verbose=False, conf=threshold)
            for result in results:
                boxes = getattr(result, "boxes", None)
                if boxes is None:
                    continue
                for box in boxes:
                    cls_id = int(box.cls.item())
                    conf = float(box.conf.item())
                    bx1, by1, bx2, by2 = [float(value) for value in box.xyxy.cpu().numpy()[0]]
                    gx1 = bx1 + x0
                    gy1 = by1 + y0
                    gx2 = bx2 + x0
                    gy2 = by2 + y0
                    class_name = str(class_names.get(cls_id, cls_id)) if isinstance(class_names, dict) else str(cls_id)
                    row = {
                        "frameIndex": int(record["frameIndex"]),
                        "timestamp": float(record["timestamp"]),
                        "model": model_name,
                        "threshold": threshold,
                        "inferenceRegion": region_name,
                        "classId": cls_id,
                        "className": class_name,
                        "confidence": conf,
                        "x1": gx1,
                        "y1": gy1,
                        "x2": gx2,
                        "y2": gy2,
                        "bboxWidth": gx2 - gx1,
                        "bboxHeight": gy2 - gy1,
                        "centerX": (gx1 + gx2) / 2.0,
                        "centerY": (gy1 + gy2) / 2.0,
                    }
                    detections.append(row)
                    color = (0, 255, 0) if region_name == "full" else (0, 200, 255)
                    cv2.rectangle(annotated, (int(gx1), int(gy1)), (int(gx2), int(gy2)), color, 2)
                    label = f"{model_name} {class_name} {conf:.2f} {region_name}"
                    cv2.putText(
                        annotated,
                        label,
                        (int(gx1), max(18, int(gy1) - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        color,
                        1,
                    )
        cv2.imwrite(str(annotated_dir / Path(record["path"]).name), annotated)

    write_detection_outputs(run_dir, detections)
    make_contact_sheet(annotated_dir, run_dir / "contact_sheet.jpg", title=f"{model_name} conf={threshold}")
    return detections


def write_detection_outputs(run_dir: Path, detections: list[dict]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "detections.json").write_text(json.dumps(detections, indent=2), encoding="utf-8")
    fields = [
        "frameIndex",
        "timestamp",
        "model",
        "threshold",
        "inferenceRegion",
        "classId",
        "className",
        "confidence",
        "x1",
        "y1",
        "x2",
        "y2",
        "bboxWidth",
        "bboxHeight",
        "centerX",
        "centerY",
    ]
    with (run_dir / "detections.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in detections:
            writer.writerow(row)


def make_contact_sheet(source_dir: Path, output_path: Path, title: str, max_images: int = 48) -> None:
    image_paths = sorted(source_dir.glob("*.jpg"))[:max_images]
    if not image_paths:
        return
    thumbs = []
    for path in image_paths:
        image = cv2.imread(str(path))
        if image is None:
            continue
        image = cv2.resize(image, (240, 135))
        cv2.putText(image, path.stem[:28], (6, 128), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        thumbs.append(image)
    if not thumbs:
        return
    cols = 4
    rows = int(math.ceil(len(thumbs) / cols))
    sheet = np.zeros((rows * 135 + 36, cols * 240, 3), dtype=np.uint8)
    cv2.putText(sheet, title, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    for idx, thumb in enumerate(thumbs):
        row = idx // cols
        col = idx % cols
        y = 36 + row * 135
        x = col * 240
        sheet[y:y + 135, x:x + 240] = thumb
    cv2.imwrite(str(output_path), sheet)


def format_conf(value: float) -> str:
    return f"{value:.2f}".replace(".", "p")


def summarize(output_dir: Path, all_detections: dict[str, dict[float, list[dict]]], frame_records: list[dict]) -> None:
    lines = [
        "# Test Video GT-window Detector Forensics",
        "",
        "- Clip window: `3.20s..4.35s`",
        f"- Full-rate frames extracted: `{len(frame_records)}`",
        "- No active-window, corridor, temporal, tracklet, or solver filtering is applied to the detections below.",
        "- `full` rows are full-frame inference; `pitch_corridor_crop` and `bounce_corridor_crop` rows answer whether crop inference recovers misses.",
        "",
        "## Detection Counts",
        "",
        "| Model | Conf | Total | Full-frame | Crop-only | Classes | Frames with detections | Max conf | Median box px |",
        "|---|---:|---:|---:|---:|---|---:|---:|---|",
    ]
    summary_payload: dict[str, dict] = {"framesExtracted": len(frame_records), "models": {}}
    for model_name, threshold_map in all_detections.items():
        summary_payload["models"][model_name] = {}
        for threshold, detections in threshold_map.items():
            full = [row for row in detections if row["inferenceRegion"] == "full"]
            crop = [row for row in detections if row["inferenceRegion"] != "full"]
            frames = sorted(set(row["frameIndex"] for row in detections))
            classes = sorted(set(str(row["className"]) for row in detections))
            widths = [float(row["bboxWidth"]) for row in detections]
            heights = [float(row["bboxHeight"]) for row in detections]
            median_box = "n/a"
            if widths and heights:
                median_box = f"{float(np.median(widths)):.1f}x{float(np.median(heights)):.1f}"
            max_conf = max([float(row["confidence"]) for row in detections], default=0.0)
            lines.append(
                f"| {model_name} | {threshold:.2f} | {len(detections)} | {len(full)} | {len(crop)} | "
                f"{', '.join(classes) if classes else 'none'} | {len(frames)} | {max_conf:.3f} | {median_box} |"
            )
            summary_payload["models"][model_name][str(threshold)] = {
                "totalDetections": len(detections),
                "fullFrameDetections": len(full),
                "cropDetections": len(crop),
                "framesWithDetections": frames,
                "classes": classes,
                "maxConfidence": max_conf,
                "medianBox": median_box,
            }
    lines.extend(["", "## Initial Diagnosis", ""])
    any_detection = any(detections for threshold_map in all_detections.values() for detections in threshold_map.values())
    if not any_detection:
        lines.append("Current detectors cannot see the ball in this clip.")
    else:
        lines.append("At least one detector emits boxes in the GT window. Use the per-model CSV/JSON and contact sheets to decide whether those boxes are on the cricket ball or wrong objects/classes.")
    lines.extend(
        [
            "",
            "## Recommendations If The Ball Is Not Visible To Detectors",
            "",
            "- Train a cricket-ball detector from this exact camera angle and distance.",
            "- Add synthetic/augmented cricket ball data with motion blur and small-object scaling.",
            "- Capture or infer a high-FPS crop around the pitch corridor before detection.",
            "- Use Watch release prior plus local crop search rather than full-frame-only inference.",
            "- Add a TrackNet-style heatmap model for tiny blurred ball localization.",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")


def make_combined_overlay(output_dir: Path, frame_records: list[dict]) -> None:
    ball_rows = _read_detection_rows(output_dir / "cricket_ball_v2_0p05" / "detections.csv")
    stump_rows = _read_detection_rows(output_dir / "cricket_stumps_v1_0p25" / "detections.csv")
    ball_by_frame: dict[int, list[dict]] = {}
    stump_by_frame: dict[int, list[dict]] = {}
    for row in ball_rows:
        ball_by_frame.setdefault(int(row["frameIndex"]), []).append(row)
    for row in stump_rows:
        stump_by_frame.setdefault(int(row["frameIndex"]), []).append(row)

    target = output_dir / "combined_ball_stump_overlay"
    target.mkdir(parents=True, exist_ok=True)
    for record in frame_records:
        frame = cv2.imread(str(record["path"]))
        if frame is None:
            continue
        frame_index = int(record["frameIndex"])
        for row in stump_by_frame.get(frame_index, []):
            _draw_row(frame, row, (255, 0, 255), "stump")
        for row in ball_by_frame.get(frame_index, []):
            _draw_row(frame, row, (0, 255, 0), "ball")
        cv2.imwrite(str(target / Path(record["path"]).name), frame)
    make_contact_sheet(target, output_dir / "combined_ball_stump_contact_sheet.jpg", "Ball + stump overlay")


def _read_detection_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _draw_row(image: np.ndarray, row: dict, color: tuple[int, int, int], label_prefix: str) -> None:
    x1 = int(float(row["x1"]))
    y1 = int(float(row["y1"]))
    x2 = int(float(row["x2"]))
    y2 = int(float(row["y2"]))
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    label = f"{label_prefix} {float(row['confidence']):.2f} {row['inferenceRegion']}"
    cv2.putText(image, label, (x1, max(18, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)


def main() -> None:
    args = parse_args()
    clip_path = resolve_clip(args.clip)
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_records = extract_frames(clip_path, output_dir, args.start_sec, args.end_sec)
    make_contact_sheet(output_dir / "frames", output_dir / "raw_frames_contact_sheet.jpg", "Raw GT-window frames")

    model_paths = resolve_model_paths()
    thresholds_by_model = {
        "cricket_ball_v2": [0.25, 0.10, 0.05],
        "cricket_stumps_v1": [0.25],
    }
    all_detections: dict[str, dict[float, list[dict]]] = {}
    for model_name, model_path in model_paths.items():
        all_detections[model_name] = {}
        for threshold in thresholds_by_model[model_name]:
            detections = run_model(model_name, model_path, threshold, frame_records, output_dir)
            all_detections[model_name][threshold] = detections

    make_combined_overlay(output_dir, frame_records)

    for threshold in [0.05]:
        sheet_dir = output_dir / f"contact_sheet_conf_{format_conf(threshold)}"
        sheet_dir.mkdir(parents=True, exist_ok=True)
        for model_name in model_paths:
            source = output_dir / f"{model_name}_{format_conf(threshold)}" / "contact_sheet.jpg"
            if source.exists():
                target = sheet_dir / f"{model_name}.jpg"
                image = cv2.imread(str(source))
                if image is not None:
                    cv2.imwrite(str(target), image)
        make_contact_sheet(sheet_dir, output_dir / f"annotated_contact_sheet_conf_{format_conf(threshold)}.jpg", f"Annotated detections conf={threshold}", max_images=5)

    summarize(output_dir, all_detections, frame_records)
    print(f"Forensics complete: {output_dir}")


if __name__ == "__main__":
    main()
