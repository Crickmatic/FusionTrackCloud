from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from app.schemas import Candidate, DeliveryMetadata, DeliveryResult


@dataclass
class RenderArtifacts:
    annotated_video_path: Path
    render_manifest_path: Path
    keyframes_dir: Path
    contact_sheet_path: Path | None
    render_time_ms: float
    frames_rendered: int
    output_video_size_mb: float


class TrajectoryRenderer:
    def render(
        self,
        clip_path: Path,
        result: DeliveryResult,
        metadata: DeliveryMetadata,
        selected_tracklet: list[Candidate],
        merged_candidates: list[Candidate],
        output_path: Path,
        mode: str = "product",
    ) -> RenderArtifacts:
        started = time.perf_counter()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        keyframes_dir = output_path.parent / f"{output_path.stem}_keyframes"
        keyframes_dir.mkdir(parents=True, exist_ok=True)
        self._current_result = result
        capture = cv2.VideoCapture(str(clip_path))
        fps = float(metadata.fps or metadata.extras.get("videoFps") or capture.get(cv2.CAP_PROP_FPS) or 30.0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or metadata.resolution.width if metadata.resolution else 1080)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or metadata.resolution.height if metadata.resolution else 1920)
        writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

        trajectory = sorted(
            (result.finalTrajectory or result.reconstructedTrajectory or result.trajectory2D or []),
            key=lambda point: self._point_t(point),
        )
        detection_times = sorted(self._candidate_time(candidate, metadata, fps) for candidate in selected_tracklet) if selected_tracklet else []
        last_detection_time = detection_times[-1] if detection_times else None
        bounce_t = result.bounceSec or result.bouncePoint.t
        release_t = result.releaseSec or result.releasePoint.t
        impact_t = result.stumpImpactSec or result.endPoint.t
        keyframe_targets = {
            "release": release_t,
            "bounce": bounce_t if result.actualBounceFrame is not None or result.eventSources.get("bounce", "").startswith("rescued") else None,
            "projected_bounce": bounce_t if result.actualBounceFrame is None else None,
            "impact": impact_t if (result.impactSource or "").startswith("detected") else None,
            "projected_impact": impact_t if not (result.impactSource or "").startswith("detected") else None,
        }
        keyframe_target_indices = {
            "release": result.releaseFrame,
            "bounce": result.actualBounceFrame,
            "projected_bounce": result.projectedBounceFrame,
            "impact": result.stumpImpactFrame,
            "projected_impact": result.projectedImpactFrame,
        }
        keyframe_target_indices = {label: index for label, index in keyframe_target_indices.items() if index is not None and keyframe_targets.get(label) is not None}
        keyframe_images: dict[str, np.ndarray] = {}
        frame_idx = 0
        # One canonical on-screen path for both debug and consumer when detections exist.
        # Consumer used to smooth finalTrajectory (physics); debug used tracklet — they diverged.
        if len(selected_tracklet) >= 2:
            self._canonical_arc_xyt = self._build_debug_tracklet_arc(
                selected_tracklet, result, metadata, fps, width, height
            )
        else:
            self._canonical_arc_xyt = None
        if mode == "consumer":
            frame_idx = self._render_consumer_flow(
                capture=capture,
                writer=writer,
                trajectory=trajectory,
                width=width,
                height=height,
                fps=fps,
            )
        else:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                frame_time = (
                    float(metadata.frameTimestamps[frame_idx])
                    if metadata.frameTimestamps and frame_idx < len(metadata.frameTimestamps)
                    else frame_idx / max(1e-6, fps)
                )
                self._draw_scene(
                    frame=frame,
                    frame_time=frame_time,
                    width=width,
                    height=height,
                    result=result,
                    metadata=metadata,
                    trajectory=trajectory,
                    last_detection_time=last_detection_time,
                    selected_tracklet=selected_tracklet,
                    merged_candidates=merged_candidates,
                    mode=mode,
                )
                writer.write(frame)
                for label, target_index in keyframe_target_indices.items():
                    if label in keyframe_images:
                        continue
                    if frame_idx >= target_index:
                        keyframe_images[label] = frame.copy()
                        cv2.imwrite(str(keyframes_dir / f"{label}.png"), frame)
                frame_idx += 1

        capture.release()
        writer.release()
        contact_sheet_path = self._write_contact_sheet(keyframe_images, output_path.parent / f"{output_path.stem}_contact_sheet.jpg")
        output_size_mb = output_path.stat().st_size / (1024 * 1024) if output_path.exists() else 0.0
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        manifest = {
            "annotatedVideoPath": str(output_path),
            "renderMode": mode,
            "renderTimeMs": elapsed_ms,
            "framesRendered": frame_idx,
            "outputVideoSizeMb": output_size_mb,
            "keyframesDir": str(keyframes_dir),
            "contactSheetPath": str(contact_sheet_path) if contact_sheet_path else None,
            "impactSource": result.impactSource,
            "stumpsHitting": result.stumpsHitting,
            "pitchedInLine": result.pitchedInLine,
            "drsDecision": result.drsDecision,
            "virtualPitchCorridor": result.virtualPitchCorridor,
            "eventSources": result.eventSources,
        }
        manifest_path = output_path.parent / f"{output_path.stem}_render_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return RenderArtifacts(
            annotated_video_path=output_path,
            render_manifest_path=manifest_path,
            keyframes_dir=keyframes_dir,
            contact_sheet_path=contact_sheet_path,
            render_time_ms=elapsed_ms,
            frames_rendered=frame_idx,
            output_video_size_mb=output_size_mb,
        )

    def _render_consumer_flow(
        self,
        capture: cv2.VideoCapture,
        writer: cv2.VideoWriter,
        trajectory,
        width: int,
        height: int,
        fps: float,
    ) -> int:
        frames: list[np.ndarray] = []
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
        if not frames:
            return 0

        arc = getattr(self, "_canonical_arc_xyt", None)
        if arc and len(arc) >= 2:
            smooth_points = [(int(x), int(y)) for x, y, _t in arc]
            trajectory_start_t = float(arc[0][2])
            trajectory_end_t = float(arc[-1][2])
        else:
            raw_points = [
                (int(self._point_x(point) * width), int(self._point_y(point) * height), self._point_t(point), self._point_style(point))
                for point in trajectory
            ]
            smooth_points = self._smooth_consumer_points(
                raw_points, bounce_t=(self._current_result.bounceSec if getattr(self, "_current_result", None) else None)
            )
            trajectory_start_t = min((point[2] for point in raw_points), default=0.0)
            trajectory_end_t = max((point[2] for point in raw_points), default=max(1e-6, len(frames) / max(1e-6, fps)))
        frame_count = 0

        # 1. Raw delivery: let the user watch the ball without analysis clutter.
        for frame in frames:
            writer.write(frame)
            frame_count += 1

        # 2. Short processing pause on the last frame.
        freeze = frames[-1].copy()
        processing_frames = max(1, int(round(fps * 1.25)))
        for _ in range(processing_frames):
            frame = freeze.copy()
            self._draw_processing_label(frame)
            writer.write(frame)
            frame_count += 1

        # 3. Final-frame reveal: animate the full DRS path once.
        reveal_frames = max(1, int(round(fps * 1.6)))
        for step in range(reveal_frames):
            frame = freeze.copy()
            progress = (step + 1) / reveal_frames
            self._draw_ar_pitch_overlay(frame, progress=min(1.0, progress * 1.25))
            self._draw_consumer_path(frame, smooth_points, progress=progress)
            if progress > 0.72:
                self._draw_consumer_info_panel(frame)
            writer.write(frame)
            frame_count += 1

        hold_frames = max(1, int(round(fps * 0.45)))
        for _ in range(hold_frames):
            frame = freeze.copy()
            self._draw_ar_pitch_overlay(frame, progress=1.0)
            self._draw_consumer_path(frame, smooth_points, progress=1.0)
            self._draw_consumer_info_panel(frame)
            writer.write(frame)
            frame_count += 1

        # 4. Replay original clip with trajectory drawn in sync with ball movement.
        for replay_index, frame in enumerate(frames):
            output = frame.copy()
            frame_time = replay_index / max(1e-6, fps)
            progress = (frame_time - trajectory_start_t) / max(1e-6, trajectory_end_t - trajectory_start_t)
            if progress > 0.0:
                self._draw_ar_pitch_overlay(output, progress=min(1.0, progress * 1.4))
            self._draw_consumer_path(output, smooth_points, progress=progress)
            if progress > 0.98:
                self._draw_consumer_info_panel(output)
            writer.write(output)
            frame_count += 1
        return frame_count

    def _draw_scene(
        self,
        frame: np.ndarray,
        frame_time: float,
        width: int,
        height: int,
        result: DeliveryResult,
        metadata: DeliveryMetadata,
        trajectory,
        last_detection_time: float | None,
        selected_tracklet: list[Candidate],
        merged_candidates: list[Candidate],
        mode: str,
    ) -> None:
        overlay = frame.copy()
        rs_debug = result.debug.reconstructionStats or {}
        polygon = metadata.corridorGeometry.pitchPolygon or (metadata.guidedBoxes.pitchCorridorNorm if metadata.guidedBoxes else [])
        if mode != "consumer" and len(polygon) >= 4 and self._valid_ground_polygon(polygon):
            pts = np.array([[int(p.x * width), int(p.y * height)] for p in polygon[:4]], dtype=np.int32)
            cv2.polylines(overlay, [pts], isClosed=True, color=(120, 120, 120), thickness=2, lineType=cv2.LINE_AA)
        stump_boxes = result.visualStumpBoxes or result.stumpRois or ({"target": result.stumpRoi} if result.stumpRoi else {})
        if mode != "consumer":
            for key, roi in stump_boxes.items():
                if not roi:
                    continue
                x1 = int(roi["x1"] * width)
                y1 = int(roi["y1"] * height)
                x2 = int(roi["x2"] * width)
                y2 = int(roi["y2"] * height)
                color = (80, 200, 80) if key == "far" else (160, 140, 80)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

        release_cutoff = result.releaseSec or result.releasePoint.t or 0.0
        min_draw_time = 0.0 if mode == "debug" else release_cutoff
        points = [
            (
                int(self._point_x(p) * width),
                int(self._point_y(p) * height),
                self._point_t(p),
                self._point_style(p),
            )
            for p in trajectory
            if min_draw_time <= self._point_t(p) <= frame_time
        ]
        if mode == "consumer":
            consumer_points = self._smooth_consumer_points(points, bounce_t=result.bounceSec)
            if len(consumer_points) >= 2:
                cv2.polylines(
                    overlay,
                    [np.array(consumer_points, dtype=np.int32)],
                    isClosed=False,
                    color=(35, 35, 235),
                    thickness=7,
                    lineType=cv2.LINE_AA,
                )
        elif mode == "debug" and getattr(self, "_canonical_arc_xyt", None):
            arc = [
                (x, y)
                for x, y, t in self._canonical_arc_xyt
                if min_draw_time <= t <= frame_time
            ]
            if len(arc) >= 2:
                cv2.polylines(
                    overlay,
                    [np.array(arc, dtype=np.int32)],
                    isClosed=False,
                    color=(25, 125, 255),
                    thickness=3,
                    lineType=cv2.LINE_AA,
                )
        else:
            for idx in range(1, len(points)):
                x1, y1, t1, style1 = points[idx - 1]
                x2, y2, t2, style2 = points[idx]
                predicted_segment = style1 == "dashed_projected" or style2 == "dashed_projected" or (
                    result.endpointSource == "projected_to_stumps"
                    and last_detection_time is not None
                    and ((t1 + t2) * 0.5) > last_detection_time
                )
                color = (25, 95, 245) if predicted_segment else (25, 125, 255)
                if predicted_segment:
                    self._draw_dashed_line(overlay, (x1, y1), (x2, y2), color, 3)
                else:
                    cv2.line(overlay, (x1, y1), (x2, y2), color, 3, cv2.LINE_AA)

        if mode != "consumer" and (mode == "debug" or frame_time >= release_cutoff):
            self._draw_marker(overlay, result.releasePoint, width, height, (255, 140, 0))
            self._draw_marker(overlay, result.bouncePoint, width, height, (0, 255, 255))
            self._draw_marker(overlay, result.endPoint, width, height, (0, 220, 80))

        if mode == "debug":
            crop_norm = rs_debug.get("bounceNetCropRectNorm")
            if isinstance(crop_norm, dict):
                cx1 = int(float(crop_norm.get("x1", 0)) * width)
                cy1 = int(float(crop_norm.get("y1", 0)) * height)
                cx2 = int(float(crop_norm.get("x2", 0)) * width)
                cy2 = int(float(crop_norm.get("y2", 0)) * height)
                cv2.rectangle(overlay, (cx1, cy1), (cx2, cy2), (200, 60, 255), 2, cv2.LINE_AA)
            bn_pt = rs_debug.get("bounceNetBouncePoint")
            if isinstance(bn_pt, dict):
                bx = int(float(bn_pt.get("x", 0)) * width)
                by = int(float(bn_pt.get("y", 0)) * height)
                cv2.circle(overlay, (bx, by), 10, (200, 60, 255), 2, cv2.LINE_AA)
                cv2.circle(overlay, (bx, by), 3, (255, 255, 255), -1, cv2.LINE_AA)
            for candidate in merged_candidates:
                candidate_time = self._candidate_time(candidate, metadata, frame_time if frame_time > 0 else 30.0)
                if candidate_time <= frame_time and candidate.bbox:
                    x1 = int(candidate.bbox.get("x1", candidate.x * width))
                    y1 = int(candidate.bbox.get("y1", candidate.y * height))
                    x2 = int(candidate.bbox.get("x2", candidate.x * width))
                    y2 = int(candidate.bbox.get("y2", candidate.y * height))
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (200, 140, 80), 1, cv2.LINE_AA)
            for candidate in selected_tracklet:
                candidate_time = self._candidate_time(candidate, metadata, frame_time if frame_time > 0 else 30.0)
                if candidate_time <= frame_time:
                    cv2.circle(overlay, (int(candidate.x * width), int(candidate.y * height)), 4, (0, 255, 255), -1, cv2.LINE_AA)

        cv2.addWeighted(overlay, 0.92, frame, 0.08, 0, frame)
        if mode != "consumer":
            text_lines = [
                f"{(result.deliverySpeedKph or result.speedKph or 0):.1f} kph / {(result.deliverySpeedMph or result.speedMph or 0):.1f} mph",
                f"Length: {result.lengthCategory or result.length}",
                f"Confidence: {result.trajectoryConfidence or result.confidence:.2f}",
                f"Calibration: {result.calibrationMode.value}",
                f"DRS: {(result.drsDecision or {}).get('finalDecision', result.impactSource or result.endpointSource or 'n/a')}",
            ]
            if mode == "debug":
                bn_conf = rs_debug.get("bounceNetConfidence")
                bn_src = rs_debug.get("bounceSource", "")
                if bn_conf is not None:
                    text_lines.append(f"BounceNet: {bn_src or 'n/a'} conf={float(bn_conf):.2f}")
            self._draw_text_box(frame, text_lines, 20, 30)

    def _draw_consumer_path(self, frame: np.ndarray, points: list[tuple[int, int]], progress: float) -> None:
        if len(points) < 2:
            return
        count = max(2, min(len(points), int(round(len(points) * max(0.0, min(1.0, progress))))))
        if progress <= 0.0:
            return
        visible = points[:count]
        glow = frame.copy()
        cv2.polylines(glow, [np.array(visible, dtype=np.int32)], isClosed=False, color=(45, 45, 180), thickness=13, lineType=cv2.LINE_AA)
        cv2.addWeighted(glow, 0.24, frame, 0.76, 0, frame)
        cv2.polylines(
            frame,
            [np.array(visible, dtype=np.int32)],
            isClosed=False,
            color=(45, 45, 245),
            thickness=7,
            lineType=cv2.LINE_AA,
        )
        self._draw_virtual_ball(frame, visible[-1], progress)

    def _draw_ar_pitch_overlay(self, frame: np.ndarray, progress: float) -> None:
        result = getattr(self, "_current_result", None)
        if result is None:
            return
        progress = max(0.0, min(1.0, progress))
        height, width = frame.shape[:2]
        self._draw_perspective_corridor(frame, result, width, height, progress)
        boxes = result.visualStumpBoxes or result.stumpRois or ({"target": result.stumpRoi} if result.stumpRoi else {})
        # Draw far stumps last so the wicket target feels anchored to the trajectory endpoint.
        for key in ("near", "target", "far"):
            roi = boxes.get(key)
            if roi:
                emphasis = 1.0 if key == "far" else 0.58
                self._draw_virtual_stumps(frame, roi, width, height, alpha=progress * emphasis)

    def _draw_perspective_corridor(self, frame: np.ndarray, result: DeliveryResult, width: int, height: int, progress: float) -> None:
        corridor = result.virtualPitchCorridor
        if len(corridor) < 4 or progress <= 0.0:
            return
        pts = np.array([[int(point["x"] * width), int(point["y"] * height)] for point in corridor[:4]], dtype=np.int32)
        overlay = frame.copy()
        fill = np.zeros_like(frame)
        cv2.fillConvexPoly(fill, pts, (95, 58, 145), lineType=cv2.LINE_AA)
        cv2.addWeighted(fill, 0.16 * progress, overlay, 1.0, 0, overlay)

        left_far, right_far, right_near, left_near = pts
        bands = 9
        for idx in range(1, bands):
            alpha = idx / bands
            left = (left_far * (1.0 - alpha) + left_near * alpha).astype(int)
            right = (right_far * (1.0 - alpha) + right_near * alpha).astype(int)
            purple_mix = 1.0 - abs(0.5 - alpha) * 1.55
            color = (
                int(210 - 90 * purple_mix),
                int(215 - 145 * purple_mix),
                int(255),
            )
            cv2.line(overlay, tuple(left), tuple(right), color, 1, cv2.LINE_AA)

        rail = (245, 240, 255)
        cv2.polylines(overlay, [np.array([left_far, left_near], dtype=np.int32)], False, rail, 2, cv2.LINE_AA)
        cv2.polylines(overlay, [np.array([right_far, right_near], dtype=np.int32)], False, rail, 2, cv2.LINE_AA)
        center_far = ((left_far + right_far) * 0.5).astype(int)
        center_near = ((left_near + right_near) * 0.5).astype(int)
        self._draw_dashed_line(overlay, tuple(center_far), tuple(center_near), (245, 245, 255), 1)
        self._draw_crease_guides(overlay, left_far, right_far, left_near, right_near)
        cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)

    def _draw_crease_guides(self, frame: np.ndarray, left_far: np.ndarray, right_far: np.ndarray, left_near: np.ndarray, right_near: np.ndarray) -> None:
        for alpha, thickness in ((0.0, 2), (0.12, 1), (0.88, 1), (1.0, 2)):
            left = (left_far * (1.0 - alpha) + left_near * alpha).astype(int)
            right = (right_far * (1.0 - alpha) + right_near * alpha).astype(int)
            cv2.line(frame, tuple(left), tuple(right), (250, 250, 255), thickness, cv2.LINE_AA)
        for alpha in (0.03, 0.97):
            left = (left_far * (1.0 - alpha) + left_near * alpha).astype(int)
            right = (right_far * (1.0 - alpha) + right_near * alpha).astype(int)
            center = ((left + right) * 0.5).astype(int)
            span = right - left
            extension = span * 0.42
            cv2.line(frame, tuple((center - extension).astype(int)), tuple((center + extension).astype(int)), (220, 220, 255), 1, cv2.LINE_AA)

    def _draw_virtual_stumps(self, frame: np.ndarray, roi: dict[str, float], width: int, height: int, alpha: float) -> None:
        alpha = max(0.0, min(1.0, alpha))
        if alpha <= 0.0:
            return
        x1 = int(float(roi["x1"]) * width)
        y1 = int(float(roi["y1"]) * height)
        x2 = int(float(roi["x2"]) * width)
        y2 = int(float(roi["y2"]) * height)
        if x2 <= x1 or y2 <= y1:
            return
        overlay = frame.copy()
        stump_width = max(2, int(round((x2 - x1) * 0.16)))
        inset = max(2, int(round((x2 - x1) * 0.16)))
        xs = np.linspace(x1 + inset, x2 - inset, 3)
        top_y = y1 + int((y2 - y1) * 0.05)
        bottom_y = y2
        glow = overlay.copy()
        for x in xs:
            start = (int(round(x)), top_y)
            end = (int(round(x)), bottom_y)
            cv2.line(glow, start, end, (20, 165, 255), stump_width + 7, cv2.LINE_AA)
        cv2.addWeighted(glow, 0.20 * alpha, overlay, 1.0 - 0.20 * alpha, 0, overlay)
        for x in xs:
            start = (int(round(x)), top_y)
            end = (int(round(x)), bottom_y)
            cv2.line(overlay, start, end, (24, 150, 255), stump_width + 1, cv2.LINE_AA)
            cv2.line(overlay, (start[0] - max(1, stump_width // 3), start[1]), (end[0] - max(1, stump_width // 3), end[1]), (235, 245, 255), max(1, stump_width // 3), cv2.LINE_AA)
            cv2.line(overlay, (start[0] + max(1, stump_width // 3), start[1] + 1), (end[0] + max(1, stump_width // 3), end[1]), (5, 70, 150), 1, cv2.LINE_AA)
        bail_y = top_y + max(1, int((y2 - y1) * 0.05))
        cv2.line(overlay, (int(xs[0]), bail_y), (int(xs[-1]), bail_y), (240, 248, 255), max(2, stump_width // 2), cv2.LINE_AA)
        cv2.ellipse(overlay, ((x1 + x2) // 2, bottom_y), (max(4, (x2 - x1) // 2), max(2, (y2 - y1) // 18)), 0, 0, 360, (20, 70, 145), 2, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.82 * alpha, frame, 1.0 - 0.82 * alpha, 0, frame)

    def _draw_virtual_ball(self, frame: np.ndarray, center: tuple[int, int], progress: float) -> None:
        if progress <= 0.0:
            return
        radius = max(5, int(round(7 + 3 * min(1.0, progress))))
        x, y = center
        shadow = frame.copy()
        cv2.ellipse(shadow, (x + radius // 3, y + radius), (radius + 3, max(3, radius // 2)), 0, 0, 360, (20, 20, 25), -1, cv2.LINE_AA)
        cv2.addWeighted(shadow, 0.16, frame, 0.84, 0, frame)
        cv2.circle(frame, (x, y), radius + 4, (235, 235, 255), 2, cv2.LINE_AA)
        cv2.circle(frame, (x, y), radius, (242, 242, 238), -1, cv2.LINE_AA)
        cv2.circle(frame, (x - radius // 3, y - radius // 3), max(2, radius // 3), (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, (x + radius // 3, y + radius // 3), max(1, radius // 4), (150, 150, 150), -1, cv2.LINE_AA)

    def _draw_processing_label(self, frame: np.ndarray) -> None:
        height, width = frame.shape[:2]
        text = "Processing trajectory..."
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.8
        thickness = 2
        (text_width, text_height), _ = cv2.getTextSize(text, font, scale, thickness)
        x = max(20, (width - text_width) // 2)
        y = max(48, height - 90)
        pad = 16
        cv2.rectangle(frame, (x - pad, y - text_height - pad), (x + text_width + pad, y + pad), (15, 15, 15), -1)
        cv2.rectangle(frame, (x - pad, y - text_height - pad), (x + text_width + pad, y + pad), (80, 80, 80), 1)
        cv2.putText(frame, text, (x, y), font, scale, (245, 245, 245), thickness, cv2.LINE_AA)

    def _draw_consumer_info_panel(self, frame: np.ndarray) -> None:
        # This method is patched at runtime by assigning result in render scope.
        result = getattr(self, "_current_result", None)
        if result is None:
            return
        drs = result.drsDecision or {}
        pitching = drs.get("pitching", {}) if isinstance(drs.get("pitching"), dict) else {}
        impact = drs.get("impact", {}) if isinstance(drs.get("impact"), dict) else {}
        wicket = drs.get("wicketHitting", {}) if isinstance(drs.get("wicketHitting"), dict) else {}
        pitch_label = str(pitching.get("zone") or ("in_line" if result.pitchedInLine else "outside_line")).replace("_", " ").title()
        impact_label = str(impact.get("zone") or "not_detected").replace("_", " ").title()
        wicket_label = str(wicket.get("zone") or ("hitting" if result.stumpsHitting else "missing")).replace("_", " ").title()
        final_decision = str(drs.get("finalDecision") or ("Out" if result.stumpsHitting else "Not Out"))
        lines = [
            f"Speed: {(result.deliverySpeedKph or result.speedKph or 0):.1f} kph",
            f"Length: {result.lengthCategory or result.length}",
            f"Pitching: {pitch_label}",
            f"Impact: {impact_label}",
            f"Wicket Hitting: {wicket_label}",
            f"Decision: {final_decision}",
        ]
        x = 22
        y = 28
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.62
        thickness = 2
        line_height = 24
        width = max(cv2.getTextSize(line, font, scale, thickness)[0][0] for line in lines) + 24
        height = line_height * len(lines) + 18
        panel = frame.copy()
        cv2.rectangle(panel, (x, y), (x + width, y + height), (12, 15, 20), -1)
        cv2.addWeighted(panel, 0.72, frame, 0.28, 0, frame)
        border = (65, 180, 75) if result.stumpsHitting else (50, 80, 220)
        cv2.rectangle(frame, (x, y), (x + width, y + height), border, 2, cv2.LINE_AA)
        for idx, line in enumerate(lines):
            color = (240, 245, 245)
            if line.startswith("Decision") or line.startswith("Wicket"):
                color = (90, 235, 110) if result.stumpsHitting else (95, 140, 250)
            if line.startswith("Impact") and "Not Detected" in line:
                color = (225, 205, 105)
            cv2.putText(frame, line, (x + 12, y + 26 + idx * line_height), font, scale, color, thickness, cv2.LINE_AA)

    def _point_x(self, point) -> float:
        return float(point.get("x", 0.0)) if isinstance(point, dict) else float(point.x)

    def _point_y(self, point) -> float:
        return float(point.get("y", 0.0)) if isinstance(point, dict) else float(point.y)

    def _point_t(self, point) -> float:
        if isinstance(point, dict):
            return float(point.get("t", 0.0) or 0.0)
        return float(point.t or 0.0)

    def _point_style(self, point) -> str:
        return str(point.get("renderStyle", "solid_observed")) if isinstance(point, dict) else "solid_observed"

    def _smooth_consumer_points(self, points: list[tuple[int, int, float, str]], bounce_t: float | None = None) -> list[tuple[int, int]]:
        if len(points) <= 4:
            return [(x, y) for x, y, _t, _style in points]
        ordered = sorted(points, key=lambda point: point[2])
        if bounce_t is not None:
            bounce_point = min(ordered, key=lambda point: abs(point[2] - float(bounce_t)))
            pre = [point for point in ordered if point[2] <= bounce_point[2]]
            post = [point for point in ordered if point[2] >= bounce_point[2]]
            if pre and post:
                pre_curve = self._fit_segment_curve(pre + [bounce_point])
                post_curve = self._fit_segment_curve([bounce_point] + post)
                if pre_curve and post_curve:
                    return pre_curve + post_curve[1:]
        segments: list[list[tuple[int, int, float, str]]] = []
        current: list[tuple[int, int, float, str]] = []
        current_projected = ordered[0][3] == "dashed_projected"
        for point in ordered:
            projected = point[3] == "dashed_projected"
            if current and projected != current_projected:
                segments.append(current)
                current = []
            current.append(point)
            current_projected = projected
        if current:
            segments.append(current)

        smoothed: list[tuple[int, int]] = []
        for segment in segments:
            curve = self._fit_segment_curve(segment)
            if smoothed and curve:
                curve = curve[1:]
            smoothed.extend(curve)
        return smoothed

    def _fit_segment_curve(self, segment: list[tuple[int, int, float, str]]) -> list[tuple[int, int]]:
        if len(segment) <= 3:
            return [(x, y) for x, y, _t, _style in segment]
        filtered = self._remove_spatial_jitter(segment)
        if len(filtered) <= 3:
            return [(x, y) for x, y, _t, _style in filtered]
        controls = self._consumer_control_points(filtered)
        return self._catmull_rom_curve(controls, samples_per_span=18)

    def _consumer_control_points(self, segment: list[tuple[int, int, float, str]]) -> list[tuple[int, int]]:
        if len(segment) <= 5:
            return [(x, y) for x, y, _t, _style in segment]
        # Customer path should be a model curve, not a per-frame trace.
        # Keep endpoints and a few robust temporal quantiles only.
        quantiles = [0.0, 0.28, 0.58, 0.82, 1.0]
        controls: list[tuple[int, int]] = []
        for q in quantiles:
            index = int(round(q * (len(segment) - 1)))
            if q == 0.0:
                controls.append((segment[0][0], segment[0][1]))
            elif q == 1.0:
                controls.append((segment[-1][0], segment[-1][1]))
            else:
                window_start = max(0, index - 2)
                window_end = min(len(segment), index + 3)
                window = segment[window_start:window_end]
                xs = sorted(point[0] for point in window)
                ys = sorted(point[1] for point in window)
                controls.append((int(xs[len(xs) // 2]), int(ys[len(ys) // 2])))
        deduped: list[tuple[int, int]] = []
        for point in controls:
            if not deduped or point != deduped[-1]:
                deduped.append(point)
        return deduped

    def _catmull_rom_curve(self, controls: list[tuple[int, int]], samples_per_span: int) -> list[tuple[int, int]]:
        if len(controls) <= 2:
            return controls
        pts = [controls[0], *controls, controls[-1]]
        output: list[tuple[int, int]] = []
        for idx in range(1, len(pts) - 2):
            p0 = np.array(pts[idx - 1], dtype=float)
            p1 = np.array(pts[idx], dtype=float)
            p2 = np.array(pts[idx + 1], dtype=float)
            p3 = np.array(pts[idx + 2], dtype=float)
            for step in range(samples_per_span):
                t = step / float(samples_per_span)
                t2 = t * t
                t3 = t2 * t
                point = 0.5 * (
                    (2.0 * p1)
                    + (-p0 + p2) * t
                    + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
                    + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
                )
                out = (int(round(point[0])), int(round(point[1])))
                if not output or out != output[-1]:
                    output.append(out)
        if output[-1] != controls[-1]:
            output.append(controls[-1])
        return output

    def _remove_spatial_jitter(self, segment: list[tuple[int, int, float, str]]) -> list[tuple[int, int, float, str]]:
        if len(segment) < 5:
            return segment
        kept = [segment[0]]
        for index in range(1, len(segment) - 1):
            prev = np.array([segment[index - 1][0], segment[index - 1][1]], dtype=float)
            curr = np.array([segment[index][0], segment[index][1]], dtype=float)
            nxt = np.array([segment[index + 1][0], segment[index + 1][1]], dtype=float)
            expected = (prev + nxt) * 0.5
            deviation = float(np.linalg.norm(curr - expected))
            local_span = float(np.linalg.norm(nxt - prev))
            if deviation <= max(10.0, local_span * 0.65):
                kept.append(segment[index])
        kept.append(segment[-1])
        return kept

    def _valid_ground_polygon(self, polygon) -> bool:
        ys = [point.y for point in polygon[:4]]
        if min(ys) < 0.0 or max(ys) > 1.0:
            return False
        near = (polygon[2].y + polygon[3].y) * 0.5
        far = (polygon[0].y + polygon[1].y) * 0.5
        return near > far

    def _draw_marker(self, frame: np.ndarray, point, width: int, height: int, color: tuple[int, int, int]) -> None:
        if point.t is None:
            return
        center = (int(point.x * width), int(point.y * height))
        cv2.circle(frame, center, 10, color, 2, cv2.LINE_AA)
        cv2.circle(frame, center, 4, color, -1, cv2.LINE_AA)

    def _draw_text_box(self, frame: np.ndarray, lines: list[str], x: int, y: int) -> None:
        line_height = 24
        width = max(cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0][0] for line in lines) + 20
        height = line_height * len(lines) + 14
        cv2.rectangle(frame, (x, y), (x + width, y + height), (15, 15, 15), -1)
        cv2.rectangle(frame, (x, y), (x + width, y + height), (80, 80, 80), 1)
        for idx, line in enumerate(lines):
            cv2.putText(frame, line, (x + 10, y + 24 + idx * line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (235, 235, 235), 2, cv2.LINE_AA)

    def _draw_dashed_line(self, frame: np.ndarray, p1: tuple[int, int], p2: tuple[int, int], color, thickness: int) -> None:
        dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        if dist < 1:
            return
        dash = 10
        gap = 6
        dx = (p2[0] - p1[0]) / dist
        dy = (p2[1] - p1[1]) / dist
        pos = 0.0
        while pos < dist:
            start = (int(p1[0] + dx * pos), int(p1[1] + dy * pos))
            end_pos = min(dist, pos + dash)
            end = (int(p1[0] + dx * end_pos), int(p1[1] + dy * end_pos))
            cv2.line(frame, start, end, color, thickness, cv2.LINE_AA)
            pos += dash + gap

    def _build_debug_tracklet_arc(
        self,
        selected_tracklet: list[Candidate],
        result: DeliveryResult,
        metadata: DeliveryMetadata,
        fps: float,
        width: int,
        height: int,
    ) -> list[tuple[int, int, float]]:
        """Single smooth arc through yellow tracklet samples (debug overlay)."""
        ordered = sorted(
            selected_tracklet,
            key=lambda c: self._candidate_time(c, metadata, fps),
        )
        if len(ordered) < 2:
            return []
        end_t = float(result.stumpImpactSec or result.endPoint.t or 0.0)
        last_t = max(end_t, self._candidate_time(ordered[-1], metadata, fps))
        t0 = self._candidate_time(ordered[0], metadata, fps)
        samples: list[tuple[int, int, float, str]] = []
        for c in ordered:
            t = self._candidate_time(c, metadata, fps)
            samples.append((int(c.x * width), int(c.y * height), t, "solid"))
        samples.append((int(result.endPoint.x * width), int(result.endPoint.y * height), last_t, "solid"))
        xy_smooth = self._smooth_consumer_points(samples, bounce_t=result.bounceSec)
        if len(xy_smooth) < 2:
            return []
        times = np.linspace(t0, last_t, len(xy_smooth))
        return [(xy_smooth[i][0], xy_smooth[i][1], float(times[i])) for i in range(len(xy_smooth))]

    def _candidate_time(self, candidate: Candidate, metadata: DeliveryMetadata, fps: float) -> float:
        if candidate.timestampSec is not None:
            return candidate.timestampSec
        if metadata.frameTimestamps and 0 <= candidate.frameIndex < len(metadata.frameTimestamps):
            return metadata.frameTimestamps[candidate.frameIndex]
        return candidate.frameIndex / max(1e-6, fps)

    def _write_contact_sheet(self, keyframe_images: dict[str, np.ndarray], target_path: Path) -> Path | None:
        ordered = [keyframe_images[label] for label in ("release", "bounce", "projected_bounce", "impact", "projected_impact") if label in keyframe_images]
        if not ordered:
            return None
        max_h = max(image.shape[0] for image in ordered)
        resized = []
        for image in ordered:
            scale = max_h / image.shape[0]
            resized.append(cv2.resize(image, (int(image.shape[1] * scale), max_h)))
        sheet = cv2.hconcat(resized)
        cv2.imwrite(str(target_path), sheet)
        return target_path
