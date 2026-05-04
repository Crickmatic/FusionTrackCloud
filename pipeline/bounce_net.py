from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.schemas import Candidate, DeliveryMetadata


@dataclass
class BounceNetResult:
    """Vision-assisted bounce localization from temporal motion in a pitch-aligned crop."""

    bounce_frame: int | None
    bounce_point_norm: tuple[float, float] | None
    bounce_confidence: float
    post_bounce_direction: tuple[float, float] | None
    source: str
    fallback_reason: str = ""
    window_frame_indices: list[int] = field(default_factory=list)
    crop_rect_norm: tuple[float, float, float, float] | None = None
    heatmap_path: str | None = None
    heatmap_overlay_path: str | None = None
    crop_montage_path: str | None = None
    debug_notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "bounceNetBounceFrame": self.bounce_frame,
            "bounceNetBouncePoint": (
                {"x": self.bounce_point_norm[0], "y": self.bounce_point_norm[1]} if self.bounce_point_norm else None
            ),
            "bounceNetConfidence": self.bounce_confidence,
            "bounceNetPostDirection": (
                {"dx": self.post_bounce_direction[0], "dy": self.post_bounce_direction[1]}
                if self.post_bounce_direction
                else None
            ),
            "bounceSource": self.source,
            "bounceFallbackReason": self.fallback_reason,
            "bounceNetWindowFrames": self.window_frame_indices,
            "bounceNetCropRectNorm": (
                {"x1": self.crop_rect_norm[0], "y1": self.crop_rect_norm[1], "x2": self.crop_rect_norm[2], "y2": self.crop_rect_norm[3]}
                if self.crop_rect_norm
                else None
            ),
            "bounceNetHeatmapPath": self.heatmap_path,
            "bounceNetHeatmapOverlayPath": self.heatmap_overlay_path,
            "bounceNetCropPath": self.crop_montage_path,
            "bounceNetDebugNotes": self.debug_notes,
        }


class BounceNetEstimator:
    """
    Heatmap-style bounce localizer using frame differencing + temporal turning cues.
    Trainable CNN can replace _motion_heatmap_stack; interface stays the same.
    """

    WINDOW_HALF = 12
    MIN_WINDOW = 8
    CONFIDENCE_ACCEPT = 0.42

    def estimate(
        self,
        frames: list[np.ndarray],
        frame_indices: list[int],
        predicted_bounce_frame_num: int | None,
        metadata: DeliveryMetadata,
        candidates: list[Candidate],
        kalman_points: list[Candidate],
        frame_width: int,
        frame_height: int,
        artifact_dir: Path | None = None,
    ) -> BounceNetResult:
        notes: list[str] = []
        if not frames or not frame_indices or len(frames) != len(frame_indices):
            return BounceNetResult(
                bounce_frame=None,
                bounce_point_norm=None,
                bounce_confidence=0.0,
                post_bounce_direction=None,
                source="physics_fallback",
                fallback_reason="no_frames",
                debug_notes=notes,
            )

        center_i = self._center_list_index(predicted_bounce_frame_num, frame_indices)
        i0 = max(0, center_i - self.WINDOW_HALF)
        i1 = min(len(frames), center_i + self.WINDOW_HALF + 1)
        if i1 - i0 < self.MIN_WINDOW:
            i0 = max(0, center_i - self.WINDOW_HALF - 2)
            i1 = min(len(frames), i0 + self.MIN_WINDOW)

        window_frames = frames[i0:i1]
        window_indices = frame_indices[i0:i1]
        notes.append(f"window_list_{i0}:{i1} video_frames={window_indices[0]}..{window_indices[-1]}")

        crop = self._pitch_aligned_crop_rect_norm(metadata, candidates, kalman_points, window_indices)
        if crop is None:
            return BounceNetResult(
                bounce_frame=None,
                bounce_point_norm=None,
                bounce_confidence=0.0,
                post_bounce_direction=None,
                source="physics_fallback",
                fallback_reason="no_crop_geometry",
                window_frame_indices=window_indices,
                debug_notes=notes,
            )

        x1n, y1n, x2n, y2n = crop
        x1, y1 = int(x1n * frame_width), int(y1n * frame_height)
        x2, y2 = int(x2n * frame_width), int(y2n * frame_height)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame_width, x2), min(frame_height, y2)
        if x2 - x1 < 32 or y2 - y1 < 24:
            return BounceNetResult(
                bounce_frame=None,
                bounce_point_norm=None,
                bounce_confidence=0.0,
                post_bounce_direction=None,
                source="physics_fallback",
                fallback_reason="crop_too_small",
                window_frame_indices=window_indices,
                crop_rect_norm=crop,
                debug_notes=notes,
            )

        grays = []
        for fr in window_frames:
            roi = fr[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            g = cv2.resize(g, (0, 0), fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
            grays.append(g)

        if len(grays) < 4:
            return BounceNetResult(
                bounce_frame=None,
                bounce_point_norm=None,
                bounce_confidence=0.0,
                post_bounce_direction=None,
                source="physics_fallback",
                fallback_reason="insufficient_gray_frames",
                window_frame_indices=window_indices,
                crop_rect_norm=crop,
                debug_notes=notes,
            )

        motion_stack = self._motion_stack(grays)
        fused_heatmap, per_frame_hm = self._heatmap_cnn_forward(motion_stack)
        cx, cy, bounce_gray_idx, conf, reason, bounce_motion_idx = self._localize_from_cnn_and_motion(
            motion_stack, fused_heatmap, per_frame_hm, notes
        )

        if bounce_gray_idx is None or conf < 0.12:
            return BounceNetResult(
                bounce_frame=None,
                bounce_point_norm=None,
                bounce_confidence=float(conf),
                post_bounce_direction=None,
                source="physics_fallback",
                fallback_reason=reason or "low_motion_confidence",
                window_frame_indices=window_indices,
                crop_rect_norm=crop,
                debug_notes=notes,
            )

        h, w = grays[0].shape[:2]
        u = float(np.clip(cx / max(1, w), 0.0, 1.0))
        v = float(np.clip(cy / max(1, h), 0.0, 1.0))
        x_norm = x1n + u * (x2n - x1n)
        y_norm = y1n + v * (y2n - y1n)
        bounce_video_frame = window_indices[int(np.clip(bounce_gray_idx, 0, len(window_indices) - 1))]

        post_dx, post_dy = self._post_direction_from_centroids(motion_stack, int(bounce_motion_idx))

        # Heatmap / motion montage PNGs were removed: fused maps are often near-uniform after
        # min–max normalize (low contrast), and frame-differencing thumbnails read as blank without
        # per-clip gain. Bounce inference still uses the same tensors internally.
        heatmap_path = None
        heatmap_overlay_path = None
        montage_path = None

        final_conf = float(np.clip(conf, 0.0, 0.95))
        src = "bounce_net" if final_conf >= self.CONFIDENCE_ACCEPT else "physics_fallback"

        return BounceNetResult(
            bounce_frame=int(bounce_video_frame),
            bounce_point_norm=(float(np.clip(x_norm, 0.0, 1.0)), float(np.clip(y_norm, 0.0, 1.0))),
            bounce_confidence=final_conf,
            post_bounce_direction=(post_dx, post_dy) if post_dx is not None else None,
            source=src,
            fallback_reason="" if src == "bounce_net" else "below_confidence_threshold",
            window_frame_indices=window_indices,
            crop_rect_norm=crop,
            heatmap_path=heatmap_path,
            heatmap_overlay_path=heatmap_overlay_path,
            crop_montage_path=montage_path,
            debug_notes=notes,
        )

    def _center_list_index(self, predicted_video_frame: int | None, frame_indices: list[int]) -> int:
        if predicted_video_frame is None:
            return len(frame_indices) // 2
        return int(np.argmin([abs(int(fi) - int(predicted_video_frame)) for fi in frame_indices]))

    def _pitch_aligned_crop_rect_norm(
        self,
        metadata: DeliveryMetadata,
        candidates: list[Candidate],
        kalman_points: list[Candidate],
        window_indices: list[int],
    ) -> tuple[float, float, float, float] | None:
        polygon = metadata.corridorGeometry.pitchPolygon
        xs_k: list[float] = []
        ys_k: list[float] = []
        win_set = set(window_indices)
        for c in kalman_points:
            if c.frameIndex in win_set:
                xs_k.append(float(c.x))
                ys_k.append(float(c.y))
        for c in candidates:
            if c.modelRole == "ball_detector" and c.frameIndex in win_set and float(c.confidence or 0) >= 0.06:
                xs_k.append(float(c.x))
                ys_k.append(float(c.y))

        cx = float(np.median(xs_k)) if xs_k else 0.5
        cy = float(np.median(ys_k)) if ys_k else 0.55

        if len(polygon) >= 4:
            far_y = float(np.mean([polygon[0].y, polygon[1].y]))
            near_y = float(np.mean([polygon[2].y, polygon[3].y]))
            left_x = min(float(polygon[0].x), float(polygon[3].x))
            right_x = max(float(polygon[1].x), float(polygon[2].x))
            y_top = min(far_y, near_y) - 0.04
            y_bot = max(far_y, near_y) + 0.12
        else:
            left_x, right_x = 0.2, 0.85
            y_top, y_bot = 0.35, 0.92

        half_w = 0.20
        x1 = float(np.clip(cx - half_w, left_x - 0.05, right_x + 0.05))
        x2 = float(np.clip(cx + half_w, left_x - 0.05, right_x + 0.05))
        y1 = float(np.clip(y_top, 0.08, 0.88))
        y2 = float(np.clip(max(y_bot, cy + 0.08), y1 + 0.12, 0.98))
        if x2 - x1 < 0.08:
            x1, x2 = max(0.0, cx - 0.12), min(1.0, cx + 0.12)
        return (x1, y1, x2, y2)

    def _motion_stack(self, grays: list[np.ndarray]) -> list[np.ndarray]:
        out: list[np.ndarray] = []
        for i in range(1, len(grays) - 1):
            a = cv2.absdiff(grays[i], grays[i - 1])
            b = cv2.absdiff(grays[i + 1], grays[i])
            m = cv2.max(a, b)
            m = cv2.GaussianBlur(m, (3, 3), 0)
            out.append(m.astype(np.float32))
        return out

    def _cnn_block(self, x: np.ndarray) -> np.ndarray:
        """Single 'layer': Laplacian + ReLU + avg-pool (approximates a tiny conv block)."""
        p99 = float(np.percentile(x, 99.0) + 1e-6)
        t = (x / p99).astype(np.float32)
        k_lap = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float32)
        k3 = np.ones((3, 3), dtype=np.float32) / 9.0
        k5 = np.ones((5, 5), dtype=np.float32) / 25.0
        feat = cv2.filter2D(t, -1, k_lap)
        feat = np.maximum(feat, 0.0)
        feat = cv2.filter2D(feat, -1, k3)
        feat = np.maximum(feat, 0.0)
        feat = cv2.filter2D(feat, -1, k5)
        feat = np.maximum(feat, 0.0)
        return feat.astype(np.float32)

    def _heatmap_cnn_forward(self, motion_stack: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """
        Multi-frame heatmap head: per-frame conv features + temporal softmax fusion.
        Returns fused HxW heatmap and TxHxW per-frame activations (trainable CNN can replace this).
        """
        if not motion_stack:
            z = np.zeros((1, 1), dtype=np.float32)
            return z, z.reshape(1, 1, 1)
        per_frame: list[np.ndarray] = [self._cnn_block(m) for m in motion_stack]
        P = np.stack(per_frame, axis=0)
        h, w = P.shape[1], P.shape[2]
        bottom = np.zeros((h, w), dtype=np.float32)
        bottom[int(0.42 * h) :, :] = 1.0

        bottom_scores = np.array([float(np.mean(P[t] * bottom)) for t in range(P.shape[0])], dtype=np.float32)
        top_scores = np.array([float(np.percentile(P[t], 99.0)) for t in range(P.shape[0])], dtype=np.float32)
        scores = 0.72 * bottom_scores + 0.28 * top_scores
        scores = scores - float(np.max(scores))
        w_t = np.exp(scores * 4.5)
        w_t = w_t / (float(np.sum(w_t)) + 1e-9)

        fused = np.zeros((h, w), dtype=np.float32)
        for t in range(P.shape[0]):
            fused += w_t[t] * P[t]
        temporal_max = np.max(P, axis=0)
        fused = 0.58 * fused + 0.42 * temporal_max
        fused = cv2.GaussianBlur(fused, (5, 5), 0)
        return fused.astype(np.float32), P

    def _localize_from_cnn_and_motion(
        self,
        motion_stack: list[np.ndarray],
        fused_heatmap: np.ndarray,
        per_frame_hm: np.ndarray,
        notes: list[str],
    ) -> tuple[float, float, int | None, float, str, int]:
        """
        Returns crop-space (cx, cy), gray-frame index into window, confidence, reason, motion_stack index.
        """
        if not motion_stack or per_frame_hm.size == 0:
            return 0.0, 0.0, None, 0.0, "empty_motion", 0

        h, w = motion_stack[0].shape[:2]
        bottom = np.zeros((h, w), dtype=np.float32)
        bottom[int(0.40 * h) :, :] = 1.0

        centroids: list[tuple[float, float]] = []
        for m in motion_stack:
            M = cv2.moments(m)
            if M["m00"] < 1e-3:
                centroids.append((w * 0.5, h * 0.5))
            else:
                centroids.append((float(M["m10"] / M["m00"]), float(M["m01"] / M["m00"])))

        cy_arr = np.array([c[1] for c in centroids], dtype=float)
        cy_s = cv2.GaussianBlur(cy_arr.reshape(-1, 1), (0, 0), sigmaX=1.15).flatten()
        depth_scores = cy_s / max(1e-3, float(np.max(cy_s)))

        cnn_bottom = np.array(
            [float(np.mean(per_frame_hm[t] * bottom)) for t in range(per_frame_hm.shape[0])],
            dtype=np.float32,
        )
        cnn_bottom = cnn_bottom / max(1e-3, float(np.max(cnn_bottom)))

        combined = 0.55 * cnn_bottom + 0.35 * depth_scores + 0.10 * np.array(
            [float(np.percentile(motion_stack[t], 98.0)) for t in range(len(motion_stack))],
            dtype=np.float32,
        )
        bounce_motion_idx = int(np.argmax(combined))
        notes.append(f"cnn_bounce_motion_idx={bounce_motion_idx} combined_peak={float(np.max(combined)):.4f}")

        turning = False
        if len(cy_s) >= 3:
            dy = np.diff(cy_s)
            pre_slope = float(np.mean(dy[max(0, bounce_motion_idx - 2) : bounce_motion_idx])) if bounce_motion_idx >= 1 else 0.0
            post_slope = (
                float(np.mean(dy[bounce_motion_idx : min(len(dy), bounce_motion_idx + 2)]))
                if bounce_motion_idx < len(dy)
                else 0.0
            )
            turning = pre_slope > 0.015 and post_slope < -0.008
            notes.append(f"turning={turning} pre/post_slope={pre_slope:.4f}/{post_slope:.4f}")

        m = motion_stack[bounce_motion_idx]
        m_n = m / max(1e-6, float(np.percentile(m, 99.0)))
        f_n = fused_heatmap / max(1e-6, float(np.percentile(fused_heatmap, 99.0)))
        blend = (0.38 * f_n + 0.62 * m_n) * bottom
        peak = float(np.max(blend))
        if peak < 1e-5:
            iy, ix = np.unravel_index(int(np.argmax(fused_heatmap * bottom)), fused_heatmap.shape)
            cx, cy = float(ix), float(iy)
        else:
            iy, ix = np.unravel_index(int(np.argmax(blend)), blend.shape)
            cx, cy = float(ix), float(iy)

        sorted_scores = np.sort(combined)[::-1]
        margin = float(sorted_scores[0] - sorted_scores[1]) if len(sorted_scores) > 1 else sorted_scores[0]
        energy_peak = float(np.percentile(m, 99.0))
        energy_mean = float(np.mean([np.mean(x) for x in motion_stack]))
        conf = 0.22 + 0.38 * min(1.8, energy_peak / max(1e-3, energy_mean * 2.8)) + 0.22 * min(1.0, margin * 3.0)
        if turning:
            conf = min(0.95, conf + 0.10)
        conf = float(np.clip(conf, 0.0, 0.95))

        # motion_stack[i] spans grays[i:i+3]; use center frame index i+1 (max len(grays)-1).
        bounce_gray_idx = int(np.clip(bounce_motion_idx + 1, 0, len(motion_stack) + 1))
        return cx, cy, bounce_gray_idx, conf, "", bounce_motion_idx

    def _post_direction_from_centroids(
        self, motion_stack: list[np.ndarray], bounce_motion_idx: int
    ) -> tuple[float | None, float | None]:
        if len(motion_stack) < bounce_motion_idx + 3:
            return None, None

        def com(mi: int) -> tuple[float, float]:
            m = motion_stack[mi]
            M = cv2.moments(m)
            if M["m00"] < 1e-6:
                return float(m.shape[1] * 0.5), float(m.shape[0] * 0.5)
            return float(M["m10"] / M["m00"]), float(M["m01"] / M["m00"])

        x0, y0 = com(bounce_motion_idx)
        x1, y1 = com(min(len(motion_stack) - 1, bounce_motion_idx + 3))
        dx, dy = x1 - x0, y1 - y0
        n = float(np.hypot(dx, dy))
        if n < 1e-3:
            return None, None
        return dx / n, dy / n
