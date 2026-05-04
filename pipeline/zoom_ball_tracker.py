from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from app.schemas import Candidate, DeliveryMetadata


@dataclass
class ZoomTrackerConfig:
    min_seed_conf: float = 0.12
    template_half_size_px: int = 14
    base_search_half_size_px: int = 36
    bounce_search_half_size_px: int = 76
    bounce_time_window_sec: float = 0.30
    bounce_upsample_factor: int = 4
    upsample_factor: int = 3
    max_step_px: float = 130.0


class ZoomBallTracker:
    def __init__(self, config: ZoomTrackerConfig | None = None) -> None:
        self.config = config or ZoomTrackerConfig()

    def generate(
        self,
        frames: list[np.ndarray],
        frame_indices: list[int],
        metadata: DeliveryMetadata,
        ball_candidates: list[Candidate],
        release_hint_sec: float | None = None,
        bounce_hint_sec: float | None = None,
    ) -> list[Candidate]:
        if len(frames) < 2 or not ball_candidates:
            return []
        frame_map = {frame_index: frame for frame, frame_index in zip(frames, frame_indices)}
        ordered_indices = sorted(frame_indices)
        seeds = sorted(
            [candidate for candidate in ball_candidates if candidate.confidence >= self.config.min_seed_conf and candidate.frameIndex in frame_map],
            key=lambda candidate: candidate.frameIndex,
        )
        if not seeds:
            return []
        seed = seeds[0]
        by_frame: dict[int, list[Candidate]] = {}
        for candidate in ball_candidates:
            by_frame.setdefault(candidate.frameIndex, []).append(candidate)
        output: list[Candidate] = []
        prev = seed
        prev_prev: Candidate | None = None
        for frame_index in ordered_indices:
            if frame_index <= seed.frameIndex:
                continue
            current_time = self._frame_time(frame_index, metadata)
            if release_hint_sec is not None and current_time < release_hint_sec:
                continue
            frame_ball_candidates = sorted(by_frame.get(frame_index, []), key=lambda candidate: candidate.confidence, reverse=True)
            if frame_ball_candidates:
                nearest = min(frame_ball_candidates, key=lambda candidate: abs(candidate.x - prev.x) + abs(candidate.y - prev.y))
                if nearest.confidence >= self.config.min_seed_conf:
                    prev_prev = prev
                    prev = nearest
            prev_frame = frame_map.get(prev.frameIndex)
            curr_frame = frame_map.get(frame_index)
            if prev_frame is None or curr_frame is None:
                continue
            velocity = self._velocity(prev_prev, prev, metadata)
            predicted_x = prev.x + velocity[0]
            predicted_y = prev.y + velocity[1]
            is_bounce_zone = (
                bounce_hint_sec is not None and abs(current_time - bounce_hint_sec) <= self.config.bounce_time_window_sec
            )
            search_half_size = self.config.bounce_search_half_size_px if is_bounce_zone else self.config.base_search_half_size_px
            upsample = self.config.bounce_upsample_factor if is_bounce_zone else self.config.upsample_factor
            refined = self._refine_in_crop(
                prev_frame=prev_frame,
                curr_frame=curr_frame,
                predicted_x=predicted_x,
                predicted_y=predicted_y,
                search_half_size=search_half_size,
                upsample_factor=upsample,
            )
            if refined is None:
                continue
            x_norm, y_norm, match_score = refined
            dx_px = abs((x_norm - prev.x) * curr_frame.shape[1])
            dy_px = abs((y_norm - prev.y) * curr_frame.shape[0])
            if np.hypot(dx_px, dy_px) > self.config.max_step_px:
                continue
            confidence = float(np.clip(0.08 + (match_score * 0.55), 0.05, 0.55))
            candidate = Candidate(
                frameIndex=frame_index,
                x=float(np.clip(x_norm, 0.0, 1.0)),
                y=float(np.clip(y_norm, 0.0, 1.0)),
                confidence=confidence,
                source="zoom_crop_tracker",
                modelAlias="zoom_crop_tracker",
                modelRole="ball_detector",
                className="sports_ball_zoom",
                bbox=self._bbox_around(x_norm, y_norm, curr_frame.shape[1], curr_frame.shape[0]),
                diagnostics={"searchHalfSizePx": search_half_size, "matchScore": match_score},
            )
            output.append(candidate)
            prev_prev = prev
            prev = candidate
        return output

    def _bbox_around(self, x_norm: float, y_norm: float, width: int, height: int) -> dict[str, float]:
        half = max(6, self.config.template_half_size_px)
        cx = float(np.clip(x_norm, 0.0, 1.0) * width)
        cy = float(np.clip(y_norm, 0.0, 1.0) * height)
        return {
            "x1": max(0.0, cx - half),
            "y1": max(0.0, cy - half),
            "x2": min(float(width - 1), cx + half),
            "y2": min(float(height - 1), cy + half),
            "width": float(half * 2),
            "height": float(half * 2),
        }

    def _refine_in_crop(
        self,
        prev_frame: np.ndarray,
        curr_frame: np.ndarray,
        predicted_x: float,
        predicted_y: float,
        search_half_size: int,
        upsample_factor: int | None = None,
    ) -> tuple[float, float, float] | None:
        height, width = curr_frame.shape[:2]
        px = int(np.clip(predicted_x * width, 0, width - 1))
        py = int(np.clip(predicted_y * height, 0, height - 1))
        t = self.config.template_half_size_px
        sx1 = max(0, px - search_half_size)
        sy1 = max(0, py - search_half_size)
        sx2 = min(width, px + search_half_size + 1)
        sy2 = min(height, py + search_half_size + 1)
        tx1 = max(0, px - t)
        ty1 = max(0, py - t)
        tx2 = min(width, px + t + 1)
        ty2 = min(height, py + t + 1)
        if (sx2 - sx1) < (2 * t + 1) or (sy2 - sy1) < (2 * t + 1):
            return None
        search = curr_frame[sy1:sy2, sx1:sx2]
        template = prev_frame[ty1:ty2, tx1:tx2]
        if template.size == 0 or search.size == 0:
            return None
        search_gray = cv2.cvtColor(search, cv2.COLOR_BGR2GRAY)
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        factor = upsample_factor if upsample_factor is not None else self.config.upsample_factor
        if factor > 1:
            search_gray = cv2.resize(search_gray, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)
            template_gray = cv2.resize(template_gray, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)
        if search_gray.shape[0] < template_gray.shape[0] or search_gray.shape[1] < template_gray.shape[1]:
            return None
        result = cv2.matchTemplate(search_gray, template_gray, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        if max_val < 0.12:
            return None
        factor = max(1, factor)
        center_x = sx1 + (max_loc[0] / factor) + (template.shape[1] / 2.0)
        center_y = sy1 + (max_loc[1] / factor) + (template.shape[0] / 2.0)
        return (center_x / width, center_y / height, float(max_val))

    def _frame_time(self, frame_index: int, metadata: DeliveryMetadata) -> float:
        if metadata.frameTimestamps and 0 <= frame_index < len(metadata.frameTimestamps):
            return float(metadata.frameTimestamps[frame_index])
        fps = float(metadata.fps or metadata.extras.get("videoFps") or 30.0)
        return frame_index / max(1e-6, fps)

    def _velocity(
        self,
        prev_prev: Candidate | None,
        prev: Candidate,
        metadata: DeliveryMetadata,
    ) -> tuple[float, float]:
        if prev_prev is None:
            return (0.0, 0.0)
        t1 = self._frame_time(prev_prev.frameIndex, metadata)
        t2 = self._frame_time(prev.frameIndex, metadata)
        dt = max(1e-6, t2 - t1)
        return ((prev.x - prev_prev.x) / dt * 0.02, (prev.y - prev_prev.y) / dt * 0.02)
