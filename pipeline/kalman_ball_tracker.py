from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.schemas import Candidate, DeliveryMetadata


@dataclass
class KalmanTrackResult:
    tracked_points: list[Candidate]
    predictions: list[Candidate]
    accepted_weak_candidates: int
    synthetic_points: int


class KalmanBallTracker:
    """Constant-velocity Kalman filter for robust ball tracking."""

    def track(
        self,
        candidates: list[Candidate],
        metadata: DeliveryMetadata,
        fps: float,
        release_hint_sec: float | None = None,
    ) -> KalmanTrackResult:
        ball_candidates = [c for c in candidates if c.modelRole == "ball_detector"]
        if not ball_candidates:
            return KalmanTrackResult(tracked_points=[], predictions=[], accepted_weak_candidates=0, synthetic_points=0)
        ordered = sorted(ball_candidates, key=lambda c: self._candidate_time(c, metadata, fps))
        by_frame: dict[int, list[Candidate]] = {}
        for candidate in ordered:
            by_frame.setdefault(candidate.frameIndex, []).append(candidate)
        frames = sorted(by_frame)
        dt = 1.0 / max(1.0, fps)

        # state = [x, y, vx, vy]
        x = np.array([ordered[0].x, ordered[0].y, 0.0, 0.0], dtype=float)
        P = np.eye(4, dtype=float) * 0.08
        F = np.array(
            [[1.0, 0.0, dt, 0.0], [0.0, 1.0, 0.0, dt], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
            dtype=float,
        )
        H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=float)
        Q = np.diag([2e-3, 2e-3, 7e-3, 7e-3]).astype(float)

        tracked: list[Candidate] = []
        predictions: list[Candidate] = []
        accepted_weak = 0
        synthetic_points = 0
        prev_frame = frames[0]
        release_sec = release_hint_sec if release_hint_sec is not None else self._candidate_time(ordered[0], metadata, fps)

        for frame in frames:
            frame_dt = max(1, frame - prev_frame) * dt
            if frame_dt != dt:
                F = np.array(
                    [[1.0, 0.0, frame_dt, 0.0], [0.0, 1.0, 0.0, frame_dt], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
                    dtype=float,
                )
            x = F @ x
            P = F @ P @ F.T + Q
            pred_x, pred_y = float(x[0]), float(x[1])
            pred = Candidate(
                frameIndex=frame,
                x=float(np.clip(pred_x, 0.0, 1.0)),
                y=float(np.clip(pred_y, 0.0, 1.0)),
                confidence=0.35,
                source="kalman_prediction",
                timestampSec=self._frame_time(frame, metadata, fps),
                modelRole="ball_detector",
            )
            predictions.append(pred)

            candidates_in_frame = by_frame.get(frame, [])
            best = self._select_measurement(candidates_in_frame, pred_x, pred_y, frame_dt, release_sec, metadata, fps)
            if best is None:
                tracked.append(pred)
                synthetic_points += 1
                prev_frame = frame
                continue
            if float(best.confidence or 0.0) < 0.25:
                accepted_weak += 1
            measurement = np.array([best.x, best.y], dtype=float)
            R_scale = max(0.18, min(1.8, 1.0 - float(best.confidence or 0.0) + 0.25))
            R = np.diag([4e-3, 4e-3]).astype(float) * R_scale
            y_residual = measurement - (H @ x)
            S = H @ P @ H.T + R
            K = P @ H.T @ np.linalg.inv(S)
            x = x + K @ y_residual
            P = (np.eye(4, dtype=float) - K @ H) @ P
            tracked.append(
                Candidate(
                    frameIndex=best.frameIndex,
                    x=float(np.clip(x[0], 0.0, 1.0)),
                    y=float(np.clip(x[1], 0.0, 1.0)),
                    confidence=max(float(best.confidence or 0.0), 0.4),
                    source="kalman_tracked",
                    timestampSec=self._candidate_time(best, metadata, fps),
                    modelRole="ball_detector",
                )
            )
            prev_frame = frame

        return KalmanTrackResult(
            tracked_points=tracked,
            predictions=predictions,
            accepted_weak_candidates=accepted_weak,
            synthetic_points=synthetic_points,
        )

    def _select_measurement(
        self,
        candidates: list[Candidate],
        pred_x: float,
        pred_y: float,
        frame_dt: float,
        release_sec: float,
        metadata: DeliveryMetadata,
        fps: float,
    ) -> Candidate | None:
        if not candidates:
            return None
        best: Candidate | None = None
        best_score = -1e9
        lane = metadata.corridorGeometry.pitchPolygon
        for candidate in candidates:
            dist = float(np.hypot(candidate.x - pred_x, candidate.y - pred_y))
            time_sec = self._candidate_time(candidate, metadata, fps)
            conf = float(candidate.confidence or 0.0)
            lane_bonus = 0.0
            if len(lane) >= 4:
                xs = [point.x for point in lane[:4]]
                ys = [point.y for point in lane[:4]]
                lane_bonus = 0.25 if (min(xs) - 0.04 <= candidate.x <= max(xs) + 0.04 and min(ys) - 0.08 <= candidate.y <= max(ys) + 0.08) else -0.25
            early_penalty = 0.0 if time_sec >= release_sec - 0.08 else 0.35
            # weak candidates are allowed if they align with Kalman prediction
            weak_alignment_bonus = 0.35 if conf < 0.25 and dist < 0.055 else 0.0
            score = (conf * 1.1) - (dist * 5.0) + lane_bonus + weak_alignment_bonus - early_penalty
            if score > best_score:
                best = candidate
                best_score = score
        if best is None:
            return None
        if np.hypot(best.x - pred_x, best.y - pred_y) > (0.11 + min(0.06, frame_dt * 0.7)):
            return None
        return best

    def _candidate_time(self, candidate: Candidate, metadata: DeliveryMetadata, fps: float) -> float:
        if candidate.timestampSec is not None:
            return float(candidate.timestampSec)
        return self._frame_time(candidate.frameIndex, metadata, fps)

    def _frame_time(self, frame_index: int, metadata: DeliveryMetadata, fps: float) -> float:
        if metadata.frameTimestamps and 0 <= frame_index < len(metadata.frameTimestamps):
            return float(metadata.frameTimestamps[frame_index])
        return frame_index / max(1.0, fps)

