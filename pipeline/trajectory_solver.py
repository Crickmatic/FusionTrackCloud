from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.schemas import Candidate, DeliveryMetadata, Point2D, Point3D


@dataclass
class TrajectorySolution:
    release_point: Point2D
    bounce_point: Point2D
    end_point: Point2D
    trajectory: list[Point3D]
    confidence: float
    robust_fit_used_points: int
    robust_fit_outliers_removed: int
    yolo_points_used: int
    ball_points_used: int
    motion_points_used: int
    motion_only_path: bool


class TrajectorySolver:
    def solve(self, candidates: list[Candidate], metadata: DeliveryMetadata) -> TrajectorySolution:
        filtered = self.filter_candidates(candidates, metadata)
        if len(filtered) < 4:
            raise ValueError("not enough temporally consistent ball candidates")

        yolo_points = sum(1 for candidate in filtered if "yolo" in candidate.source)
        ball_points = sum(1 for candidate in filtered if candidate.modelRole == "ball_detector" or candidate.source == "cricket_ball_v2")
        motion_points = sum(1 for candidate in filtered if "optical_flow" in candidate.source)
        motion_only = ball_points == 0 and yolo_points == 0

        t = self._candidate_times(filtered, metadata)
        xs = np.array([candidate.x for candidate in filtered], dtype=float)
        ys = np.array([candidate.y for candidate in filtered], dtype=float)
        weights = np.array([max(0.05, self._candidate_weight(candidate)) for candidate in filtered], dtype=float)

        normalized_t = (t - t[0]) / max(1e-6, t[-1] - t[0])
        inlier_mask = np.ones(len(filtered), dtype=bool)
        for _ in range(2):
            x_poly = np.polyfit(normalized_t[inlier_mask], xs[inlier_mask], deg=2, w=weights[inlier_mask])
            y_poly = np.polyfit(normalized_t[inlier_mask], ys[inlier_mask], deg=2, w=weights[inlier_mask])
            residuals = np.hypot(np.polyval(x_poly, normalized_t) - xs, np.polyval(y_poly, normalized_t) - ys)
            threshold = max(0.04, np.percentile(residuals, 75) * 1.4)
            new_mask = residuals <= threshold
            if new_mask.sum() < 4:
                break
            if np.array_equal(new_mask, inlier_mask):
                break
            inlier_mask = new_mask
        if inlier_mask.sum() < 4:
            raise ValueError("robust fit rejected too many outliers")

        x_poly = np.polyfit(normalized_t[inlier_mask], xs[inlier_mask], deg=2, w=weights[inlier_mask])
        y_poly = np.polyfit(normalized_t[inlier_mask], ys[inlier_mask], deg=2, w=weights[inlier_mask])
        sample_t = np.linspace(0, 1, num=max(16, min(48, len(filtered) * 4)))
        path_x = np.clip(np.polyval(x_poly, sample_t), 0.0, 1.0)
        path_y = np.clip(np.polyval(y_poly, sample_t), 0.0, 1.0)

        fit_error = float(
            np.mean(
                np.hypot(
                    np.polyval(x_poly, normalized_t[inlier_mask]) - xs[inlier_mask],
                    np.polyval(y_poly, normalized_t[inlier_mask]) - ys[inlier_mask],
                )
            )
        )
        fit_error_threshold = 0.055 if motion_only else 0.09
        if fit_error > fit_error_threshold:
            raise ValueError(f"trajectory fit error too high: {fit_error:.3f}")

        release_idx = 0
        bounce_idx = self._detect_bounce_index(path_y)
        end_idx = len(sample_t) - 1
        time_values = t[0] + sample_t * (t[-1] - t[0])
        trajectory = [
            Point3D(
                x=float(path_x[index]),
                y=float(self._height_proxy(index, bounce_idx, len(sample_t))),
                z=float(path_y[index]),
                t=float(time_values[index]),
            )
            for index in range(len(sample_t))
        ]

        confidence = float(
            min(
                0.94,
                max(
                    0.28,
                    np.mean(weights) * 0.38 + min(1, len(filtered) / 10) * 0.26 + max(0, 1 - fit_error / 0.08) * 0.36,
                ),
            )
        )
        if motion_only:
            if len(filtered) < 8:
                raise ValueError("motion-only path has insufficient temporal support")
            confidence = min(confidence, 0.62)
        return TrajectorySolution(
            release_point=Point2D(x=float(path_x[release_idx]), y=float(path_y[release_idx]), t=float(time_values[release_idx])),
            bounce_point=Point2D(x=float(path_x[bounce_idx]), y=float(path_y[bounce_idx]), t=float(time_values[bounce_idx])),
            end_point=Point2D(x=float(path_x[end_idx]), y=float(path_y[end_idx]), t=float(time_values[end_idx])),
            trajectory=trajectory,
            confidence=confidence,
            robust_fit_used_points=int(inlier_mask.sum()),
            robust_fit_outliers_removed=int((~inlier_mask).sum()),
            yolo_points_used=yolo_points,
            ball_points_used=ball_points,
            motion_points_used=motion_points,
            motion_only_path=motion_only,
        )

    def filter_candidates(self, candidates: list[Candidate], metadata: DeliveryMetadata) -> list[Candidate]:
        del metadata
        grouped: dict[int, list[Candidate]] = {}
        for candidate in candidates:
            grouped.setdefault(candidate.frameIndex, []).append(candidate)

        selected: list[Candidate] = []
        for frame_index in sorted(grouped):
            best = max(grouped[frame_index], key=self._candidate_score)
            if selected:
                jump = float(np.hypot(best.x - selected[-1].x, best.y - selected[-1].y))
                if jump > 0.28:
                    continue
            selected.append(best)

        if len(selected) < 4:
            return []

        # Enforce forward travel in image/pitch coordinates. Direction can vary by
        # camera, so require meaningful travel rather than a hard sign.
        travel = float(np.hypot(selected[-1].x - selected[0].x, selected[-1].y - selected[0].y))
        if travel < 0.08:
            return []
        return selected

    def _candidate_score(self, candidate: Candidate) -> float:
        return self._candidate_weight(candidate)

    def _candidate_weight(self, candidate: Candidate) -> float:
        source_bias = 0.0
        if "yolo" in candidate.source:
            source_bias = 0.20
        elif "optical_flow" in candidate.source:
            source_bias = 0.02
        elif "local_" in candidate.source:
            source_bias = 0.08
        return (candidate.score or candidate.confidence) + source_bias

    def _candidate_times(self, candidates: list[Candidate], metadata: DeliveryMetadata) -> np.ndarray:
        if metadata.frameTimestamps:
            max_index = len(metadata.frameTimestamps) - 1
            return np.array(
                [
                    metadata.frameTimestamps[min(max(0, candidate.frameIndex), max_index)]
                    for candidate in candidates
                ],
                dtype=float,
            )
        fps = float(metadata.extras.get("videoFps") or 30.0)
        return np.array([candidate.frameIndex / fps for candidate in candidates], dtype=float)

    def _detect_bounce_index(self, ys: np.ndarray) -> int:
        if len(ys) < 5:
            return int(np.argmax(ys))
        gradients = np.gradient(ys)
        best_index = int(np.argmax(ys))
        best_score = -float("inf")
        for index in range(2, len(ys) - 2):
            reversal = gradients[index - 1] - gradients[index + 1]
            score = float(ys[index] + max(0, reversal) * 0.5)
            if score > best_score:
                best_score = score
                best_index = index
        return best_index

    def _height_proxy(self, index: int, bounce_index: int, count: int) -> float:
        if count <= 1:
            return 0.0
        t = index / (count - 1)
        bounce_t = max(0.05, min(0.95, bounce_index / (count - 1)))
        if t <= bounce_t:
            return float(max(0.0, 1.0 - (t / bounce_t)) * 1.6)
        return float(max(0.0, 1.0 - ((t - bounce_t) / max(0.01, 1 - bounce_t))) * 0.45)
