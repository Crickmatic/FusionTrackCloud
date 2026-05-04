from __future__ import annotations

import cv2
import numpy as np

from app.schemas import Candidate, DeliveryMetadata


class MotionCandidateGenerator:
    def __init__(self) -> None:
        self.last_stats: dict[str, int] = {}

    def generate(
        self,
        frames: list[np.ndarray],
        frame_indices: list[int],
        metadata: DeliveryMetadata,
    ) -> list[Candidate]:
        if len(frames) < 2:
            self.last_stats = {
                "motion_before_gating": 0,
                "motion_after_corridor": 0,
                "motion_after_cluster": 0,
            }
            return []

        candidates: list[Candidate] = []
        previous_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        last_center: tuple[float, float] | None = None
        last_vector: tuple[float, float] | None = None
        corridor_polygon = self._corridor_polygon(metadata)
        before_gating = 0
        after_corridor = 0

        for frame, frame_index in zip(frames[1:], frame_indices[1:]):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(
                previous_gray,
                gray,
                None,
                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0,
            )
            magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            mask = magnitude > max(1.6, float(np.percentile(magnitude, 96)))
            mask = mask.astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            height, width = gray.shape[:2]
            frame_candidates: list[Candidate] = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 3 or area > (width * height * 0.08):
                    continue
                before_gating += 1
                moments = cv2.moments(contour)
                if moments["m00"] == 0:
                    continue
                cx = float(moments["m10"] / moments["m00"]) / max(width, 1)
                cy = float(moments["m01"] / moments["m00"]) / max(height, 1)
                if not self._inside_corridor((cx, cy), corridor_polygon):
                    continue
                after_corridor += 1

                contour_perimeter = max(1e-6, cv2.arcLength(contour, True))
                compactness = float((4 * np.pi * area) / (contour_perimeter * contour_perimeter))
                speed_magnitude = float(np.mean(magnitude[mask > 0])) if np.any(mask > 0) else 0.0
                corridor_score = self._corridor_score((cx, cy), corridor_polygon)
                distance_previous = 0.0
                direction_consistency = 0.5

                current_vector = None
                if last_center is not None:
                    dx = cx - last_center[0]
                    dy = cy - last_center[1]
                    distance_previous = float(np.hypot(dx, dy))
                    if distance_previous > 0.24:
                        continue
                    current_vector = (dx, dy)
                    if last_vector is not None:
                        dot = (last_vector[0] * dx) + (last_vector[1] * dy)
                        denom = max(1e-6, np.hypot(*last_vector) * np.hypot(dx, dy))
                        direction_consistency = float(max(0.0, min(1.0, (dot / denom + 1.0) / 2.0)))

                score = float(
                    min(
                        1.0,
                        max(
                            0.05,
                            (min(1.0, area / 160.0) * 0.28)
                            + (min(1.0, speed_magnitude / 10.0) * 0.22)
                            + (direction_consistency * 0.20)
                            + (max(0.0, 1.0 - (distance_previous / 0.2)) * 0.15)
                            + (corridor_score * 0.10)
                            + (max(0.0, compactness) * 0.05),
                        ),
                    )
                )
                frame_candidates.append(
                    Candidate(
                        frameIndex=frame_index,
                        x=cx,
                        y=cy,
                        confidence=score,
                        score=score,
                        source="optical_flow",
                        diagnostics={
                            "blobArea": float(area),
                            "speedMagnitude": speed_magnitude,
                            "directionConsistency": direction_consistency,
                            "distanceToPrevious": distance_previous,
                            "corridorScore": corridor_score,
                            "compactness": compactness,
                        },
                    )
                )
                if current_vector is not None and np.hypot(*current_vector) > 0.003:
                    last_vector = current_vector

            frame_candidates.sort(key=lambda candidate: (candidate.score or candidate.confidence), reverse=True)
            selected = frame_candidates[:3]
            candidates.extend(selected)
            if selected:
                strongest = selected[0]
                last_center = (strongest.x, strongest.y)

            previous_gray = gray
        self.last_stats = {
            "motion_before_gating": before_gating,
            "motion_after_corridor": after_corridor,
            "motion_after_cluster": len(candidates),
        }
        return candidates

    def _corridor_polygon(self, metadata: DeliveryMetadata) -> np.ndarray:
        if metadata.corridorGeometry.pitchPolygon and len(metadata.corridorGeometry.pitchPolygon) >= 4:
            points = [(point.x, point.y) for point in metadata.corridorGeometry.pitchPolygon[:4]]
        else:
            points = [(0.25, 0.10), (0.75, 0.10), (0.82, 0.95), (0.18, 0.95)]
        return np.array(points, dtype=np.float32)

    def _inside_corridor(self, point: tuple[float, float], corridor_polygon: np.ndarray) -> bool:
        return cv2.pointPolygonTest(corridor_polygon, point, False) >= 0

    def _corridor_score(self, point: tuple[float, float], corridor_polygon: np.ndarray) -> float:
        distance = cv2.pointPolygonTest(corridor_polygon, point, True)
        return float(max(0.0, min(1.0, (distance + 0.1) / 0.2)))
