from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from app.schemas import Candidate, DeliveryMetadata


def candidate_key(candidate: Candidate) -> str:
    return f"{candidate.frameIndex}:{candidate.x:.6f}:{candidate.y:.6f}:{candidate.source}:{candidate.confidence:.6f}"


@dataclass
class Tracklet:
    id: int
    candidates: list[Candidate]
    score: float
    rejected: bool
    rejection_reasons: list[str]
    metrics: dict[str, float]


@dataclass
class TrackletBuildResult:
    tracklets: list[Tracklet]
    best_tracklet: Tracklet | None
    selected_candidates: list[Candidate]
    rejected_candidate_reasons: dict[str, str]


class CandidateTrackletBuilder:
    def __init__(
        self,
        max_frame_gap: int = 2,
        max_pixel_displacement: float = 110.0,
        min_tracklet_length: int = 4,
        min_detector_ratio: float = 0.25,
        allow_motion_only_start: bool = False,
    ) -> None:
        self.max_frame_gap = max(1, max_frame_gap)
        self.max_pixel_displacement = max(10.0, max_pixel_displacement)
        self.min_tracklet_length = max(3, min_tracklet_length)
        self.min_detector_ratio = max(0.0, min(1.0, min_detector_ratio))
        self.allow_motion_only_start = allow_motion_only_start

    def build(
        self,
        candidates: list[Candidate],
        metadata: DeliveryMetadata,
        frame_width: int,
        frame_height: int,
        active_window: dict[str, float | str],
    ) -> TrackletBuildResult:
        by_frame: dict[int, list[Candidate]] = {}
        for candidate in candidates:
            by_frame.setdefault(candidate.frameIndex, []).append(candidate)
        ordered_frames = sorted(by_frame)
        tracklets: list[list[Candidate]] = []
        rejected_candidate_reasons: dict[str, str] = {}

        for frame_index in ordered_frames:
            frame_candidates = sorted(by_frame[frame_index], key=self._candidate_priority, reverse=True)
            for candidate in frame_candidates:
                linked = False
                for tracklet in tracklets:
                    last = tracklet[-1]
                    frame_gap = candidate.frameIndex - last.frameIndex
                    if frame_gap < 1 or frame_gap > self.max_frame_gap:
                        continue
                    if not self._can_link(last, candidate, frame_gap, frame_width, frame_height):
                        continue
                    tracklet.append(candidate)
                    linked = True
                    break
                if not linked:
                    if self._is_detector(candidate) or self.allow_motion_only_start:
                        tracklets.append([candidate])
                    else:
                        rejected_candidate_reasons[candidate_key(candidate)] = "motion cannot seed tracklet"

        built_tracklets: list[Tracklet] = []
        for index, seq in enumerate(tracklets):
            rejected, reasons, metrics = self._evaluate_tracklet(seq, metadata, frame_width, frame_height, active_window)
            built_tracklets.append(
                Tracklet(
                    id=index,
                    candidates=seq,
                    score=self._score_tracklet(seq, metrics, rejected),
                    rejected=rejected,
                    rejection_reasons=reasons,
                    metrics=metrics,
                )
            )
            if rejected:
                for candidate in seq:
                    rejected_candidate_reasons.setdefault(candidate_key(candidate), "; ".join(reasons))

        valid_tracklets = [tracklet for tracklet in built_tracklets if not tracklet.rejected]
        best_tracklet = max(valid_tracklets, key=lambda tracklet: tracklet.score) if valid_tracklets else None
        selected_candidates = best_tracklet.candidates if best_tracklet else []
        return TrackletBuildResult(
            tracklets=sorted(built_tracklets, key=lambda tracklet: tracklet.score, reverse=True),
            best_tracklet=best_tracklet,
            selected_candidates=selected_candidates,
            rejected_candidate_reasons=rejected_candidate_reasons,
        )

    def _candidate_priority(self, candidate: Candidate) -> float:
        source_bonus = 0.0
        if self._is_detector(candidate):
            source_bonus = 0.25
        elif "optical_flow" in candidate.source:
            source_bonus = 0.02
        return float((candidate.score or candidate.confidence) + source_bonus)

    def _is_detector(self, candidate: Candidate) -> bool:
        source = candidate.source.lower()
        return source == "cricket_ball_v2" or source.startswith("local_") or candidate.modelRole == "ball_detector"

    def _can_link(self, previous: Candidate, current: Candidate, frame_gap: int, frame_width: int, frame_height: int) -> bool:
        dx = (current.x - previous.x) * frame_width
        dy = (current.y - previous.y) * frame_height
        distance = float(math.hypot(dx, dy))
        limit = self.max_pixel_displacement * max(1.0, frame_gap * 0.85)
        return distance <= limit

    def _evaluate_tracklet(
        self,
        seq: list[Candidate],
        metadata: DeliveryMetadata,
        frame_width: int,
        frame_height: int,
        active_window: dict[str, float | str],
    ) -> tuple[bool, list[str], dict[str, float]]:
        reasons: list[str] = []
        detector_count = sum(1 for candidate in seq if self._is_detector(candidate))
        motion_count = sum(1 for candidate in seq if "optical_flow" in candidate.source)
        detector_ratio = detector_count / max(1, len(seq))
        times = self._candidate_times(seq, metadata)
        duration = max(0.0, times[-1] - times[0]) if len(times) > 1 else 0.0
        confidence_mean = float(np.mean([candidate.confidence for candidate in seq])) if seq else 0.0

        displacements = []
        for prev, curr in zip(seq, seq[1:]):
            displacements.append(float(math.hypot((curr.x - prev.x) * frame_width, (curr.y - prev.y) * frame_height)))
        avg_speed_px = float(np.mean(displacements)) if displacements else 0.0
        max_jump_px = float(max(displacements)) if displacements else 0.0

        smoothness = self._smoothness(seq, frame_width, frame_height)
        monotonic = self._monotonic_forward_ratio(seq, metadata)
        early_start = times[0] if len(times) else 0.0
        active_start = float(active_window.get("activeStartSec", 0.0))

        if len(seq) < self.min_tracklet_length:
            reasons.append("too short")
        if detector_count == 0 and not self.allow_motion_only_start:
            reasons.append("motion-only tracklet")
        if detector_ratio < self.min_detector_ratio and not self.allow_motion_only_start:
            reasons.append("low detector support ratio")
        if max_jump_px > self.max_pixel_displacement * 1.2:
            reasons.append("frame-to-frame jump too large")
        if smoothness < 0.20:
            reasons.append("zig-zag movement")
        if avg_speed_px > self.max_pixel_displacement * 0.95:
            reasons.append("implausible image-space speed")
        if monotonic < 0.45:
            reasons.append("non-monotonic forward movement")
        if early_start < 0.7 and detector_count == 0 and active_start < 1.0:
            reasons.append("early clip start without detector support")

        metrics = {
            "length": float(len(seq)),
            "durationSec": duration,
            "detectorCount": float(detector_count),
            "motionCount": float(motion_count),
            "detectorRatio": detector_ratio,
            "avgConfidence": confidence_mean,
            "smoothness": smoothness,
            "avgSpeedPxPerFrame": avg_speed_px,
            "maxJumpPx": max_jump_px,
            "forwardMonotonicRatio": monotonic,
            "startSec": early_start,
        }
        return (len(reasons) > 0), reasons, metrics

    def _score_tracklet(self, seq: list[Candidate], metrics: dict[str, float], rejected: bool) -> float:
        base = (
            metrics["length"] * 2.0
            + metrics["durationSec"] * 8.0
            + metrics["detectorRatio"] * 8.0
            + metrics["avgConfidence"] * 4.0
            + metrics["smoothness"] * 3.5
            + metrics["forwardMonotonicRatio"] * 2.0
        )
        if rejected:
            base -= 100.0
        return float(base)

    def _smoothness(self, seq: list[Candidate], frame_width: int, frame_height: int) -> float:
        if len(seq) < 3:
            return 0.5
        penalties = []
        for a, b, c in zip(seq, seq[1:], seq[2:]):
            v1 = np.array([(b.x - a.x) * frame_width, (b.y - a.y) * frame_height], dtype=float)
            v2 = np.array([(c.x - b.x) * frame_width, (c.y - b.y) * frame_height], dtype=float)
            n1 = float(np.linalg.norm(v1))
            n2 = float(np.linalg.norm(v2))
            if n1 < 1e-6 or n2 < 1e-6:
                continue
            cos_sim = float(np.dot(v1, v2) / max(1e-6, n1 * n2))
            penalties.append((cos_sim + 1.0) / 2.0)
        if not penalties:
            return 0.5
        return float(np.mean(penalties))

    def _monotonic_forward_ratio(self, seq: list[Candidate], metadata: DeliveryMetadata) -> float:
        if len(seq) < 3:
            return 0.5
        spine = metadata.corridorGeometry.spine
        if len(spine) >= 2:
            s0 = np.array([spine[0].x, spine[0].y], dtype=float)
            s1 = np.array([spine[-1].x, spine[-1].y], dtype=float)
            axis = s1 - s0
            denom = float(np.linalg.norm(axis))
            if denom > 1e-6:
                axis = axis / denom
                projections = [float(np.dot(np.array([candidate.x, candidate.y]) - s0, axis)) for candidate in seq]
                increasing = sum(1 for prev, curr in zip(projections, projections[1:]) if curr >= prev - 0.01)
                return increasing / max(1, len(projections) - 1)
        # Fallback: y increasing down pitch in many views.
        increasing = sum(1 for prev, curr in zip(seq, seq[1:]) if curr.y >= prev.y - 0.01)
        return increasing / max(1, len(seq) - 1)

    def _candidate_times(self, candidates: list[Candidate], metadata: DeliveryMetadata) -> list[float]:
        if metadata.frameTimestamps:
            max_index = len(metadata.frameTimestamps) - 1
            return [float(metadata.frameTimestamps[min(max(0, candidate.frameIndex), max_index)]) for candidate in candidates]
        fps = float(metadata.extras.get("videoFps") or 30.0)
        return [candidate.frameIndex / fps for candidate in candidates]
