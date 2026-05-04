from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np

from app.schemas import Candidate, DeliveryMetadata


@dataclass
class TrajectoryAiCorrection:
    bounce_offset_y: float
    post_bounce_angle_delta: float
    confidence: float


class TrajectoryAiRefiner:
    """
    Lightweight learned correction layer (tiny MLP).
    Uses optional external weights; falls back to bundled deterministic weights.
    """

    def __init__(self, weights_path: str | None = None):
        self.weights = self._load_weights(weights_path)

    def refine(
        self,
        tracked_points: list[Candidate],
        metadata: DeliveryMetadata,
        estimated_bounce_sec: float | None,
        fps: float,
    ) -> TrajectoryAiCorrection:
        if len(tracked_points) < 4:
            return TrajectoryAiCorrection(0.0, 0.0, 0.2)
        ordered = sorted(tracked_points, key=lambda c: self._candidate_time(c, metadata, fps))
        tail = ordered[-10:] if len(ordered) >= 10 else ordered
        features = []
        for c in tail:
            features.extend([float(c.x), float(c.y)])
        while len(features) < 20:
            features.extend([features[-2] if features else 0.5, features[-1] if features else 0.5])
        v = self._velocity(ordered, metadata, fps)
        features.extend([v[0], v[1]])
        features.append(float(estimated_bounce_sec or self._candidate_time(ordered[-1], metadata, fps)))
        features.append(self._pitch_center_x(metadata))
        x = np.array(features, dtype=float)
        x = (x - self.weights["mean"]) / np.maximum(self.weights["std"], 1e-5)
        hidden = np.tanh((self.weights["w1"] @ x) + self.weights["b1"])
        out = (self.weights["w2"] @ hidden) + self.weights["b2"]
        bounce_offset_y = float(np.clip(out[0], -0.04, 0.05))
        post_angle_delta = float(np.clip(out[1], -0.24, 0.24))
        confidence = float(np.clip(0.55 + out[2] * 0.25, 0.2, 0.92))
        return TrajectoryAiCorrection(
            bounce_offset_y=bounce_offset_y,
            post_bounce_angle_delta=post_angle_delta,
            confidence=confidence,
        )

    def _load_weights(self, weights_path: str | None) -> dict[str, np.ndarray]:
        if weights_path:
            path = Path(weights_path)
            if path.exists():
                blob = json.loads(path.read_text(encoding="utf-8"))
                return {key: np.array(value, dtype=float) for key, value in blob.items()}
        # Default tiny model weights (stable fallback).
        rng = np.random.default_rng(42)
        input_size = 24
        hidden_size = 18
        w1 = rng.normal(0.0, 0.08, size=(hidden_size, input_size))
        b1 = np.zeros(hidden_size, dtype=float)
        w2 = rng.normal(0.0, 0.06, size=(3, hidden_size))
        b2 = np.array([0.0, 0.0, 0.08], dtype=float)
        mean = np.zeros(input_size, dtype=float)
        std = np.ones(input_size, dtype=float)
        return {"w1": w1, "b1": b1, "w2": w2, "b2": b2, "mean": mean, "std": std}

    def _velocity(self, candidates: list[Candidate], metadata: DeliveryMetadata, fps: float) -> tuple[float, float]:
        if len(candidates) < 2:
            return (0.0, 0.0)
        a, b = candidates[-2], candidates[-1]
        ta = self._candidate_time(a, metadata, fps)
        tb = self._candidate_time(b, metadata, fps)
        dt = max(1e-6, tb - ta)
        return ((b.x - a.x) / dt, (b.y - a.y) / dt)

    def _pitch_center_x(self, metadata: DeliveryMetadata) -> float:
        polygon = metadata.corridorGeometry.pitchPolygon
        if len(polygon) < 4:
            return 0.5
        return float(np.mean([point.x for point in polygon[:4]]))

    def _candidate_time(self, candidate: Candidate, metadata: DeliveryMetadata, fps: float) -> float:
        if candidate.timestampSec is not None:
            return float(candidate.timestampSec)
        if metadata.frameTimestamps and 0 <= candidate.frameIndex < len(metadata.frameTimestamps):
            return float(metadata.frameTimestamps[candidate.frameIndex])
        return candidate.frameIndex / max(1.0, fps)

