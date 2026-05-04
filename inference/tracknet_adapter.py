from __future__ import annotations

import logging

import numpy as np

from app.schemas import Candidate

logger = logging.getLogger(__name__)


class TrackNetAdapter:
    """Future TrackNetV3 temporal heatmap candidate source.

    Do not make this required yet. The adapter shape is here so the orchestrator
    can add TrackNet as another candidate generator without changing API
    contracts or solver logic.
    """

    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self.loaded_model_names: list[str] = []

    def load(self) -> None:
        if self.enabled:
            logger.info("TrackNet adapter enabled, but no model path configured yet")

    def generate(self, frames: list[np.ndarray], frame_indices: list[int]) -> list[Candidate]:
        del frames, frame_indices
        return []
