from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from app.config import Settings
from app.schemas import Candidate

logger = logging.getLogger(__name__)


class YoloUltralyticsCandidateGenerator:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.models: list[tuple[str, object]] = []
        self.missing_model_files: list[str] = []
        self.ultralytics_version: str | None = None
        self.last_stats: dict[str, int] = {}

    @property
    def loaded_model_names(self) -> list[str]:
        return [name for name, _ in self.models]

    def load(self) -> None:
        try:
            import ultralytics
            from ultralytics import YOLO
        except Exception as exc:
            logger.warning("Ultralytics unavailable, YOLO candidates disabled: %s", exc)
            return
        self.ultralytics_version = getattr(ultralytics, "__version__", None)

        for model_path in self.settings.yolo_model_paths:
            if not model_path.exists():
                logger.warning("YOLO model not found: %s", model_path)
                self.missing_model_files.append(str(model_path))
                continue
            try:
                self.models.append((model_path.name, YOLO(str(model_path))))
                logger.info("Loaded YOLO model %s", model_path)
            except Exception as exc:
                logger.warning("Failed to load YOLO model %s: %s", model_path, exc)

    def generate(self, frames: list[np.ndarray], frame_indices: list[int]) -> list[Candidate]:
        candidates: list[Candidate] = []
        primary_count = 0
        fallback_count = 0
        crop_count = 0
        primary_conf = self.settings.yolo_confidence_threshold
        fallback_conf = 0.10

        crop_specs = {
            "full": (0.0, 0.0, 1.0, 1.0),
            "pitch_corridor": (0.18, 0.08, 0.64, 0.86),
            "bounce_corridor": (0.25, 0.35, 0.50, 0.55),
        }

        for model_name, model in self.models:
            for frame, frame_index in zip(frames, frame_indices):
                height, width = frame.shape[:2]
                for crop_name, (x0, y0, w, h) in crop_specs.items():
                    sx = int(max(0, min(width - 1, x0 * width)))
                    sy = int(max(0, min(height - 1, y0 * height)))
                    ex = int(max(sx + 1, min(width, (x0 + w) * width)))
                    ey = int(max(sy + 1, min(height, (y0 + h) * height)))
                    crop = frame[sy:ey, sx:ex]
                    if crop.size == 0:
                        continue
                    for threshold, threshold_tag in [(primary_conf, "primary"), (fallback_conf, "fallback")]:
                        detections = self._predict_candidates(
                            model=model,
                            frame=crop,
                            frame_index=frame_index,
                            source=f"{model_name}:{threshold_tag}:{crop_name}",
                            threshold=threshold,
                            crop_offset=(sx, sy),
                            full_size=(width, height),
                        )
                        for candidate in detections:
                            if threshold_tag == "primary":
                                primary_count += 1
                            else:
                                fallback_count += 1
                            if crop_name != "full":
                                crop_count += 1
                            candidates.append(candidate)
        self.last_stats = {
            "yolo_primary": primary_count,
            "yolo_fallback": fallback_count,
            "yolo_crop": crop_count,
            "yolo_total": len(candidates),
        }
        return candidates

    def _predict_candidates(
        self,
        model: object,
        frame: np.ndarray,
        frame_index: int,
        source: str,
        threshold: float,
        crop_offset: tuple[int, int],
        full_size: tuple[int, int],
    ) -> list[Candidate]:
        results = model.predict(frame, verbose=False, conf=threshold)
        full_width, full_height = full_size
        offset_x, offset_y = crop_offset
        candidates: list[Candidate] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                cls = int(box.cls.item())
                if cls != self.settings.sports_ball_class_id:
                    continue
                confidence = float(box.conf.item())
                xyxy = box.xyxy.cpu().numpy()[0]
                x1, y1, x2, y2 = xyxy
                center_x = ((x1 + x2) / 2) + offset_x
                center_y = ((y1 + y2) / 2) + offset_y
                candidates.append(
                    Candidate(
                        frameIndex=frame_index,
                        x=float(center_x / max(full_width, 1)),
                        y=float(center_y / max(full_height, 1)),
                        confidence=confidence,
                        score=confidence,
                        source=source,
                        diagnostics={
                            "bboxWidthNorm": float(max(0.0, (x2 - x1) / max(full_width, 1))),
                            "bboxHeightNorm": float(max(0.0, (y2 - y1) / max(full_height, 1))),
                        },
                    )
                )
        return candidates
