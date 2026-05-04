from __future__ import annotations

import logging

import numpy as np

from app.config import Settings
from app.schemas import Candidate, DeliveryMetadata

logger = logging.getLogger(__name__)


class CricketModelCandidateGenerator:
    """Cricket-specific model adapter with explicit ball/stump roles."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.models: list[tuple[str, str, str, object]] = []
        self.loaded_model_names: list[str] = []
        self.missing_model_files: list[str] = []
        self.load_errors: list[str] = []
        self.last_stump_detections: list[Candidate] = []

    def load(self) -> None:
        model_paths = self.settings.cricket_model_paths
        if not model_paths:
            logger.info("No cricket models configured")
            return
        try:
            from ultralytics import YOLO
        except Exception as exc:
            self.load_errors.append(f"ultralytics unavailable for cricket models: {exc}")
            logger.warning("Ultralytics unavailable for cricket models: %s", exc)
            return

        for model_path in model_paths:
            if not model_path.exists():
                self.missing_model_files.append(str(model_path))
                continue
            suffix = model_path.suffix.lower()
            if suffix in {".pt", ".onnx", ".engine"}:
                try:
                    model = YOLO(str(model_path))
                    alias, role = self._model_identity(model_path.name)
                    self.models.append((alias, role, model_path.name, model))
                    self.loaded_model_names.append(alias)
                except Exception as exc:
                    self.load_errors.append(f"{model_path.name}: {exc}")
            else:
                self.load_errors.append(
                    f"{model_path.name}: unsupported format '{suffix}' (expected .pt/.onnx/.engine for cloud inference)"
                )

        if self.load_errors:
            for reason in self.load_errors:
                logger.warning("Cricket model load issue: %s", reason)

    def generate(
        self,
        frames: list[np.ndarray],
        frame_indices: list[int],
        metadata: DeliveryMetadata,
    ) -> list[Candidate]:
        ball_candidates: list[Candidate] = []
        stump_candidates: list[Candidate] = []
        if self.models:
            for model_alias, model_role, model_name, model in self.models:
                predictions = self._predict_model(model_alias, model_role, model_name, model, frames, frame_indices, metadata)
                if model_role == "ball_detector":
                    ball_candidates.extend(predictions)
                elif model_role == "stump_detector":
                    stump_candidates.extend(predictions)
            self.last_stump_detections = stump_candidates
            return ball_candidates

        del frames, frame_indices
        self.last_stump_detections = []
        return [
            Candidate(
                frameIndex=candidate.frameIndex,
                x=candidate.x,
                y=candidate.y,
                confidence=candidate.confidence,
                source=f"local_{candidate.source}",
            )
            for candidate in metadata.localYoloCandidates
        ]

    def _model_identity(self, model_name: str) -> tuple[str, str]:
        lowered = model_name.lower()
        if "v2" in lowered or "ball" in lowered:
            return "cricket_ball_v2", "ball_detector"
        if "v1" in lowered or "stump" in lowered or "wicket" in lowered:
            return "cricket_stumps_v1", "stump_detector"
        return model_name.rsplit(".", 1)[0], "unknown"

    def _predict_model(
        self,
        model_alias: str,
        model_role: str,
        model_name: str,
        model: object,
        frames: list[np.ndarray],
        frame_indices: list[int],
        metadata: DeliveryMetadata,
    ) -> list[Candidate]:
        predictions: list[Candidate] = []
        threshold = max(0.08, self.settings.yolo_confidence_threshold * 0.75)
        class_names = getattr(model, "names", {}) or {}
        for frame, frame_index in zip(frames, frame_indices):
            results = model.predict(frame, verbose=False, conf=threshold)
            height, width = frame.shape[:2]
            for result in results:
                boxes = getattr(result, "boxes", None)
                if boxes is None:
                    continue
                for box in boxes:
                    cls = int(box.cls.item())
                    class_name = str(class_names.get(cls, cls)) if isinstance(class_names, dict) else str(cls)
                    if not self._accept_class(model_role, cls, class_name):
                        continue
                    conf = float(box.conf.item())
                    x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0]
                    timestamp = self._timestamp_for_frame(frame_index, metadata)
                    predictions.append(
                        Candidate(
                            frameIndex=frame_index,
                            x=float(((x1 + x2) / 2) / max(width, 1)),
                            y=float(((y1 + y2) / 2) / max(height, 1)),
                            confidence=conf,
                            score=conf,
                            source=model_alias,
                            timestampSec=timestamp,
                            modelAlias=model_alias,
                            modelRole=model_role,
                            classId=cls,
                            className=class_name,
                            bbox={
                                "x1": float(x1),
                                "y1": float(y1),
                                "x2": float(x2),
                                "y2": float(y2),
                                "width": float(x2 - x1),
                                "height": float(y2 - y1),
                            },
                            diagnostics={
                                "modelFile": model_name,
                                "bboxWidthNorm": float(max(0.0, (x2 - x1) / max(width, 1))),
                                "bboxHeightNorm": float(max(0.0, (y2 - y1) / max(height, 1))),
                            },
                        )
                    )
        return predictions

    def _accept_class(self, model_role: str, class_id: int, class_name: str) -> bool:
        normalized = class_name.lower().replace("-", "_").replace(" ", "_")
        if model_role == "ball_detector":
            return class_id == 1 or normalized in {"ball", "cricket_ball"} or class_id == self.settings.sports_ball_class_id
        if model_role == "stump_detector":
            return normalized in {"stump", "stumps", "wicket", "wickets"} or class_id == 0
        return False

    def _timestamp_for_frame(self, frame_index: int, metadata: DeliveryMetadata) -> float | None:
        if not metadata.frameTimestamps:
            return None
        max_index = len(metadata.frameTimestamps) - 1
        return float(metadata.frameTimestamps[min(max(0, frame_index), max_index)])
