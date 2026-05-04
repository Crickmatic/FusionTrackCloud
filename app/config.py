from functools import lru_cache
import os
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    service_name: str = "fusiontrack-cloud"
    environment: str = "local"
    model_dir: Path = Field(default=Path("models"))
    upload_dir: Path = Field(default=Path("storage/uploads"))
    result_dir: Path = Field(default=Path("storage/results"))
    models: str = "cricket_ball_v2,cricket_stumps_v1"
    enable_yolo26s: bool = True
    enable_yolov8s: bool = False
    enable_cricket_models: bool = True
    enable_tracknet_adapter: bool = False
    sports_ball_class_id: int = 32
    yolo_confidence_threshold: float = 0.25
    device: str = "cuda:0"
    debug: bool = False
    max_upload_mb: int = 250
    max_clip_duration_sec: float = 8.0
    request_timeout_sec: float = 30.0
    warmup_on_startup: bool = True
    dense_release_window_seconds: float = 0.9
    moderate_frame_stride: int = 3
    min_temporal_candidates: int = 4
    max_candidate_jump: float = 0.28
    require_forward_motion: bool = True

    model_config = SettingsConfigDict(env_prefix="FUSIONTRACK_", env_file=".env", extra="ignore")

    @property
    def model_names(self) -> list[str]:
        names: list[str] = []
        for raw_name in self.models.split(","):
            name = raw_name.strip()
            if name:
                names.append(name)
        return names

    @property
    def yolo_model_paths(self) -> list[Path]:
        names = []
        for name in self.model_names:
            if name.startswith("cricket_"):
                continue
            names.append(self._normalize_model_name(name))
        return [self._resolve_model_path(name) for name in names]

    @property
    def cricket_model_paths(self) -> list[Path]:
        names = []
        for name in self.model_names:
            if name.startswith("cricket_"):
                names.append(self._normalize_model_name(name))
        return [self._resolve_model_path(name) for name in names]

    def _normalize_model_name(self, name: str) -> str:
        if Path(name).suffix:
            return name
        return f"{name}.pt"

    def _resolve_model_path(self, name: str) -> Path:
        candidates = [
            self.model_dir / name,
            Path(name),
            Path("..") / name,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return self.model_dir / name


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    if "FUSIONTRACK_CONF" in os.environ:
        settings.yolo_confidence_threshold = float(os.environ["FUSIONTRACK_CONF"])
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    settings.result_dir.mkdir(parents=True, exist_ok=True)
    settings.model_dir.mkdir(parents=True, exist_ok=True)
    return settings
