from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import BinaryIO

from app.config import Settings
from app.schemas import DeliveryJobStatus, DeliveryMetadata, JobStatus


class LocalDeliveryStore:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.settings.upload_dir.mkdir(parents=True, exist_ok=True)
        self.settings.result_dir.mkdir(parents=True, exist_ok=True)

    def job_dir(self, job_id: str) -> Path:
        path = self.settings.upload_dir / job_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save_upload(self, job_id: str, file_obj: BinaryIO, filename: str) -> Path:
        safe_name = Path(filename or "delivery.bin").name
        target = self.job_dir(job_id) / safe_name
        with target.open("wb") as output:
            shutil.copyfileobj(file_obj, output)
        return target

    def save_metadata(self, job_id: str, metadata: DeliveryMetadata) -> Path:
        target = self.job_dir(job_id) / "metadata.json"
        target.write_text(metadata.model_dump_json(indent=2), encoding="utf-8")
        return target

    def write_status(self, status: DeliveryJobStatus) -> None:
        target = self.settings.result_dir / f"{status.jobId}.json"
        target.write_text(status.model_dump_json(indent=2), encoding="utf-8")

    def read_status(self, job_id: str) -> DeliveryJobStatus | None:
        target = self.settings.result_dir / f"{job_id}.json"
        if not target.exists():
            return None
        raw = json.loads(target.read_text(encoding="utf-8"))
        return DeliveryJobStatus.model_validate(raw)

    def set_status(self, job_id: str, status: JobStatus, error: str | None = None) -> None:
        current = self.read_status(job_id)
        self.write_status(
            DeliveryJobStatus(
                jobId=job_id,
                status=status,
                result=current.result if current else None,
                error=error,
            )
        )
