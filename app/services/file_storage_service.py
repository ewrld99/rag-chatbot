from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import re
from typing import BinaryIO
from uuid import UUID, uuid4

from app.core.config import settings


class FileStorageError(RuntimeError):
    pass


class UploadTooLargeError(FileStorageError):
    pass


@dataclass(frozen=True)
class StagedFile:
    source_path: str
    target_path: str
    filename: str
    sha256: str
    size: int


class FileStorageService:
    """Local durable storage with every path confined to the uploads root."""

    chunk_size = 1024 * 1024

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root or settings.UPLOADS_DIR).resolve()
        self.staging_root = self.root / ".staging"
        self.failed_root = self.root / ".failed"
        self.trash_root = self.root / ".trash"
        for directory in (self.root, self.staging_root, self.failed_root, self.trash_root):
            directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def safe_filename(filename: str) -> str:
        name = Path(str(filename).replace("\\", "/")).name.strip().lower()
        name = re.sub(r"[^a-z0-9._-]+", "_", name).strip("._")
        return name or "document"

    def stage_stream(
        self,
        stream: BinaryIO,
        filename: str,
        *,
        max_bytes: int,
    ) -> StagedFile:
        safe_name = self.safe_filename(filename)
        stage = self.staging_root / f"{uuid4().hex}.{safe_name}.part"
        target = self._available_target(safe_name)
        digest = hashlib.sha256()
        size = 0
        try:
            with stage.open("xb") as output:
                while True:
                    chunk = stream.read(self.chunk_size)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > max_bytes:
                        raise UploadTooLargeError(
                            f"Upload exceeds the {max_bytes // (1024 * 1024)} MB limit"
                        )
                    digest.update(chunk)
                    output.write(chunk)
                output.flush()
                os.fsync(output.fileno())
        except Exception:
            stage.unlink(missing_ok=True)
            raise
        return StagedFile(str(stage), str(target), target.name, digest.hexdigest(), size)

    def stage_bytes(self, content: bytes, filename: str) -> StagedFile:
        from io import BytesIO

        return self.stage_stream(BytesIO(content), filename, max_bytes=max(len(content), 1))

    def promote(self, source_path: str, target_path: str) -> None:
        source = self.resolve(source_path)
        target = self.resolve(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and not source.exists():
            return
        if not source.is_file():
            raise FileStorageError(f"Staged source is missing: {source}")
        os.replace(source, target)

    def retain_failed(self, source_path: str, document_id: UUID, filename: str) -> str:
        source = self.resolve(source_path)
        target = self.failed_root / f"{document_id}.{self.safe_filename(filename)}"
        if source.exists():
            os.replace(source, target)
        elif not target.exists():
            raise FileStorageError(f"Failed source is missing: {source}")
        return str(target)

    def trash_path(self, document_id: UUID, filename: str) -> str:
        return str(self.trash_root / f"{document_id}.{self.safe_filename(filename)}")

    def move_to_trash(self, source_path: str, target_path: str) -> None:
        source = self.resolve(source_path)
        target = self.resolve(target_path)
        if target.exists() and not source.exists():
            return
        if source.exists():
            os.replace(source, target)

    def remove(self, path: str) -> None:
        self.resolve(path).unlink(missing_ok=True)

    def resolve(self, value: str | Path) -> Path:
        candidate = Path(value)
        if candidate.is_absolute():
            resolved = candidate.resolve()
        else:
            direct = candidate.resolve()
            try:
                direct.relative_to(self.root)
                resolved = direct
            except ValueError:
                resolved = (self.root / candidate).resolve()
        try:
            resolved.relative_to(self.root)
        except ValueError as exc:
            raise FileStorageError(f"Path escapes uploads root: {value}") from exc
        return resolved

    def _available_target(self, filename: str) -> Path:
        candidate = self.root / filename
        if not candidate.exists():
            return candidate
        stem = Path(filename).stem
        suffix = Path(filename).suffix
        return self.root / f"{stem}_{uuid4().hex[:10]}{suffix}"
