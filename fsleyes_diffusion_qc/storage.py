"""Crash-conscious JSON persistence and single-writer locking."""

from __future__ import annotations

import fcntl
import json
import os
import socket
import tempfile
from pathlib import Path
from typing import Any


class ProjectLockedError(RuntimeError):
    """Raised when another process currently owns a QC project lock."""


class ProjectLock:
    """An advisory lock released automatically if the process exits."""

    def __init__(self, project_path: str | Path) -> None:
        self.path = Path(f"{project_path}.lock")
        self._stream = None

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        stream = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            stream.close()
            raise ProjectLockedError(f"QC project is already open for writing: {self.path}") from exc
        stream.seek(0)
        stream.truncate()
        stream.write(json.dumps({"pid": os.getpid(), "host": socket.gethostname()}))
        stream.flush()
        os.fsync(stream.fileno())
        self._stream = stream

    def release(self) -> None:
        if self._stream is not None:
            fcntl.flock(self._stream.fileno(), fcntl.LOCK_UN)
            self._stream.close()
            self._stream = None

    def __enter__(self) -> ProjectLock:
        self.acquire()
        return self

    def __exit__(self, *_: object) -> None:
        self.release()


class ProjectStore:
    """Read and atomically replace a project JSON file with a recoverable backup."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.backup_path = Path(f"{self.path}.bak")

    def load(self) -> dict[str, Any]:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as primary_error:
            try:
                recovered = json.loads(self.backup_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                raise primary_error
            self.save(recovered, keep_backup=False)
            return recovered

    def save(self, data: dict[str, Any], *, keep_backup: bool = True) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if keep_backup and self.path.exists():
            self._atomic_write(self.backup_path, self.path.read_bytes())
        payload = json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        self._atomic_write(self.path, payload.encode("utf-8"))

    @staticmethod
    def _atomic_write(path: Path, payload: bytes) -> None:
        fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        temp_path = Path(temp_name)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_path, path)
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            temp_path.unlink(missing_ok=True)