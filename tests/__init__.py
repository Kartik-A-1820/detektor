"""Shared unittest helpers for detektor."""

from __future__ import annotations

import os
import shutil
import tempfile
import uuid
from pathlib import Path


_TEST_TMP_ROOT = Path(__file__).resolve().parents[1] / ".tmp_testdata"
_TEST_TMP_ROOT.mkdir(parents=True, exist_ok=True)

for _env_var in ("TMP", "TEMP", "TMPDIR"):
    os.environ[_env_var] = str(_TEST_TMP_ROOT)
tempfile.tempdir = str(_TEST_TMP_ROOT)


def get_test_tmp_root() -> Path:
    """Return the repo-local writable temp root used by tests."""
    return _TEST_TMP_ROOT


def _next_temp_path(
    *,
    prefix: str = "tmp",
    suffix: str = "",
    directory: Path | None = None,
) -> Path:
    """Generate a unique temp path beneath the repo-local writable root."""
    target_dir = directory or _TEST_TMP_ROOT
    target_dir.mkdir(parents=True, exist_ok=True)
    while True:
        candidate = target_dir / f"{prefix}{uuid.uuid4().hex}{suffix}"
        if not candidate.exists():
            return candidate


def _mkdtemp(suffix: str = "", prefix: str = "tmp", dir: str | None = None) -> str:
    """Create temp directories without relying on the platform tempfile backend."""
    path = _next_temp_path(prefix=prefix, suffix=suffix, directory=Path(dir) if dir else _TEST_TMP_ROOT)
    path.mkdir(parents=True, exist_ok=False)
    return str(path)


class _DeleteOnCloseFile:
    """Small wrapper that deletes the temp file when closed."""

    def __init__(self, handle, path: Path) -> None:
        self._handle = handle
        self.name = str(path)

    def __getattr__(self, attr):
        return getattr(self._handle, attr)

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.close()
        Path(self.name).unlink(missing_ok=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False


def _named_temporary_file(
    mode: str = "w+b",
    buffering: int = -1,
    encoding: str | None = None,
    newline: str | None = None,
    suffix: str = "",
    prefix: str = "tmp",
    dir: str | None = None,
    delete: bool = True,
    errors: str | None = None,
):
    """Create named temp files under the repo-local writable root."""
    path = _next_temp_path(prefix=prefix, suffix=suffix, directory=Path(dir) if dir else _TEST_TMP_ROOT)
    handle = open(path, mode, buffering=buffering, encoding=encoding, newline=newline, errors=errors)
    if delete:
        return _DeleteOnCloseFile(handle, path)
    return handle


class _TemporaryDirectory:
    """Simplified TemporaryDirectory compatible with the tests in this repo."""

    def __init__(
        self,
        suffix: str = "",
        prefix: str = "tmp",
        dir: str | None = None,
        ignore_cleanup_errors: bool = False,
    ) -> None:
        del ignore_cleanup_errors
        self.name = _mkdtemp(suffix=suffix, prefix=prefix, dir=dir)

    def cleanup(self) -> None:
        shutil.rmtree(self.name, ignore_errors=True)

    def __enter__(self) -> str:
        return self.name

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.cleanup()
        return False


tempfile.mkdtemp = _mkdtemp
tempfile.NamedTemporaryFile = _named_temporary_file
tempfile.TemporaryDirectory = _TemporaryDirectory


def make_test_temp_dir(prefix: str = "tmp") -> str:
    """Create a temp directory under the repo-local writable temp root."""
    return _mkdtemp(prefix=prefix)
