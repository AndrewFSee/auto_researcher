"""Shared pytest fixtures."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest

_REPO_TMP = Path(__file__).resolve().parent.parent / ".pytest_tmp"


@pytest.fixture
def repo_tmp_path():
    """
    Temporary directory inside the repository (``.pytest_tmp/``, git-ignored).

    On some Windows machines the OS temp directory is not listable, which
    breaks pytest's built-in ``tmp_path``. Tests that write files use this
    fixture instead so they run everywhere.
    """
    _REPO_TMP.mkdir(parents=True, exist_ok=True)
    path = Path(tempfile.mkdtemp(prefix="t_", dir=str(_REPO_TMP)))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)
