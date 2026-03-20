"""Pytest bootstrap for local source imports and offline test dependencies."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
SRC = REPO_ROOT / "src"
OFFLINE_DEPS = REPO_ROOT / "offline_test_deps"


if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

if importlib.util.find_spec("numpy") is None and OFFLINE_DEPS.exists():
    sys.path.insert(0, str(OFFLINE_DEPS))
