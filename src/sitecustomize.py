"""
Bootstrap local offline test dependencies when third-party packages are absent.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
OFFLINE_DEPS = REPO_ROOT / "offline_test_deps"


if importlib.util.find_spec("numpy") is None and OFFLINE_DEPS.exists():
    sys.path.insert(0, str(OFFLINE_DEPS))
