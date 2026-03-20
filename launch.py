#!/usr/bin/env python3
"""
Repository launcher for the carom simulator.

What it does:
1. Ensures required output directories exist.
2. Verifies that run.py exists in the repository root.
3. Runs run.py using the same Python interpreter that launched this script.
4. Passes through any command-line arguments.

Usage:
    python launch.py
    python launch.py --help
    python launch.py <any arguments intended for run.py>
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def ensure_directories(repo_root: Path) -> None:
    required_dirs = [
        repo_root / "outputs",
        repo_root / "outputs" / "animations",
        repo_root / "outputs" / "plots",
        repo_root / "outputs" / "tables",
    ]

    for directory in required_dirs:
        directory.mkdir(parents=True, exist_ok=True)


def main() -> int:
    repo_root = Path(__file__).resolve().parent
    run_script = repo_root / "run.py"

    if not run_script.is_file():
        print(f"Error: could not find {run_script}", file=sys.stderr)
        return 1

    ensure_directories(repo_root)

    command = [sys.executable, str(run_script), *sys.argv[1:]]

    try:
        completed = subprocess.run(command, cwd=repo_root)
        return completed.returncode
    except KeyboardInterrupt:
        print("\nExecution interrupted by user.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Error while launching run.py: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())