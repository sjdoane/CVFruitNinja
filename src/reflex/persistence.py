"""Best-score persistence in the platform's per-user data dir.

A single int in a text file, atomically replaced on write. We never crash on
read or write errors; a missing/garbled file just shows best=0 this session.
"""
from __future__ import annotations

import os
from pathlib import Path

import platformdirs

APP_NAME = "Reflex"
FILENAME = "best_score.txt"


def best_score_path() -> Path:
    # appauthor=False skips the Windows-only company-name folder layer so the
    # path is %LOCALAPPDATA%\Reflex\best_score.txt rather than ...\Reflex\Reflex\... .
    return Path(platformdirs.user_data_dir(APP_NAME, appauthor=False)) / FILENAME


def load_best_score(path: Path | None = None) -> int:
    p = path if path is not None else best_score_path()
    try:
        if not p.exists():
            return 0
        return int(p.read_text(encoding="utf-8").strip())
    except (ValueError, OSError):
        return 0


def save_best_score(score: int, path: Path | None = None) -> bool:
    p = path if path is not None else best_score_path()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(str(int(score)), encoding="utf-8")
        os.replace(tmp, p)
        return True
    except OSError:
        return False
