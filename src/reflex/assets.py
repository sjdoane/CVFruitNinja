"""Locate and (if needed) fetch the MediaPipe hand-landmark model file."""
from __future__ import annotations

import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

from reflex import config

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)


def ensure_model(path: Path | None = None) -> Path:
    """Return the model path, downloading it once if it is missing."""
    target = path if path is not None else config.MODEL_PATH
    if target.exists() and target.stat().st_size > 0:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".part")
    print(f"fetching hand model to {target}", file=sys.stderr)
    try:
        urllib.request.urlretrieve(MODEL_URL, tmp)
    except (urllib.error.URLError, OSError) as exc:
        if tmp.exists():
            tmp.unlink()
        raise RuntimeError(
            f"could not fetch MediaPipe model from {MODEL_URL}: {exc}"
        ) from exc
    os.replace(tmp, target)
    return target
