"""Tiny shared helpers."""
from __future__ import annotations

import queue
from typing import TypeVar

T = TypeVar("T")


def publish_latest(q: "queue.Queue[T]", item: T) -> None:
    """Drop-on-full publish: discard any pending item, then push the new one.

    Safe only when this is the sole producer for q; otherwise two producers can
    interleave and both lose their writes.
    """
    try:
        q.get_nowait()
    except queue.Empty:
        pass
    try:
        q.put_nowait(item)
    except queue.Full:
        pass
