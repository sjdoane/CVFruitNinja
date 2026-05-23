"""Threaded webcam capture. Publishes the latest mirrored RGB frame.

Mirroring happens once here so all downstream math runs in the same space
the player sees on screen.
"""
from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np

from reflex import config
from reflex.util import publish_latest


@dataclass(frozen=True)
class Frame:
    """Mirrored RGB frame plus capture timestamp.

    The rgb array is a fresh allocation per frame, so consumers can hold it
    across an async tracker callback without torn pixels.
    """

    rgb: np.ndarray  # (H, W, 3) uint8, C-contiguous, RGB
    timestamp_ms: int


class CameraThread(threading.Thread):
    """Reads frames from a webcam and publishes the latest one to a queue.

    The queue has maxsize=1 and uses drop-on-full semantics: every new frame
    replaces the previous one. Consumers should treat get_nowait failures as
    "no new frame yet" and reuse their previous frame.
    """

    def __init__(
        self,
        out: queue.Queue[Frame],
        device_index: int = 0,
        request_size: tuple[int, int] = (config.WINDOW_W, config.WINDOW_H),
    ) -> None:
        super().__init__(name="CameraThread", daemon=True)
        self._out = out
        self._device_index = device_index
        self._request_size = request_size
        self._stop_event = threading.Event()
        self._open_error: Exception | None = None
        self._opened = threading.Event()
        self.frames_published = 0
        self.read_failures = 0

    # Stop reading after this many consecutive failures rather than spinning
    # forever at 200 Hz on a dead camera.
    MAX_CONSECUTIVE_FAILURES = 100

    def stop(self) -> None:
        self._stop_event.set()

    def wait_until_opened(self, timeout: float = 5.0) -> bool:
        return self._opened.wait(timeout)

    @property
    def open_error(self) -> Exception | None:
        return self._open_error

    def run(self) -> None:
        try:
            cap = cv2.VideoCapture(self._device_index, cv2.CAP_DSHOW)
        except Exception as exc:
            self._open_error = exc
            self._opened.set()
            return
        if not cap.isOpened():
            self._open_error = RuntimeError(
                f"could not open webcam at index {self._device_index} via CAP_DSHOW"
            )
            self._opened.set()
            return

        req_w, req_h = self._request_size
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, req_w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, req_h)
        # FPS hint is advisory; most DirectShow drivers ignore it and run ~28-30.
        cap.set(cv2.CAP_PROP_FPS, 30)
        self._opened.set()

        consecutive_failures = 0
        try:
            while not self._stop_event.is_set():
                ok, bgr = cap.read()
                if not ok or bgr is None:
                    self.read_failures += 1
                    consecutive_failures += 1
                    if consecutive_failures >= self.MAX_CONSECUTIVE_FAILURES:
                        self._open_error = RuntimeError(
                            f"camera read failed {consecutive_failures} times in a row"
                        )
                        return
                    time.sleep(0.005)
                    continue
                consecutive_failures = 0

                # Mirror once, then convert. cv2.flip with code 1 is a horizontal
                # mirror; the result is contiguous so frombuffer is happy.
                mirrored = cv2.flip(bgr, 1)
                if mirrored.shape[1] != req_w or mirrored.shape[0] != req_h:
                    mirrored = cv2.resize(mirrored, (req_w, req_h), interpolation=cv2.INTER_LINEAR)
                rgb = cv2.cvtColor(mirrored, cv2.COLOR_BGR2RGB)

                frame = Frame(rgb=rgb, timestamp_ms=int(time.monotonic() * 1000))
                publish_latest(self._out, frame)
                self.frames_published += 1
        finally:
            cap.release()
