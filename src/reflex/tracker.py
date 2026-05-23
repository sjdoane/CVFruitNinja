"""MediaPipe HandLandmarker in LIVE_STREAM mode, on its own thread.

Pulls Frames off an in_queue, calls detect_async with monotonic ms
timestamps, and the result callback publishes a TrackerResult downstream.
"""
from __future__ import annotations

import queue
import sys
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core import base_options as mp_base

from reflex import config
from reflex.camera import Frame
from reflex.util import publish_latest


@dataclass(frozen=True)
class HandObservation:
    """One hand's 21 landmarks in canvas pixel space, plus handedness."""

    landmarks_px: tuple[tuple[int, int], ...]
    handedness: str
    score: float


@dataclass(frozen=True)
class TrackerResult:
    hands: tuple[HandObservation, ...]
    timestamp_ms: int
    # End-to-end latency from frame capture to result callback, in ms.
    latency_ms: float


LM_INDEX_TIP = 8

_TS_MAP_LIMIT = 128


class TrackerThread(threading.Thread):
    def __init__(
        self,
        in_queue: queue.Queue[Frame],
        out_queue: queue.Queue[TrackerResult],
        model_path: Path,
        num_hands: int = 2,
        canvas_size: tuple[int, int] | None = None,
    ) -> None:
        super().__init__(name="TrackerThread", daemon=True)
        self._in = in_queue
        self._out = out_queue
        self._model_path = model_path
        self._num_hands = num_hands
        cw, ch = canvas_size if canvas_size is not None else (config.WINDOW_W, config.WINDOW_H)
        self._canvas_w, self._canvas_h = cw, ch
        self._stop_event = threading.Event()
        self._last_ts_ms = 0
        # Map send-timestamp back to capture-timestamp so we can compute end
        # to end latency in the callback. OrderedDict for O(1) eviction.
        self._capture_ts_lock = threading.Lock()
        self._capture_ts_by_send: OrderedDict[int, int] = OrderedDict()

        self.frames_received = 0
        self.frames_sent_to_detector = 0
        self.callbacks_received = 0
        self.detect_failures = 0

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        options = vision.HandLandmarkerOptions(
            base_options=mp_base.BaseOptions(model_asset_path=str(self._model_path)),
            running_mode=vision.RunningMode.LIVE_STREAM,
            num_hands=self._num_hands,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            result_callback=self._on_result,
        )
        with vision.HandLandmarker.create_from_options(options) as landmarker:
            while not self._stop_event.is_set():
                try:
                    frame = self._in.get(timeout=0.05)
                except queue.Empty:
                    continue
                self.frames_received += 1
                ts = frame.timestamp_ms
                if ts <= self._last_ts_ms:
                    ts = self._last_ts_ms + 1
                self._last_ts_ms = ts

                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame.rgb)
                with self._capture_ts_lock:
                    self._capture_ts_by_send[ts] = frame.timestamp_ms
                    while len(self._capture_ts_by_send) > _TS_MAP_LIMIT:
                        self._capture_ts_by_send.popitem(last=False)
                try:
                    landmarker.detect_async(mp_image, ts)
                    self.frames_sent_to_detector += 1
                except Exception as exc:
                    self.detect_failures += 1
                    print(f"detect_async failed: {exc}", file=sys.stderr)
                    with self._capture_ts_lock:
                        self._capture_ts_by_send.pop(ts, None)

            # Drain in-flight callbacks before letting the landmarker close,
            # otherwise the callback may fire against a dead MediaPipe object.
            deadline = time.monotonic() + 0.2
            while (
                self.callbacks_received < self.frames_sent_to_detector
                and time.monotonic() < deadline
            ):
                time.sleep(0.005)

    def _on_result(
        self,
        result: vision.HandLandmarkerResult,
        output_image: mp.Image,
        timestamp_ms: int,
    ) -> None:
        self.callbacks_received += 1
        now_ms = int(time.monotonic() * 1000)
        with self._capture_ts_lock:
            capture_ms = self._capture_ts_by_send.pop(timestamp_ms, None)
            # Evict any older entries; they belonged to frames MediaPipe dropped.
            while self._capture_ts_by_send:
                oldest = next(iter(self._capture_ts_by_send))
                if oldest >= timestamp_ms:
                    break
                self._capture_ts_by_send.popitem(last=False)
        latency = float(now_ms - capture_ms) if capture_ms is not None else 0.0

        hands_out: list[HandObservation] = []
        for hand_idx, landmarks in enumerate(result.hand_landmarks):
            px: list[tuple[int, int]] = []
            for lm in landmarks:
                x = int(round(lm.x * self._canvas_w))
                y = int(round(lm.y * self._canvas_h))
                px.append((x, y))
            label = "Unknown"
            score = 0.0
            if hand_idx < len(result.handedness) and result.handedness[hand_idx]:
                top = result.handedness[hand_idx][0]
                label = top.category_name
                score = float(top.score)
            hands_out.append(
                HandObservation(landmarks_px=tuple(px), handedness=label, score=score)
            )

        snapshot = TrackerResult(
            hands=tuple(hands_out),
            timestamp_ms=int(timestamp_ms),
            latency_ms=latency,
        )
        publish_latest(self._out, snapshot)
