"""Gesture math: one-euro smoothing, per-hand trail, slash collision.

The one-euro filter is a low-pass with a cutoff that adapts to velocity:
low at rest (kills jitter), high during fast motion (no lag). Pass sample
times in seconds, not frame indices.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

Point = tuple[float, float]


@dataclass
class OneEuroFilter:
    min_cutoff: float = 1.0
    beta: float = 0.007
    d_cutoff: float = 1.0
    _x_prev: float | None = field(default=None, init=False)
    _dx_prev: float = field(default=0.0, init=False)
    _t_prev: float | None = field(default=None, init=False)

    def __call__(self, t: float, x: float) -> float:
        if self._t_prev is None or self._x_prev is None:
            self._t_prev = t
            self._x_prev = x
            return x
        dt = t - self._t_prev
        if dt <= 0.0:
            return self._x_prev

        alpha_d = _smoothing_factor(dt, self.d_cutoff)
        dx = (x - self._x_prev) / dt
        dx_hat = alpha_d * dx + (1.0 - alpha_d) * self._dx_prev

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        alpha = _smoothing_factor(dt, cutoff)
        x_hat = alpha * x + (1.0 - alpha) * self._x_prev

        self._x_prev = x_hat
        self._dx_prev = dx_hat
        self._t_prev = t
        return x_hat

    def reset(self) -> None:
        self._x_prev = None
        self._dx_prev = 0.0
        self._t_prev = None


def _smoothing_factor(dt: float, cutoff: float) -> float:
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


@dataclass
class HandState:
    """Per-hand state: smoothing filters plus a recent trail of points."""

    filter_x: OneEuroFilter
    filter_y: OneEuroFilter
    trail: deque[tuple[float, float, float]]
    last_seen_ms: int = 0


_ACCEPTED_LABELS = ("Left", "Right")


@dataclass
class HandsState:
    """Per-hand state keyed by MediaPipe handedness label.

    Only "Left" or "Right" are accepted; low-confidence "Unknown" hands are
    skipped so the registry never grows without bound.
    """

    trail_maxlen: int = 12
    timeout_ms: int = 500
    # Hands not seen in this long get evicted from the registry entirely so a
    # long session does not accumulate dead buckets.
    delete_after_ms: int = 5000
    by_handedness: dict[str, HandState] = field(default_factory=dict)

    def _ensure(self, label: str) -> HandState:
        state = self.by_handedness.get(label)
        if state is None:
            state = HandState(
                filter_x=OneEuroFilter(),
                filter_y=OneEuroFilter(),
                trail=deque(maxlen=self.trail_maxlen),
            )
            self.by_handedness[label] = state
        return state

    def update(
        self, label: str, tip_xy: tuple[int, int], timestamp_ms: int
    ) -> tuple[float, float] | None:
        if label not in _ACCEPTED_LABELS:
            return None
        state = self._ensure(label)
        t_sec = timestamp_ms / 1000.0
        sx = state.filter_x(t_sec, float(tip_xy[0]))
        sy = state.filter_y(t_sec, float(tip_xy[1]))
        state.trail.append((sx, sy, t_sec))
        state.last_seen_ms = timestamp_ms
        return sx, sy

    def prune(self, now_ms: int) -> None:
        """Clear stale trails; delete buckets that have been quiet much longer.

        Filter state is reset on clear so re-acquisition does not start with
        stale derivative and ghost the user.
        """
        for label, state in list(self.by_handedness.items()):
            elapsed = now_ms - state.last_seen_ms
            if elapsed > self.delete_after_ms:
                del self.by_handedness[label]
                continue
            if elapsed > self.timeout_ms and state.trail:
                state.trail.clear()
                state.filter_x.reset()
                state.filter_y.reset()


def segment_circle_intersects(p1: Point, p2: Point, center: Point, radius: float) -> bool:
    """True if the closed segment p1-p2 passes within `radius` of `center`."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq < 1e-12:
        cdx = center[0] - p1[0]
        cdy = center[1] - p1[1]
        return cdx * cdx + cdy * cdy <= radius * radius
    fx = center[0] - p1[0]
    fy = center[1] - p1[1]
    t = (fx * dx + fy * dy) / seg_len_sq
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    closest_x = p1[0] + t * dx
    closest_y = p1[1] + t * dy
    cdx = center[0] - closest_x
    cdy = center[1] - closest_y
    return cdx * cdx + cdy * cdy <= radius * radius


def segment_length(p1: Point, p2: Point) -> float:
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])
