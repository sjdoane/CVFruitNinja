"""Tests for the one-euro filter and per-hand trail state."""
from __future__ import annotations

from reflex.gesture import HandsState, OneEuroFilter


def test_first_sample_is_passthrough() -> None:
    f = OneEuroFilter()
    assert f(0.0, 100.0) == 100.0


def test_stationary_input_converges() -> None:
    # Feeding a constant value many times should leave the output very close to it.
    # Loose tolerance because float arithmetic is platform-sensitive.
    f = OneEuroFilter(min_cutoff=1.0, beta=0.007)
    t = 0.0
    out = f(t, 100.0)
    for _ in range(100):
        t += 1 / 60.0
        out = f(t, 100.0)
    assert abs(out - 100.0) < 1e-4


def test_noisy_stationary_input_smooths_out() -> None:
    # Alternating +1/-1 noise around 100 should pull the filter toward 100.
    f = OneEuroFilter(min_cutoff=1.0, beta=0.007)
    t = 0.0
    f(t, 100.0)
    for i in range(200):
        t += 1 / 60.0
        sign = 1.0 if i % 2 == 0 else -1.0
        out = f(t, 100.0 + sign)
    assert abs(out - 100.0) < 0.5


def test_fast_input_tracks_without_excessive_lag() -> None:
    # A signal ramping at 600 px/s for half a second should land near the target.
    # A causal low-pass cannot exceed input, so we only assert it tracks closely.
    f = OneEuroFilter(min_cutoff=1.0, beta=0.007)
    t = 0.0
    f(t, 0.0)
    out = 0.0
    for _ in range(30):
        t += 1 / 60.0
        out = f(t, t * 600.0)
    assert out > 285.0


def test_dt_zero_or_negative_returns_previous() -> None:
    f = OneEuroFilter()
    f(1.0, 50.0)
    same = f(1.0, 999.0)
    assert same == 50.0
    same = f(0.5, 999.0)
    assert same == 50.0


def test_hands_state_tracks_smoothed_point() -> None:
    h = HandsState(trail_maxlen=4)
    sx1, sy1 = h.update("Right", (100, 100), 0)
    sx2, sy2 = h.update("Right", (101, 99), 33)
    # Smoothing keeps the second sample very close to the first.
    assert abs(sx2 - 100.0) < 2.0
    assert abs(sy2 - 100.0) < 2.0
    assert len(h.by_handedness["Right"].trail) == 2


def test_hands_state_prunes_stale_trail() -> None:
    h = HandsState(trail_maxlen=4, timeout_ms=100)
    h.update("Right", (100, 100), 0)
    h.update("Right", (102, 100), 33)
    h.prune(now_ms=200)
    assert len(h.by_handedness["Right"].trail) == 0


def test_hands_state_independent_per_label() -> None:
    h = HandsState()
    h.update("Right", (100, 100), 0)
    h.update("Left", (500, 500), 0)
    assert h.by_handedness["Right"].trail[-1][0] == 100.0
    assert h.by_handedness["Left"].trail[-1][0] == 500.0


def test_hands_state_rejects_unknown_label() -> None:
    h = HandsState()
    assert h.update("Unknown", (100, 100), 0) is None
    assert "Unknown" not in h.by_handedness


def test_hands_state_deletes_long_stale_entry() -> None:
    h = HandsState(timeout_ms=100, delete_after_ms=200)
    h.update("Right", (100, 100), 0)
    h.prune(now_ms=300)
    assert "Right" not in h.by_handedness


def test_hands_state_two_hands_have_independent_trails() -> None:
    h = HandsState(trail_maxlen=8)
    # Build a real swipe for each hand at different vertical bands.
    for step in range(4):
        ts = step * 33
        h.update("Right", (200 + step * 80, 200), ts)
        h.update("Left", (1000 - step * 80, 500), ts)
    right_trail = h.by_handedness["Right"].trail
    left_trail = h.by_handedness["Left"].trail
    assert len(right_trail) == 4
    assert len(left_trail) == 4
    # The Right trail is sweeping right at y=200; the Left is sweeping left at y=500.
    assert right_trail[-1][0] > right_trail[0][0]
    assert left_trail[-1][0] < left_trail[0][0]
    assert abs(right_trail[-1][1] - 200) < 5
    assert abs(left_trail[-1][1] - 500) < 5


def test_hands_state_one_hand_stops_other_keeps_going() -> None:
    h = HandsState(trail_maxlen=8, timeout_ms=100)
    # Both hands seen at t=0.
    h.update("Right", (200, 200), 0)
    h.update("Left", (1000, 500), 0)
    # Right keeps updating; Left stops.
    h.update("Right", (260, 200), 50)
    h.update("Right", (320, 200), 100)
    h.prune(now_ms=200)
    assert len(h.by_handedness["Right"].trail) == 3
    assert len(h.by_handedness["Left"].trail) == 0
