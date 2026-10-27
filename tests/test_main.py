"""Headless game-loop regressions; no camera or hand model is opened."""
from __future__ import annotations

import queue
import time

import pytest

from reflex import main
from reflex.gesture import HandsState
from reflex.tracker import HandObservation, TrackerResult


@pytest.mark.parametrize("age_ms", [100, 500, 501, 5000, 5001])
def test_game_loop_prunes_hands_in_tracker_clock(
    monkeypatch: pytest.MonkeyPatch, age_ms: int
) -> None:
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    # Camera/tracker timestamps are milliseconds of monotonic system uptime,
    # independent of the time since pygame.init().
    monkeypatch.setattr(time, "monotonic", lambda: 1000.0)
    hands = HandsState()
    state = hands._ensure("Right")
    # Deliver a synthetic result through the real game-loop queue consumer.
    # No camera or MediaPipe detector runs.
    result = TrackerResult(
        hands=(HandObservation(((100, 100),) * 9, "Right", 1.0),),
        timestamp_ms=1_000_000 - age_ms,
        latency_ms=0.0,
    )
    queue_factory = queue.Queue
    created = 0

    def make_queue(*args, **kwargs):
        nonlocal created
        created += 1
        output = queue_factory(*args, **kwargs)
        if created == 3:  # frame_queue, tracker_in, then tracker_out
            output.put_nowait(result)
        return output

    monkeypatch.setattr(main.queue, "Queue", make_queue)
    monkeypatch.setattr(main, "HandsState", lambda **kwargs: hands)
    monkeypatch.setattr(main.persistence, "load_best_score", lambda: 0)

    assert main.run(smoke_test_frames=2, use_camera=False, use_tracker=False) == 0

    if age_ms > hands.delete_after_ms:
        assert "Right" not in hands.by_handedness
    elif age_ms > hands.timeout_ms:
        assert not state.trail
        assert state.filter_x._t_prev is None
        assert state.filter_y._t_prev is None
    else:
        assert len(state.trail) == 1
        assert state.filter_x._t_prev == result.timestamp_ms / 1000.0
        assert state.trail[-1][:2] == (100.0, 100.0)
