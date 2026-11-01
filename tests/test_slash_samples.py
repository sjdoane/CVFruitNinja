"""A tracker swipe is an event, not a persistent collision surface."""
from __future__ import annotations

import pytest

from reflex.fruit import Fruit
from reflex.gesture import HandsState
from reflex.main import _try_slash


def swipe(hands: HandsState, label: str = "Right", timestamp_ms: int = 1030) -> None:
    state = hands._ensure(label)
    state.trail.extend([(0.0, 50.0, (timestamp_ms - 30) / 1000),
                        (100.0, 50.0, timestamp_ms / 1000)])
    state.last_seen_ms = timestamp_ms


def fruit(*, bomb: bool = False) -> Fruit:
    return Fruit(50, 50, 0, 0, 10, (255, 0, 0), is_bomb=bomb)


@pytest.mark.parametrize("bomb", [False, True])
def test_old_swipe_cannot_hit_target_entering_on_later_render(bomb: bool) -> None:
    hands = HandsState()
    swipe(hands)
    assert _try_slash(hands, [fruit()])[0] == 1
    later = fruit(bomb=bomb)
    assert _try_slash(hands, [later]) == (0, False, [])
    assert later.alive and not later.sliced


def test_swipe_is_consumed_even_when_no_fruit_intersected() -> None:
    hands = HandsState()
    swipe(hands)
    assert _try_slash(hands, []) == (0, False, [])
    later = fruit()
    assert _try_slash(hands, [later]) == (0, False, [])
    assert later.alive
    # Consumption must not remove the points used to draw the fading trail.
    assert len(hands.by_handedness["Right"].trail) == 2


def test_fresh_swipe_still_slices_multiple_fruit_and_detects_bomb() -> None:
    hands = HandsState()
    swipe(hands)
    count, bomb, events = _try_slash(hands, [fruit(), fruit(), fruit(bomb=True)])
    assert count == 2 and bomb
    assert len(events) == 3


def test_new_sample_and_other_hand_are_not_consumed_by_old_swipe() -> None:
    hands = HandsState()
    swipe(hands)
    assert _try_slash(hands, [fruit()])[0] == 1
    swipe(hands, "Left")  # same timestamp as Right, independent hand
    assert _try_slash(hands, [fruit()])[0] == 1
    swipe(hands, "Right", 1060)
    assert _try_slash(hands, [fruit()])[0] == 1
    assert _try_slash(hands, [fruit()]) == (0, False, [])


def test_fresh_left_hand_does_not_reactivate_old_right_path() -> None:
    hands = HandsState()
    swipe(hands, "Right")
    assert _try_slash(hands, []) == (0, False, [])
    swipe(hands, "Left", 1060)
    left = hands.by_handedness["Left"]
    left.trail.clear()
    left.trail.extend([(0.0, 150.0, 1.03), (100.0, 150.0, 1.06)])
    on_old_right = fruit()
    on_new_left = fruit()
    on_new_left.y = 150
    count, bomb, events = _try_slash(hands, [on_old_right, on_new_left])
    assert count == 1 and not bomb
    assert [event.fruit for event in events] == [on_new_left]
    assert on_old_right.alive


def test_stationary_new_sample_does_not_replay_previous_motion() -> None:
    hands = HandsState()
    swipe(hands)
    _try_slash(hands, [])
    state = hands.by_handedness["Right"]
    state.trail.append((100.0, 50.0, 1.06))
    state.last_seen_ms = 1060
    assert _try_slash(hands, [fruit()]) == (0, False, [])


@pytest.mark.parametrize("reset", ["timeout", "eviction", "restart"])
def test_fresh_swipe_after_clear_eviction_or_round_reset(reset: str) -> None:
    hands = HandsState()
    swipe(hands)
    assert _try_slash(hands, [fruit()])[0] == 1
    if reset == "restart":
        hands = HandsState()  # begin_playing creates a fresh registry
    else:
        hands.prune(1600 if reset == "timeout" else 6100)
        assert not hands.by_handedness.get("Right") or not hands.by_handedness["Right"].trail
    hands.update("Right", (0, 50), 7000)
    assert _try_slash(hands, [fruit()]) == (0, False, [])
    hands.update("Right", (1000, 50), 7030)
    assert _try_slash(hands, [fruit()])[0] == 1
    assert _try_slash(hands, [fruit()]) == (0, False, [])


@pytest.mark.parametrize("dx, dt", [(1.0, .03), (100.0, .1)])
def test_consumption_preserves_velocity_and_recency_gates(dx: float, dt: float) -> None:
    hands = HandsState()
    state = hands._ensure("Right")
    state.trail.extend([(0.0, 50.0, 1.0), (dx, 50.0, 1.0 + dt)])
    state.last_seen_ms = int((1.0 + dt) * 1000)
    target = fruit()
    assert _try_slash(hands, [target]) == (0, False, [])
    assert target.alive
