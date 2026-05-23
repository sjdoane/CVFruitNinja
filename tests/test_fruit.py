"""Tests for fruit physics and slash math."""
from __future__ import annotations

import random

from reflex import config
from reflex.fruit import Fruit, spawn_fruit
from reflex.gesture import segment_circle_intersects, segment_length


def test_segment_through_circle_intersects() -> None:
    # Segment crosses straight through the circle.
    assert segment_circle_intersects((0, 50), (200, 50), (100, 50), 20.0)


def test_segment_tangent_to_circle_just_intersects() -> None:
    # Segment grazes the top of a circle whose center is exactly radius below.
    assert segment_circle_intersects((0, 0), (200, 0), (100, 20), 20.0)


def test_segment_outside_circle_does_not_intersect() -> None:
    # Segment passes well above the circle.
    assert not segment_circle_intersects((0, 0), (200, 0), (100, 100), 20.0)


def test_segment_ending_before_circle_does_not_intersect() -> None:
    # Endpoints are both left of the circle; the closed segment must NOT
    # be projected past p2 (this is the closed-segment vs ray bug).
    assert not segment_circle_intersects((0, 50), (50, 50), (200, 50), 20.0)


def test_degenerate_zero_length_segment_uses_point_test() -> None:
    # p1 == p2; the test should still return True when the point is inside.
    assert segment_circle_intersects((100, 100), (100, 100), (100, 100), 5.0)
    assert not segment_circle_intersects((0, 0), (0, 0), (100, 100), 5.0)


def test_segment_length_is_euclidean() -> None:
    assert segment_length((0, 0), (3, 4)) == 5.0


def test_fruit_gravity_pulls_down() -> None:
    f = Fruit(x=100.0, y=100.0, vx=0.0, vy=0.0, radius=30, color=(1, 2, 3))
    f.update(dt=1.0)
    assert f.vy > 0
    assert f.y > 100.0


def test_fruit_off_screen_when_below_canvas() -> None:
    f = Fruit(x=100.0, y=float(config.WINDOW_H + 200), vx=0, vy=0, radius=30, color=(1, 2, 3))
    assert f.out_of_play
    assert f.fell_off_bottom


def test_fruit_off_side_is_out_of_play_but_not_fall_off_bottom() -> None:
    f = Fruit(x=-500.0, y=200.0, vx=-100, vy=0, radius=30, color=(1, 2, 3))
    assert f.out_of_play
    assert not f.fell_off_bottom


def test_spawn_fruit_starts_below_canvas_moving_up() -> None:
    rng = random.Random(42)
    f = spawn_fruit(rng)
    assert f.y > config.WINDOW_H
    assert f.vy < 0
    assert config.FRUIT_SPAWN_X_PAD <= f.x <= config.WINDOW_W - config.FRUIT_SPAWN_X_PAD


def test_spawn_fruit_color_and_radius_from_config() -> None:
    rng = random.Random(0)
    f = spawn_fruit(rng)
    assert f.color in config.FRUIT_COLORS
    assert f.radius in config.FRUIT_RADII


def test_spawn_interval_ramps_down_with_time() -> None:
    from reflex.main import _spawn_interval_ms

    at_zero = _spawn_interval_ms(0.0)
    at_half = _spawn_interval_ms(config.SPAWN_RAMP_SECONDS / 2)
    at_full = _spawn_interval_ms(config.SPAWN_RAMP_SECONDS)
    past_full = _spawn_interval_ms(config.SPAWN_RAMP_SECONDS * 2)
    assert at_zero == config.SPAWN_INTERVAL_START_MS
    assert at_zero > at_half > at_full
    assert at_full == past_full == float(config.SPAWN_INTERVAL_PEAK_MS)


def test_two_hands_both_contribute_segments_for_slashing() -> None:
    from reflex.gesture import HandsState
    from reflex.main import _try_slash

    h = HandsState(trail_maxlen=4)
    # Right hand sweeps across a fruit at (300, 300); Left across (900, 300).
    h.update("Right", (250, 300), 0)
    h.update("Right", (350, 300), 33)
    h.update("Left", (950, 300), 0)
    h.update("Left", (850, 300), 33)
    fruit_a = Fruit(x=300.0, y=300.0, vx=0.0, vy=0.0, radius=40, color=(180, 70, 90))
    fruit_b = Fruit(x=900.0, y=300.0, vx=0.0, vy=0.0, radius=40, color=(180, 70, 90))
    n, bomb_hit, events = _try_slash(h, [fruit_a, fruit_b])
    assert n == 2
    assert bomb_hit is False
    assert len(events) == 2
