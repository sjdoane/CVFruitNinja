"""Tests for combo math and bomb probability ramp."""
from __future__ import annotations

import random

from reflex import config
from reflex.fruit import bomb_probability, spawn_bomb, spawn_one
from reflex.main import _apply_combo


def test_single_slice_no_multiplier() -> None:
    add, combo, ts = _apply_combo(n_new=1, now_ms=1000, last_combo_ms=0, combo_count=0)
    assert add == 1
    assert combo == 1
    assert ts == 1000


def test_two_slices_in_window_doubles() -> None:
    add1, c1, t1 = _apply_combo(1, 1000, 0, 0)
    add2, c2, t2 = _apply_combo(1, 1000 + 100, t1, c1)
    assert add1 == 1
    assert c2 == 2
    assert add2 == 2  # the second slice scores 2x


def test_two_in_single_frame_uses_per_unit_multipliers() -> None:
    # A swipe that slices 2 fruits in one frame: first 1x, second 2x = 3.
    add, c, _ = _apply_combo(2, 1000, last_combo_ms=0, combo_count=0)
    assert c == 2
    assert add == 3


def test_three_in_single_frame_per_unit_multipliers() -> None:
    # Cold start, n=3 in one frame: 1 + 2 + 3 = 6.
    add, c, _ = _apply_combo(3, 1000, last_combo_ms=0, combo_count=0)
    assert c == 3
    assert add == 6


def test_three_in_window_triples() -> None:
    _, c, t = _apply_combo(1, 1000, 0, 0)
    _, c, t = _apply_combo(1, 1000 + 80, t, c)
    add, c, _ = _apply_combo(1, 1000 + 160, t, c)
    assert c == 3
    assert add == 3


def test_window_expiry_resets_combo() -> None:
    _, c, t = _apply_combo(1, 1000, 0, 0)
    add, c, _ = _apply_combo(1, 1000 + config.COMBO_WINDOW_MS + 50, t, c)
    assert c == 1
    assert add == 1


def test_zero_slices_does_not_touch_state() -> None:
    add, c, t = _apply_combo(0, 1000, last_combo_ms=500, combo_count=2)
    assert add == 0
    assert c == 2
    assert t == 500


def test_bomb_probability_zero_at_start() -> None:
    assert bomb_probability(0.0) == 0.0
    assert bomb_probability(-1.0) == 0.0


def test_bomb_probability_caps_at_max() -> None:
    assert bomb_probability(config.BOMB_RAMP_SECONDS) == config.BOMB_PROB_MAX
    assert bomb_probability(config.BOMB_RAMP_SECONDS * 10) == config.BOMB_PROB_MAX


def test_spawn_bomb_has_bomb_flag() -> None:
    f = spawn_bomb(random.Random(0))
    assert f.is_bomb is True
    assert f.color == config.BOMB_COLOR


def test_spawn_one_eventually_returns_a_bomb_after_ramp() -> None:
    # At high elapsed_s and many samples, at least one bomb should appear.
    rng = random.Random(1)
    has_bomb = any(spawn_one(60.0, rng).is_bomb for _ in range(500))
    assert has_bomb


def test_spawn_one_at_t_zero_never_returns_a_bomb() -> None:
    # bomb_probability(0) == 0, so 500 samples must all be fruits.
    rng = random.Random(2)
    assert not any(spawn_one(0.0, rng).is_bomb for _ in range(500))


def test_bomb_probability_midpoint_is_linear() -> None:
    mid = config.BOMB_RAMP_SECONDS / 2
    assert abs(bomb_probability(mid) - config.BOMB_PROB_MAX / 2) < 1e-9
