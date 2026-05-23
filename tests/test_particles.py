"""Tests for the particle and fruit-half spawn helpers and physics."""
from __future__ import annotations

import math
import random

from reflex.fruit import Fruit
from reflex.particles import FruitHalf, Particle, emit_burst, emit_halves


def test_emit_burst_creates_expected_count() -> None:
    particles: list[Particle] = []
    emit_burst(particles, 100.0, 100.0, (200, 50, 50), rng=random.Random(0), count=12)
    assert len(particles) == 12
    for p in particles:
        assert p.life_s > 0
        assert p.color == (200, 50, 50)


def test_particle_fades_to_dead() -> None:
    p = Particle(x=0, y=0, vx=0, vy=0, color=(1, 2, 3), radius=4, life_s=0.5, max_life_s=0.5)
    p.update(0.6)
    assert not p.alive


def test_emit_halves_produces_two_halves_with_opposite_velocities() -> None:
    fruit = Fruit(x=200.0, y=200.0, vx=0.0, vy=-100.0, radius=40, color=(180, 70, 90))
    halves: list[FruitHalf] = []
    # Slash going right (dx > 0, dy == 0). The perpendicular kick is purely vertical.
    emit_halves(halves, fruit, slash_dx=300.0, slash_dy=0.0, rng=random.Random(0))
    assert len(halves) == 2
    # One half gets +perp_y kick, the other -perp_y kick. Sum of perpendicular
    # velocities should cancel (within the shared inheritance plus upward bias).
    perp_sum = sum(h.vy - fruit.vy + 60.0 for h in halves)
    assert abs(perp_sum) < 1e-6


def test_emit_halves_skipped_for_bomb() -> None:
    bomb = Fruit(x=200.0, y=200.0, vx=0.0, vy=0.0, radius=40, color=(30, 30, 30), is_bomb=True)
    halves: list[FruitHalf] = []
    emit_halves(halves, bomb, slash_dx=300.0, slash_dy=0.0)
    assert halves == []


def test_fruit_half_dies_after_life_expires() -> None:
    h = FruitHalf(
        x=0, y=0, vx=0, vy=0, color=(1, 2, 3), radius=20,
        angle=0.0, angular_velocity=0.0, life_s=0.5, max_life_s=0.5,
    )
    h.update(0.6)
    assert not h.alive


def test_fruit_half_rotation_advances_with_dt() -> None:
    h = FruitHalf(
        x=0, y=0, vx=0, vy=0, color=(1, 2, 3), radius=20,
        angle=0.0, angular_velocity=math.pi, life_s=5.0, max_life_s=5.0,
    )
    h.update(0.5)
    assert abs(h.angle - math.pi * 0.5) < 1e-9
