"""Slice juice: particle bursts and spinning fruit halves.

Independent of the game loop; the loop owns the lists and calls update / draw
each frame, plus emit_* helpers when a slice happens.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

import pygame

from reflex import config
from reflex.fruit import Fruit


@dataclass
class Particle:
    x: float
    y: float
    vx: float
    vy: float
    color: tuple[int, int, int]
    radius: int
    life_s: float
    max_life_s: float

    def update(self, dt: float) -> None:
        self.vy += config.GRAVITY_PX_S2 * 0.55 * dt
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.life_s -= dt

    @property
    def alive(self) -> bool:
        return self.life_s > 0.0

    def draw(self, surface: pygame.Surface) -> None:
        # Caller must hand in a SRCALPHA surface; the per-particle alpha needs it.
        if not self.alive:
            return
        alpha = max(0, min(255, int(255 * (self.life_s / self.max_life_s))))
        col = (self.color[0], self.color[1], self.color[2], alpha)
        pygame.draw.circle(surface, col, (int(self.x), int(self.y)), self.radius)


# Keep halves around briefly after they leave the screen so slow ones drifting
# near the bottom edge do not pop out of existence mid-fall.
_HALF_OFFSCREEN_GRACE_PX = 200


@dataclass
class FruitHalf:
    x: float
    y: float
    vx: float
    vy: float
    color: tuple[int, int, int]
    radius: int
    angle: float
    angular_velocity: float
    life_s: float
    max_life_s: float

    def update(self, dt: float) -> None:
        self.vy += config.GRAVITY_PX_S2 * dt
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.angle += self.angular_velocity * dt
        self.life_s -= dt

    @property
    def alive(self) -> bool:
        if self.life_s <= 0.0:
            return False
        return self.y - self.radius < config.WINDOW_H + _HALF_OFFSCREEN_GRACE_PX

    def draw(self, surface: pygame.Surface) -> None:
        if not self.alive:
            return
        n_segments = 14
        points: list[tuple[int, int]] = []
        for i in range(n_segments + 1):
            theta = self.angle + i * math.pi / n_segments
            px = self.radius * math.cos(theta)
            py = self.radius * math.sin(theta)
            points.append((int(self.x + px), int(self.y + py)))
        pygame.draw.polygon(surface, self.color, points)
        outline = tuple(max(0, c - 50) for c in self.color)
        pygame.draw.polygon(surface, outline, points, width=2)


def emit_burst(
    particles: list[Particle],
    x: float,
    y: float,
    color: tuple[int, int, int],
    rng: random.Random | None = None,
    count: int = 14,
) -> None:
    r = rng if rng is not None else random
    for _ in range(count):
        angle = r.uniform(0.0, 2.0 * math.pi)
        speed = r.uniform(200.0, 520.0)
        vx = math.cos(angle) * speed
        vy = math.sin(angle) * speed - 80.0  # slight upward bias
        life = r.uniform(0.35, 0.7)
        particles.append(
            Particle(
                x=x,
                y=y,
                vx=vx,
                vy=vy,
                color=color,
                radius=r.choice((3, 4, 5)),
                life_s=life,
                max_life_s=life,
            )
        )


def emit_halves(
    halves: list[FruitHalf],
    fruit: Fruit,
    slash_dx: float,
    slash_dy: float,
    rng: random.Random | None = None,
) -> None:
    """Spawn two semicircle halves kicked perpendicular to the slash.

    Bombs explode (burst only) rather than flying apart; callers should not
    emit halves for bomb slices.
    """
    if fruit.is_bomb:
        return
    r = rng if rng is not None else random
    seg_len = math.hypot(slash_dx, slash_dy)
    if seg_len < 1e-6:
        slash_dx, slash_dy, seg_len = 1.0, 0.0, 1.0
    perp_x = -slash_dy / seg_len
    perp_y = slash_dx / seg_len
    kick = 220.0
    # The half's initial rotation aligns its flat edge with the slash so the
    # cut visually matches the swipe direction.
    base_angle = math.atan2(slash_dy, slash_dx)
    for sign in (+1.0, -1.0):
        halves.append(
            FruitHalf(
                x=fruit.x,
                y=fruit.y,
                vx=fruit.vx + sign * perp_x * kick,
                vy=fruit.vy + sign * perp_y * kick - 60.0,
                color=fruit.color,
                radius=fruit.radius,
                angle=base_angle + (0.0 if sign > 0 else math.pi),
                angular_velocity=r.uniform(-6.0, 6.0),
                life_s=1.2,
                max_life_s=1.2,
            )
        )
