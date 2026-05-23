"""Fruit physics and rendering, plus bombs."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

import pygame

from reflex import config


@dataclass
class Fruit:
    """A circular target launched from below the canvas with gravity.

    `is_bomb` flips behavior: bomb-colored draw, slicing one is a game-over
    rather than a score. We keep this as a flag rather than a subclass so the
    spawner, physics, and slash math stay uniform.
    """

    x: float
    y: float
    vx: float
    vy: float
    radius: int
    color: tuple[int, int, int]
    is_bomb: bool = False
    alive: bool = True
    sliced: bool = False
    anim_phase: float = 0.0  # advances by dt; bomb fuse pulse reads it

    def update(self, dt: float) -> None:
        self.vy += config.GRAVITY_PX_S2 * dt
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.anim_phase += dt

    @property
    def fell_off_bottom(self) -> bool:
        return self.y - self.radius > config.WINDOW_H

    @property
    def out_of_play(self) -> bool:
        if self.fell_off_bottom:
            return True
        pad = config.FRUIT_SIDE_OFFSCREEN_PAD
        if self.x + self.radius < -pad or self.x - self.radius > config.WINDOW_W + pad:
            return True
        return False

    def draw(self, surface: pygame.Surface) -> None:
        if not self.alive or self.sliced:
            return
        cx, cy = int(self.x), int(self.y)
        if self.is_bomb:
            _draw_bomb(surface, cx, cy, self.radius, self.anim_phase)
            return
        # Body, then a darker outline for definition against the dimmed webcam.
        pygame.draw.circle(surface, self.color, (cx, cy), self.radius)
        outline = tuple(max(0, c - 50) for c in self.color)
        pygame.draw.circle(surface, outline, (cx, cy), self.radius, width=3)
        # Tiny stem nub so spheres do not look like blank dots.
        pygame.draw.line(
            surface,
            outline,
            (cx, cy - self.radius),
            (cx + max(2, self.radius // 6), cy - self.radius - 6),
            width=3,
        )


def _draw_bomb(surface: pygame.Surface, cx: int, cy: int, radius: int, phase: float) -> None:
    pygame.draw.circle(surface, config.BOMB_COLOR, (cx, cy), radius)
    pygame.draw.circle(surface, (0, 0, 0), (cx, cy), radius, width=3)
    # Highlight pip on the upper-left so the sphere reads as a sphere.
    pygame.draw.circle(surface, (90, 90, 95), (cx - radius // 3, cy - radius // 3), max(3, radius // 6))
    # Fuse and spark. The spark radius pulses on `phase` so it does not look static.
    fuse_top = (cx + radius // 2, cy - radius - 8)
    pygame.draw.line(surface, (160, 110, 60), (cx + radius // 4, cy - radius + 4), fuse_top, width=4)
    pulse = 4 + int(2 * (1 + math.sin(phase * 16.0)))
    pygame.draw.circle(surface, config.BOMB_FUSE_COLOR, fuse_top, pulse)


def _launch_params(rng: random.Random) -> tuple[float, float, float]:
    x = rng.uniform(config.FRUIT_SPAWN_X_PAD, config.WINDOW_W - config.FRUIT_SPAWN_X_PAD)
    bias_toward_center = (config.WINDOW_W / 2 - x) * config.FRUIT_SPAWN_X_BIAS
    vx = rng.uniform(*config.FRUIT_SPAWN_VX_RANGE) + bias_toward_center
    vy = rng.uniform(*config.FRUIT_SPAWN_VY_RANGE)
    return x, vx, vy


def spawn_fruit(rng: random.Random | None = None) -> Fruit:
    r = rng if rng is not None else random
    x, vx, vy = _launch_params(r)
    radius = r.choice(config.FRUIT_RADII)
    color = r.choice(config.FRUIT_COLORS)
    return Fruit(
        x=x,
        y=float(config.WINDOW_H + radius + 10),
        vx=vx,
        vy=vy,
        radius=radius,
        color=color,
    )


def spawn_bomb(rng: random.Random | None = None) -> Fruit:
    r = rng if rng is not None else random
    x, vx, vy = _launch_params(r)
    radius = r.choice(config.BOMB_RADII)
    return Fruit(
        x=x,
        y=float(config.WINDOW_H + radius + 10),
        vx=vx,
        vy=vy,
        radius=radius,
        color=config.BOMB_COLOR,
        is_bomb=True,
    )


def bomb_probability(elapsed_s: float) -> float:
    """0 at t=0 ramping to BOMB_PROB_MAX at BOMB_RAMP_SECONDS, then flat."""
    if elapsed_s <= 0.0:
        return 0.0
    if elapsed_s >= config.BOMB_RAMP_SECONDS:
        return config.BOMB_PROB_MAX
    return (elapsed_s / config.BOMB_RAMP_SECONDS) * config.BOMB_PROB_MAX


def spawn_one(elapsed_s: float, rng: random.Random | None = None) -> Fruit:
    r = rng if rng is not None else random
    return spawn_bomb(r) if r.random() < bomb_probability(elapsed_s) else spawn_fruit(r)


