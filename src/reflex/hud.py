"""HUD rendering: score, lives, combos. No game logic lives here."""
from __future__ import annotations

import pygame

from reflex import config


def draw_score(surface: pygame.Surface, font: pygame.font.Font, score: int) -> None:
    label = font.render(f"{score}", True, config.TEXT_COLOR)
    shadow = font.render(f"{score}", True, (0, 0, 0))
    surface.blit(shadow, (20, 18))
    surface.blit(label, (18, 16))


def draw_lives(
    surface: pygame.Surface,
    lives: int,
    max_lives: int = 3,
    right_margin: int = 24,
    top: int = 18,
    spacing: int = 36,
) -> None:
    for i in range(max_lives):
        cx = surface.get_width() - right_margin - i * spacing
        cy = top + 12
        filled = i < lives
        _draw_heart(surface, (cx, cy), filled)


def _draw_heart(surface: pygame.Surface, center: tuple[int, int], filled: bool) -> None:
    cx, cy = center
    color = (235, 70, 90) if filled else (90, 90, 100)
    pygame.draw.circle(surface, color, (cx - 7, cy - 4), 9)
    pygame.draw.circle(surface, color, (cx + 7, cy - 4), 9)
    pygame.draw.polygon(
        surface,
        color,
        [(cx - 14, cy - 1), (cx + 14, cy - 1), (cx, cy + 14)],
    )


def draw_combo(
    surface: pygame.Surface, font: pygame.font.Font, combo_count: int, multiplier: int
) -> None:
    text = f"COMBO {combo_count}   +{multiplier}x"
    label = font.render(text, True, (255, 220, 90))
    shadow = font.render(text, True, (0, 0, 0))
    rect = label.get_rect(center=(surface.get_width() // 2, 36))
    surface.blit(shadow, rect.move(2, 2))
    surface.blit(label, rect)


def draw_game_over(
    surface: pygame.Surface,
    title_font: pygame.font.Font,
    body_font: pygame.font.Font,
    final_score: int,
    best_score: int,
    is_new_best: bool,
) -> None:
    w, h = surface.get_size()
    overlay = pygame.Surface((w, h), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 170))
    surface.blit(overlay, (0, 0))

    title = title_font.render("GAME OVER", True, (250, 70, 90))
    surface.blit(title, title.get_rect(center=(w // 2, h // 2 - 100)))

    score_label = body_font.render(f"final score  {final_score}", True, (240, 240, 240))
    surface.blit(score_label, score_label.get_rect(center=(w // 2, h // 2 - 20)))

    if is_new_best:
        best_label = body_font.render(f"NEW BEST  {best_score}", True, (255, 220, 90))
    else:
        best_label = body_font.render(f"best  {best_score}", True, (190, 190, 200))
    surface.blit(best_label, best_label.get_rect(center=(w // 2, h // 2 + 30)))

    prompt = body_font.render("press SPACE to play again", True, (200, 200, 210))
    surface.blit(prompt, prompt.get_rect(center=(w // 2, h // 2 + 100)))


def draw_menu(
    surface: pygame.Surface,
    title_font: pygame.font.Font,
    body_font: pygame.font.Font,
    best_score: int,
    show_camera_hint: bool = False,
) -> None:
    w, h = surface.get_size()
    overlay = pygame.Surface((w, h), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 120))
    surface.blit(overlay, (0, 0))

    title = title_font.render("REFLEX", True, (80, 220, 255))
    surface.blit(title, title.get_rect(center=(w // 2, h // 2 - 80)))

    tagline = body_font.render("slice with your hand", True, (235, 235, 245))
    surface.blit(tagline, tagline.get_rect(center=(w // 2, h // 2 - 10)))

    prompt = body_font.render("press SPACE to start", True, (235, 235, 245))
    surface.blit(prompt, prompt.get_rect(center=(w // 2, h // 2 + 60)))

    if best_score > 0:
        best = body_font.render(f"best  {best_score}", True, (190, 190, 200))
        surface.blit(best, best.get_rect(center=(w // 2, h // 2 + 110)))

    if show_camera_hint:
        hint = body_font.render("waiting for camera...", True, (255, 180, 90))
        surface.blit(hint, hint.get_rect(center=(w // 2, h - 60)))


def draw_debug(
    surface: pygame.Surface,
    body_font: pygame.font.Font,
    render_fps: float,
    render_ms: float,
    camera_fps: float,
    tracker_fps: float,
    tracker_latency_ms: float,
    drops_per_s: float,
) -> None:
    lines = [
        f"render  {render_fps:5.1f} fps  {render_ms:5.1f} ms",
        f"camera  {camera_fps:5.1f} fps",
        f"tracker {tracker_fps:5.1f} fps  {tracker_latency_ms:5.1f} ms",
        f"drops   {drops_per_s:5.1f} /s",
    ]
    pad = 8
    line_h = body_font.get_linesize()
    w = max(body_font.size(line)[0] for line in lines) + pad * 2
    h = line_h * len(lines) + pad * 2
    x = surface.get_width() - w - 16
    y = surface.get_height() - h - 16
    bg = pygame.Surface((w, h), pygame.SRCALPHA)
    bg.fill((0, 0, 0, 160))
    surface.blit(bg, (x, y))
    for i, line in enumerate(lines):
        label = body_font.render(line, True, (220, 230, 240))
        surface.blit(label, (x + pad, y + pad + i * line_h))
