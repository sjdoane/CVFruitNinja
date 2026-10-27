"""Game entry point: window, threads, and the main loop."""
from __future__ import annotations

import argparse
import queue
import sys
import time
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pygame

from reflex import assets, config, hud, persistence
from reflex.camera import CameraThread, Frame
from reflex.fruit import Fruit, spawn_one
from reflex.gesture import HandsState, segment_circle_intersects, segment_length
from reflex.particles import FruitHalf, Particle, emit_burst, emit_halves
from reflex.tracker import LM_INDEX_TIP, TrackerResult, TrackerThread
from reflex.util import publish_latest


class Scene(Enum):
    MENU = "menu"
    PLAYING = "playing"
    GAME_OVER = "game_over"


@dataclass
class SliceEvent:
    fruit: Fruit
    p1: tuple[float, float]
    p2: tuple[float, float]


def _spawn_interval_ms(elapsed_s: float) -> float:
    """Linear ramp from SPAWN_INTERVAL_START_MS down to SPAWN_INTERVAL_PEAK_MS."""
    t = elapsed_s / config.SPAWN_RAMP_SECONDS
    if t <= 0.0:
        return float(config.SPAWN_INTERVAL_START_MS)
    if t >= 1.0:
        return float(config.SPAWN_INTERVAL_PEAK_MS)
    return (
        config.SPAWN_INTERVAL_START_MS * (1.0 - t)
        + config.SPAWN_INTERVAL_PEAK_MS * t
    )


def _try_slash(
    hands_state: HandsState, fruits: list[Fruit]
) -> tuple[int, bool, list[SliceEvent]]:
    """Slice pass. Returns (non_bomb_count, bomb_hit, slice_events).

    Per-hand swipe validity (segment recency and length) is checked once and
    cached so we do not redo the work for every fruit on a busy frame. The
    returned events let the caller spawn particle effects without re-walking
    the geometry.
    """
    active_segments: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for state in hands_state.by_handedness.values():
        if len(state.trail) < 2:
            continue
        t_prev = state.trail[-2][2]
        t_curr = state.trail[-1][2]
        if t_curr - t_prev > config.MAX_SLASH_STEP_DT_S:
            continue
        p1 = (state.trail[-2][0], state.trail[-2][1])
        p2 = (state.trail[-1][0], state.trail[-1][1])
        if segment_length(p1, p2) < config.MIN_SLASH_PIXELS_PER_TRAIL_STEP:
            continue
        active_segments.append((p1, p2))

    if not active_segments:
        return 0, False, []

    sliced_count = 0
    bomb_hit = False
    events: list[SliceEvent] = []
    for fruit in fruits:
        if not fruit.alive:
            continue
        for p1, p2 in active_segments:
            if segment_circle_intersects(p1, p2, (fruit.x, fruit.y), fruit.radius):
                fruit.alive = False
                fruit.sliced = True
                if fruit.is_bomb:
                    bomb_hit = True
                else:
                    sliced_count += 1
                events.append(SliceEvent(fruit=fruit, p1=p1, p2=p2))
                break
    return sliced_count, bomb_hit, events


COMBO_NEVER = -10**9


def _apply_combo(
    n_new: int, now_ms: int, last_combo_ms: int, combo_count: int
) -> tuple[int, int, int]:
    """Return (score_to_add, new_combo_count, new_last_combo_ms).

    Slices within COMBO_WINDOW_MS of the previous slice stack into the combo.
    Multipliers are applied per individual fruit: when n_new > 1 in a single
    frame, the first fruit may score 1x and the second 2x within the same
    swipe, rather than both jumping straight to 2x. This matches the canonical
    "Fruit Ninja" multiplier intuition.
    """
    if n_new <= 0:
        return 0, combo_count, last_combo_ms
    if now_ms - last_combo_ms > config.COMBO_WINDOW_MS:
        combo_count = 0
    score_add = 0
    for _ in range(n_new):
        combo_count += 1
        if combo_count >= 3:
            score_add += 3
        elif combo_count == 2:
            score_add += 2
        else:
            score_add += 1
    return score_add, combo_count, now_ms


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="reflex", description="Gesture-controlled Fruit Ninja clone.")
    parser.add_argument(
        "--smoke-test",
        type=int,
        default=0,
        metavar="N",
        help="Exit after N frames. Used by CI / scaffolding checks.",
    )
    parser.add_argument(
        "--no-camera",
        action="store_true",
        help="Skip the webcam. The window opens but stays empty. Useful for CI.",
    )
    parser.add_argument(
        "--no-tracker",
        action="store_true",
        help="Skip MediaPipe. Webcam still renders. Useful for isolating issues.",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Index of the webcam to open (default 0).",
    )
    return parser.parse_args(argv)


def _frame_to_surface(rgb: np.ndarray) -> pygame.Surface:
    h, w, _ = rgb.shape
    raw = pygame.image.frombuffer(rgb.data, (w, h), "RGB")
    return raw.convert()


def _color_for_label(label: str) -> tuple[int, int, int]:
    return config.HAND_COLOR_BY_LABEL.get(label, config.DEFAULT_HAND_COLOR)


def _draw_trails(overlay: pygame.Surface, hands_state: HandsState) -> None:
    """Draw a fading trail per hand onto a SRCALPHA overlay."""
    overlay.fill((0, 0, 0, 0))
    for label, state in hands_state.by_handedness.items():
        if len(state.trail) < 2:
            continue
        rgb = _color_for_label(label)
        pts = list(state.trail)
        n = len(pts)
        for i in range(n - 1):
            # progress 0 at the tail, 1 at the head; quadratic ease so the
            # head looks bold and the tail is wispy.
            progress = (i + 1) / (n - 1)
            eased = progress * progress
            width = max(
                config.TRAIL_TAIL_WIDTH,
                int(config.TRAIL_TAIL_WIDTH + (config.TRAIL_HEAD_WIDTH - config.TRAIL_TAIL_WIDTH) * eased),
            )
            alpha = int(255 * eased)
            x1, y1, _ = pts[i]
            x2, y2, _ = pts[i + 1]
            pygame.draw.line(
                overlay,
                (rgb[0], rgb[1], rgb[2], alpha),
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                width=width,
            )


def _draw_hand_overlay(screen: pygame.Surface, result: TrackerResult) -> None:
    for hand in result.hands:
        color = _color_for_label(hand.handedness)
        if LM_INDEX_TIP < len(hand.landmarks_px):
            x, y = hand.landmarks_px[LM_INDEX_TIP]
            pygame.draw.circle(screen, color, (x, y), 12, width=2)


def run(
    smoke_test_frames: int = 0,
    use_camera: bool = True,
    use_tracker: bool = True,
    camera_index: int = 0,
) -> int:
    camera: CameraThread | None = None
    tracker: TrackerThread | None = None
    frame_queue: queue.Queue[Frame] = queue.Queue(maxsize=1)
    tracker_in: queue.Queue[Frame] = queue.Queue(maxsize=1)
    tracker_out: queue.Queue[TrackerResult] = queue.Queue(maxsize=1)
    frame = 0
    clock: pygame.time.Clock | None = None
    try:
        pygame.init()
        screen = pygame.display.set_mode((config.WINDOW_W, config.WINDOW_H))
        pygame.display.set_caption("Reflex")
        font = pygame.font.SysFont("consolas", 22)
        small = pygame.font.SysFont("consolas", 18)
        hud_font = pygame.font.SysFont("consolas", config.HUD_FONT_SIZE, bold=True)
        clock = pygame.time.Clock()

        if use_camera:
            camera = CameraThread(frame_queue, device_index=camera_index)
            camera.start()
            if not camera.wait_until_opened(timeout=5.0):
                print("camera did not signal open within 5 s", file=sys.stderr)
            if camera.open_error is not None:
                print(f"camera open error: {camera.open_error}", file=sys.stderr)

        tracker_enabled = use_camera and use_tracker
        if tracker_enabled:
            try:
                model_path = assets.ensure_model()
            except RuntimeError as exc:
                print(f"hand-tracking disabled: {exc}", file=sys.stderr)
                tracker_enabled = False
            else:
                tracker = TrackerThread(tracker_in, tracker_out, model_path)
                tracker.start()

        running = True
        last_surface: pygame.Surface | None = None
        last_result: TrackerResult | None = None
        last_result_ts_ms = -1
        last_fps_label: pygame.Surface | None = None
        last_fps_value: float | None = None
        dim_overlay = pygame.Surface((config.WINDOW_W, config.WINDOW_H))
        dim_overlay.fill((0, 0, 0))
        dim_overlay.set_alpha(config.WEBCAM_DIM_ALPHA)
        trail_overlay = pygame.Surface((config.WINDOW_W, config.WINDOW_H), pygame.SRCALPHA)
        hands_state = HandsState(
            trail_maxlen=config.TRAIL_MAXLEN,
            timeout_ms=config.HAND_TIMEOUT_MS,
        )

        # F3 toggles a small debug overlay with per-stage stats.
        show_debug = False
        rate_window_start_ms = pygame.time.get_ticks()
        rate_window_camera_start = 0
        rate_window_tracker_start = 0
        rate_window_tracker_recv_start = 0
        camera_fps = 0.0
        tracker_fps = 0.0
        drops_per_s = 0.0

        fruits: list[Fruit] = []
        particles: list[Particle] = []
        halves: list[FruitHalf] = []
        particle_overlay = pygame.Surface((config.WINDOW_W, config.WINDOW_H), pygame.SRCALPHA)
        # play_start_ms anchors the spawn ramp. We reset it on every PLAYING
        # entry so a slow camera open or a stay-on-menu pause never eats the
        # early easy frames.
        play_start_ms: int | None = None
        next_spawn_at_ms = pygame.time.get_ticks() + config.FIRST_SPAWN_DELAY_MS
        score = 0
        lives = config.STARTING_LIVES
        combo_count = 0
        last_combo_ms = COMBO_NEVER
        scene = Scene.MENU
        best_score = persistence.load_best_score()
        is_new_best = False
        title_font = pygame.font.SysFont("consolas", config.TITLE_FONT_SIZE, bold=True)
        if smoke_test_frames:
            # Skip the menu wait so the smoke harness exercises the playing path.
            scene = Scene.PLAYING

        def begin_playing() -> None:
            nonlocal fruits, particles, halves, play_start_ms, next_spawn_at_ms
            nonlocal score, lives, combo_count, last_combo_ms, last_result, last_result_ts_ms
            nonlocal hands_state, scene, is_new_best
            fruits = []
            particles = []
            halves = []
            play_start_ms = pygame.time.get_ticks()
            next_spawn_at_ms = play_start_ms + config.FIRST_SPAWN_DELAY_MS
            score = 0
            lives = config.STARTING_LIVES
            combo_count = 0
            last_combo_ms = COMBO_NEVER
            last_result = None
            last_result_ts_ms = -1
            hands_state = HandsState(
                trail_maxlen=config.TRAIL_MAXLEN,
                timeout_ms=config.HAND_TIMEOUT_MS,
            )
            is_new_best = False
            scene = Scene.PLAYING

        def end_game() -> None:
            nonlocal scene, best_score, is_new_best
            scene = Scene.GAME_OVER
            if score > best_score:
                best_score = score
                is_new_best = True
                persistence.save_best_score(best_score)

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif (
                    event.type == pygame.KEYDOWN
                    and event.key == pygame.K_SPACE
                    and scene in (Scene.MENU, Scene.GAME_OVER)
                ):
                    begin_playing()
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_F3:
                    show_debug = not show_debug

            try:
                latest = frame_queue.get_nowait()
                last_surface = _frame_to_surface(latest.rgb)
                if play_start_ms is None:
                    play_start_ms = pygame.time.get_ticks()
                    next_spawn_at_ms = play_start_ms + config.FIRST_SPAWN_DELAY_MS
                if tracker_enabled:
                    publish_latest(tracker_in, latest)
            except queue.Empty:
                pass

            try:
                last_result = tracker_out.get_nowait()
            except queue.Empty:
                pass

            # Feed the smoother only when a fresh result arrives. Render uses
            # whatever the smoother last produced, even between tracker frames.
            if last_result is not None and last_result.timestamp_ms != last_result_ts_ms:
                last_result_ts_ms = last_result.timestamp_ms
                for hand in last_result.hands:
                    if LM_INDEX_TIP < len(hand.landmarks_px):
                        hands_state.update(
                            hand.handedness,
                            hand.landmarks_px[LM_INDEX_TIP],
                            last_result.timestamp_ms,
                        )

            # Tracker timestamps use monotonic system uptime, while pygame
            # ticks start at init. Prune in the tracker clock even when no
            # fresh result arrives; keep game timers on pygame ticks.
            hands_state.prune(int(time.monotonic() * 1000))
            now_ms = pygame.time.get_ticks()

            # Cap dt so a window-hitch or tab-switch cannot teleport a fruit
            # through the trail. clock.get_time() returns the prior frame's ms.
            dt = min(clock.get_time() / 1000.0, config.MAX_PHYSICS_DT_S)

            if (
                scene == Scene.PLAYING
                and play_start_ms is not None
                and now_ms >= next_spawn_at_ms
            ):
                elapsed_s = (now_ms - play_start_ms) / 1000.0
                fruits.append(spawn_one(elapsed_s))
                next_spawn_at_ms = now_ms + int(_spawn_interval_ms(elapsed_s))

            for fruit in fruits:
                fruit.update(dt)

            if scene == Scene.PLAYING:
                n_sliced, bomb_hit, slice_events = _try_slash(hands_state, fruits)
                for ev in slice_events:
                    burst_count = 26 if ev.fruit.is_bomb else 14
                    burst_color = (
                        config.BOMB_FUSE_COLOR if ev.fruit.is_bomb else ev.fruit.color
                    )
                    emit_burst(
                        particles, ev.fruit.x, ev.fruit.y, burst_color, count=burst_count
                    )
                    dx = ev.p2[0] - ev.p1[0]
                    dy = ev.p2[1] - ev.p1[1]
                    emit_halves(halves, ev.fruit, dx, dy)
                if bomb_hit:
                    end_game()
                else:
                    add, combo_count, last_combo_ms = _apply_combo(
                        n_sliced, now_ms, last_combo_ms, combo_count
                    )
                    score += add

            # Drop fruits that have left play. A non-bomb fruit falling past
            # the bottom unsliced costs a life only while we are still playing.
            remaining: list[Fruit] = []
            for fruit in fruits:
                if fruit.out_of_play or not fruit.alive:
                    if (
                        scene == Scene.PLAYING
                        and fruit.fell_off_bottom
                        and not fruit.sliced
                        and not fruit.is_bomb
                    ):
                        lives = max(0, lives - 1)
                        if lives == 0:
                            end_game()
                    continue
                remaining.append(fruit)
            fruits = remaining

            # Idle-frame decay: when no slice arrived this frame, drop the
            # HUD-visible combo back to 0 after the window expires. The
            # _apply_combo function also resets count on a fresh slice past
            # the window, so this branch only handles the no-new-slice case.
            if combo_count and now_ms - last_combo_ms > config.COMBO_WINDOW_MS:
                combo_count = 0

            # Advance and cull effects.
            for p in particles:
                p.update(dt)
            particles = [p for p in particles if p.alive]
            for h in halves:
                h.update(dt)
            halves = [h for h in halves if h.alive]

            if last_surface is not None:
                screen.blit(last_surface, (0, 0))
                screen.blit(dim_overlay, (0, 0))
            else:
                screen.fill(config.BG_COLOR)
                if use_camera and camera is not None and camera.open_error is None:
                    msg = small.render("waiting for first webcam frame...", True, config.TEXT_COLOR)
                    screen.blit(msg, (16, 48))
                elif use_camera and camera is not None and camera.open_error is not None:
                    msg = small.render(f"camera error: {camera.open_error}", True, (255, 120, 120))
                    screen.blit(msg, (16, 48))

            for h in halves:
                h.draw(screen)

            for fruit in fruits:
                fruit.draw(screen)

            if particles:
                particle_overlay.fill((0, 0, 0, 0))
                for p in particles:
                    p.draw(particle_overlay)
                screen.blit(particle_overlay, (0, 0))

            _draw_trails(trail_overlay, hands_state)
            screen.blit(trail_overlay, (0, 0))

            if last_result is not None:
                _draw_hand_overlay(screen, last_result)

            if scene != Scene.MENU:
                hud.draw_score(screen, hud_font, score)
                hud.draw_lives(screen, lives, config.STARTING_LIVES)
            if combo_count >= 2:
                mult = 3 if combo_count >= 3 else 2
                hud.draw_combo(screen, hud_font, combo_count, mult)
            if scene == Scene.GAME_OVER:
                hud.draw_game_over(screen, title_font, hud_font, score, best_score, is_new_best)
            elif scene == Scene.MENU:
                show_hint = last_surface is None and use_camera
                hud.draw_menu(screen, title_font, hud_font, best_score, show_hint)

            # Refresh the per-second rate counters once a second so the debug
            # overlay shows a stable rolling number, not a jittery instant.
            if now_ms - rate_window_start_ms >= 1000:
                window_s = (now_ms - rate_window_start_ms) / 1000.0
                cam_now = camera.frames_published if camera is not None else 0
                trk_now = tracker.callbacks_received if tracker is not None else 0
                trk_recv = tracker.frames_received if tracker is not None else 0
                camera_fps = (cam_now - rate_window_camera_start) / window_s
                tracker_fps = (trk_now - rate_window_tracker_start) / window_s
                # "drops" = camera frames we did NOT manage to feed into the
                # tracker because tracker_in was full. Approximated as the
                # gap between what camera produced and what tracker received.
                drops = max(0, (cam_now - rate_window_camera_start) - (trk_recv - rate_window_tracker_recv_start))
                drops_per_s = drops / window_s
                rate_window_start_ms = now_ms
                rate_window_camera_start = cam_now
                rate_window_tracker_start = trk_now
                rate_window_tracker_recv_start = trk_recv

            if show_debug:
                render_ms = clock.get_time()
                trk_latency = last_result.latency_ms if last_result is not None else 0.0
                hud.draw_debug(
                    screen,
                    small,
                    render_fps=clock.get_fps(),
                    render_ms=float(render_ms),
                    camera_fps=camera_fps,
                    tracker_fps=tracker_fps,
                    tracker_latency_ms=trk_latency,
                    drops_per_s=drops_per_s,
                )

            fps = clock.get_fps()
            rounded = round(fps, 1)
            if rounded != last_fps_value or last_fps_label is None:
                hands_n = len(last_result.hands) if last_result is not None else 0
                last_fps_label = small.render(
                    f"{rounded:5.1f} fps   hands={hands_n}   fruits={len(fruits)}   Q/ESC",
                    True,
                    config.TEXT_COLOR,
                )
                last_fps_value = rounded
            screen.blit(last_fps_label, (16, config.WINDOW_H - 28))
            pygame.display.flip()

            clock.tick(config.TARGET_FPS)
            frame += 1
            if smoke_test_frames and frame >= smoke_test_frames:
                running = False
    finally:
        if tracker is not None:
            tracker.stop()
            tracker.join(timeout=2.0)
        if camera is not None:
            camera.stop()
            camera.join(timeout=2.0)
        if smoke_test_frames:
            published = camera.frames_published if camera is not None else 0
            cam_failures = camera.read_failures if camera is not None else 0
            t_recv = tracker.frames_received if tracker is not None else 0
            t_sent = tracker.frames_sent_to_detector if tracker is not None else 0
            t_cb = tracker.callbacks_received if tracker is not None else 0
            fps_now = clock.get_fps() if clock is not None else 0.0
            print(
                f"smoke summary  rendered={frame}  "
                f"camera_published={published}  camera_read_failures={cam_failures}  "
                f"tracker_recv={t_recv}  tracker_sent={t_sent}  tracker_callbacks={t_cb}  "
                f"avg_fps={fps_now:.1f}",
                flush=True,
            )
        pygame.quit()
    return 0


def main() -> int:
    args = parse_args()
    try:
        return run(
            smoke_test_frames=args.smoke_test,
            use_camera=not args.no_camera,
            use_tracker=not args.no_tracker,
            camera_index=args.camera_index,
        )
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
