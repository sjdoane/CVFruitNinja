"""Tunable constants. Keep all magic numbers here so playtest tweaks are one diff."""
from __future__ import annotations

from pathlib import Path

WINDOW_W = 1280
WINDOW_H = 720
TARGET_FPS = 60

BG_COLOR = (10, 12, 18)
TEXT_COLOR = (235, 235, 245)

# Webcam background dimming, 0 (transparent) to 255 (opaque black).
WEBCAM_DIM_ALPHA = 100

# Per-hand colors keyed by MediaPipe handedness label so they stay stable
# regardless of which hand the player raises first. Mirrored-camera labels
# match the player's own perspective.
HAND_COLOR_BY_LABEL: dict[str, tuple[int, int, int]] = {
    "Right": (80, 220, 255),
    "Left": (255, 110, 200),
}
DEFAULT_HAND_COLOR = (200, 200, 210)

# Trail: maxlen is the number of smoothed samples drawn. At ~30 Hz tracker
# updates, 12 samples is roughly 400 ms of history.
TRAIL_MAXLEN = 12
TRAIL_HEAD_WIDTH = 18
TRAIL_TAIL_WIDTH = 2
# Stop tracking a hand we have not seen in this many ms (clears the trail).
HAND_TIMEOUT_MS = 500

# Slash detection: minimum fingertip displacement between consecutive tracker
# samples (~30 Hz) that counts as a real swipe. Keeps a stationary finger from
# slicing fruits that drift into it.
MIN_SLASH_PIXELS_PER_TRAIL_STEP = 25.0
# A slash segment older than this is a tracking gap, not a swipe. Caps the
# segment length we consider so a re-acquisition cannot sweep half the screen.
MAX_SLASH_STEP_DT_S = 0.08
# Physics clamp: never advance simulation by more than this per render frame,
# even after a long event-loop hitch. Avoids fruits teleporting through trails.
MAX_PHYSICS_DT_S = 0.05

# Fruit physics.
GRAVITY_PX_S2 = 1500.0
FRUIT_RADII = [34, 40, 50]
FRUIT_COLORS = [(220, 60, 80), (240, 150, 40), (110, 180, 60)]
FRUIT_SPAWN_X_PAD = 180
FRUIT_SPAWN_VX_RANGE = (-220.0, 220.0)
FRUIT_SPAWN_VY_RANGE = (-1150.0, -950.0)
FRUIT_SPAWN_X_BIAS = 0.6  # 0 = no bias, 1 = strong inward bias for edge spawns
FRUIT_SIDE_OFFSCREEN_PAD = 200
FIRST_SPAWN_DELAY_MS = 500
# Spawn interval ramps linearly from start to peak over RAMP_SECONDS, then flat.
SPAWN_INTERVAL_START_MS = 2000
SPAWN_INTERVAL_PEAK_MS = 600
SPAWN_RAMP_SECONDS = 60.0

# Bombs ramp from 0 to BOMB_PROB_MAX over BOMB_RAMP_SECONDS.
BOMB_PROB_MAX = 0.06
BOMB_RAMP_SECONDS = 30.0
BOMB_COLOR = (35, 35, 40)
BOMB_FUSE_COLOR = (255, 180, 60)
BOMB_RADII = [38, 44]

# Combo window: any extra slices within this many ms of the last slice stack
# into the combo multiplier. 2x for 2 in window, 3x for 3 or more.
COMBO_WINDOW_MS = 250

STARTING_LIVES = 3
HUD_FONT_SIZE = 36
TITLE_FONT_SIZE = 96

# Project root (the folder containing pyproject.toml) and the model file path.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "hand_landmarker.task"
