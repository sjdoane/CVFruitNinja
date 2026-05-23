# CV Fruit Ninja

A webcam-controlled Fruit Ninja clone. Slice with your index finger.
Python, MediaPipe, pygame.

![demo](docs/demo.gif)

## Run

You need [uv](https://docs.astral.sh/uv/) and a webcam.

```bash
uv run reflex
```

That's it. uv handles the venv and dependencies on first run, and the
MediaPipe hand model is already in the repo.

## Controls

- Index fingertip slices fruit. Both hands work; combos stack.
- `SPACE` start / restart
- `F3` debug overlay (render fps, camera fps, tracker fps, end-to-end latency, dropped frames)
- `Q` or `ESC` quit

Lose a life when a fruit hits the bottom unsliced. Slicing a bomb ends the round.

## How it works

Three threads connected by `queue.Queue(maxsize=1)` with drop-on-full
semantics, so the render loop never blocks on the camera or the model:

```
CameraThread  -- Frame -->  main loop  -- Frame -->  TrackerThread
                                  |                        |
                                  v                        v
                              render 60 FPS        result_callback
                                  ^                        |
                                  +---- TrackerResult <----+
```

The camera mirrors at the source so all landmark math happens in the
coordinate space the player actually sees. MediaPipe's `HandLandmarker`
runs in `LIVE_STREAM` mode with monotonic millisecond timestamps and is
allowed to drop frames.

A few choices worth pointing at:

- **Segment-vs-circle slash test.** For each hand, the line from the
  previous smoothed fingertip to the current one is tested against every
  fruit circle. A stationary finger never scores (velocity gate), a fast
  swipe through multiple fruits slices them all.
- **One-euro filter** smooths the fingertip without the lag a fixed
  low-pass would add. Timestamps are passed in seconds; passing frame
  indices is the documented footgun and is avoided.
- **Velocity + recency gates** reject swipes shorter than 25 px between
  tracker samples or older than 80 ms apart, so a tracking glitch cannot
  fake a giant swipe.

## Layout

```
src/reflex/
  camera.py       cv2 capture, mirror, RGB convert, publish
  tracker.py      MediaPipe LIVE_STREAM, monotonic timestamps
  gesture.py      OneEuroFilter, per-hand state, segment-vs-circle
  fruit.py        physics, bomb, spawn, draw
  particles.py    slice particles and spinning halves
  hud.py          score, lives, combo, menu, game over, F3 overlay
  persistence.py  best score
  assets.py       fetch the hand-landmark model on first run
  main.py         entry point and game loop
  config.py       tunables
models/           hand_landmarker.task (pre-downloaded)
tests/            pytest suite
```

## Tests

```bash
uv run python -m pytest -q
```

49 tests covering the one-euro filter, slash math, fruit physics, combo
multipliers, bomb spawn ramps, and best-score persistence.

## License

MIT. The MediaPipe model is distributed under Apache 2.0 by Google.
