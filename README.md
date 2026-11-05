# Gesture Music Controller

Real-time, hands-free music loop station driven entirely by your webcam.  
Hand gestures start and stop audio loops; in the Artistic view, head nods flash the screen. No hardware beyond a laptop required.

Built with **MediaPipe**, **OpenCV**, and **pygame** as a computer-vision portfolio project, inspired by loop stations and DJ performance.

---

## Demo

> Run `python demo_simple.py` for an audio-free hand-tracking preview.  
> Run `python main.py --demo` to see the full UI without needing any audio files.

---

## Features

| Feature | Detail |
|---|---|
| **Real-time hand detection** | MediaPipe Hands — 21 landmarks, one hand tracked |
| **Handedness-aware finger counting** | The thumb test flips direction for left and right hands |
| **EMA-smoothed swipe detection** | Exponential moving average over wrist velocity damps jitter |
| **Loops start on master-loop boundaries** | The first loop starts immediately and sets the master loop; later loops start, and stopped loops end, at its next boundary |
| **Pinch volume control** | Thumb–index spread adjusts master volume in real time |
| **Head-nod detection** | In Artistic mode, a nose-tip Y-velocity reversal flashes the screen |
| **Three sound sets** | HousePiano / BluesGuitar / FunkGuitar — switching stops the current loops |
| **Two visual modes** | Reality (camera passthrough) and Artistic (landmark skeleton on black) |
| **Loop-progress rings** | Dashboard circles fill and pulse once per master loop |
| **Animated HUD panels** | Slide-in/out action notifications with cubic easing |
| **Screenshot on demand** | Press `s` — saved to project root |
| **Measurement tools** | Record clips, replay them headless, and compute tracking loss, false triggers, hit rate and software latency ([docs/MEASUREMENT.md](docs/MEASUREMENT.md)) |
| **CLI arguments** | Camera index, resolution, swipe threshold, demo mode, record / replay / log |

---

## Technical Highlights

- **Temporal smoothing** — an EMA (α = 0.4) over wrist x-velocity; a swipe fires above 45 px/frame. Smoothing damps tremor at the cost of some lag; neither has been measured yet.
- **Loop quantisation** — `math.ceil((t - clock_start) / loop_length)` gives the next master-loop boundary. Queued loops start, and stopping loops end, on that boundary, so loops stay phase-aligned. It aligns to the master loop, not to individual beats.
- **One frame loop** — hand landmarks, face landmarks and the master-loop clock are handled together in one pass per camera frame.
- **Graceful degradation** — the `--demo` flag disables audio entirely; missing asset files produce warnings rather than crashes.
- **Handedness-aware thumb check** — MediaPipe labels hands assuming a mirrored image, so the frame is flipped first and `count_fingers()` uses opposite x-directions for the left and right thumb.
- **Priority-aware animation system** — lower-priority HUD notifications are suppressed when a more important action fires.
- **Replayable measurement** — recorded clips keep each frame's capture time, so replay reproduces the timing that cooldowns and loop boundaries depend on.

---

## Tech Stack

- **Python 3.9+**
- **MediaPipe 0.10.21** (pinned) — Hands, FaceMesh
- **OpenCV 4.8+** — capture, drawing, display
- **NumPy** — frame compositing, overlay blending
- **pygame** — audio playback (44.1 kHz, 512-sample buffer)

---

## Installation

```bash
git clone https://github.com/KIM-ocean-HZ/gesture-music-controller.git
cd gesture-music-controller
pip install -r requirements.txt
```

### Audio files

The `.wav` loops are not included in the repository.  
Place **five audio loops per set** in the `assets/` folder using these names:

```
assets/
├── loop1.wav  … loop5.wav      # HousePiano set
├── guitar1.wav … guitar5.wav   # BluesGuitar set
├── funk1.wav  … funk5.wav      # FunkGuitar set
├── startup.wav                  # played on launch
└── confirm.wav                  # played on sound-set switch
```

Any 44 100 Hz WAV works; the first loop you start sets the master-loop length.  
The app logs a warning for missing files and keeps running.

---

## Usage

```bash
# Full controller (requires audio files in assets/)
python main.py

# Specify a different camera or resolution
python main.py --camera 1 --width 1920 --height 1080

# Audio-free demo — no .wav files needed
python main.py --demo

# Lower swipe sensitivity (default: 45 px/frame)
python main.py --swipe-threshold 30

# Minimal hand-tracking only (no audio, simpler UI)
python demo_simple.py
```

### Gesture controls

| Gesture | Action |
|---|---|
| Show **N** fingers (1–5) | Target loop N |
| **Swipe right** | Start targeted loop — immediately if nothing is playing, otherwise at the next master-loop boundary |
| **Swipe left** | Stop targeted loop at the next master-loop boundary |
| **Pinch** thumb + index | Control master volume (spread = louder) |

### Keyboard

| Key | Action |
|---|---|
| `1` / `2` | Artistic / Reality visual mode |
| `p` / `g` / `f` | HousePiano / BluesGuitar / FunkGuitar |
| `s` | Save screenshot |
| `h` | Toggle help overlay |
| `q` | Quit |

---

## Measurement

Changes to recognition are measured on the same recorded clips before and after, under normal light, low light, fast motion and partial occlusion. The protocol and metric definitions are in [docs/MEASUREMENT.md](docs/MEASUREMENT.md).

```bash
python main.py --demo --record recordings/s1-normal-idle.mp4   # record a clip
python bench/bench.py replay --label baseline                   # replay all clips
python bench/bench.py compare results/baseline results/<label>  # before → after
python -m unittest discover tests                               # tests for the metrics
```

No results yet; they will be added here with the commit and clip set they came from.

---

## Project Structure

```
gesture-music-controller/
├── main.py            # Full gesture controller (AudioEngine, GestureEngine, Renderer)
├── demo_simple.py     # Audio-free hand-tracking demo
├── bench/
│   ├── bench.py       # Replay clips, compute metrics, compare runs
│   └── clips.example.csv
├── docs/
│   └── MEASUREMENT.md # Recording protocol and metric definitions
├── tests/
│   └── test_bench.py
├── requirements.txt
├── .gitignore
└── assets/            # Place your .wav files here (excluded from git)
    └── .gitkeep
```

---

## Roadmap

- [ ] Baseline measurement of the current version on recorded clips
- [ ] Low-light handling, measured before and after
- [ ] Motion-blur handling, measured before and after
- [ ] Partial-occlusion handling, measured before and after
- [ ] End-to-end gesture-to-sound latency (external recording)
- [ ] Custom gesture mapping via a JSON config file
- [ ] Two-hand support — one hand selects loop, other hand controls FX

---

## Changelog

- **2026-11-05** — Restarted the project. Corrected README claims that had no code or measurement behind them (a CPU latency figure, "nearest beat" quantisation, the reason audio files are not included). Added clip recording, headless replay, per-frame logs, a measurement script and tests. A swipe now restarts the cooldown even when it does not change a loop. Pinned MediaPipe to 0.10.21.
- **2026-04-10** — First public version.

---

## Author

**Hanze JIN** — BSc Computer Science, University of Nottingham  
[GitHub: KIM-ocean-HZ](https://github.com/KIM-ocean-HZ)
