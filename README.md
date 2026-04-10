# Gesture Music Controller

Real-time, hands-free music loop station driven entirely by your webcam.  
Hand gestures trigger and stop audio loops; head nods fire visual effects — no hardware beyond a laptop required.

Built with **MediaPipe**, **OpenCV**, and **pygame** as a computer-vision portfolio project.

---

## Demo

> Run `python demo_simple.py` for an audio-free hand-tracking preview.  
> Run `python main.py --demo` to see the full UI without needing any audio files.

---

## Features

| Feature | Detail |
|---|---|
| **Real-time hand detection** | MediaPipe Hands — 21 keypoints, ~30 ms latency on CPU |
| **Handedness-aware finger counting** | Correct thumb detection for both left and right hands |
| **EMA-smoothed swipe detection** | Exponential moving average suppresses jitter without adding lag |
| **Beat-quantised loop playback** | Loops snap to a master clock — no off-beat triggering |
| **Pinch volume control** | Thumb–index spread adjusts master volume in real time |
| **Head-nod detection** | Nose-tip Y-velocity reversal triggers rhythmic visual flash |
| **Three sound sets** | HousePiano / BluesGuitar / FunkGuitar — hot-swap mid-performance |
| **Two visual modes** | Reality (camera passthrough) and Artistic (landmark skeleton on black) |
| **Beat-sync glow pulse** | Dashboard circles pulse in time with the master clock |
| **Animated HUD panels** | Slide-in/out action notifications with cubic easing |
| **Screenshot on demand** | Press `s` — saved to project root |
| **CLI arguments** | Camera index, resolution, swipe threshold, demo mode |

---

## Technical Highlights

This project addresses several real-world computer vision engineering challenges:

- **Temporal smoothing** — EMA over raw wrist displacement reduces high-frequency hand tremor without introducing noticeable lag
- **Beat quantisation** — `math.ceil((t - clock_start) / loop_length)` snaps loop starts to the nearest beat boundary, keeping multiple loops phase-coherent
- **Multi-signal fusion** — hand pose, facial landmark stream, and a master timing signal are processed within a single event loop at full frame rate
- **Graceful degradation** — the `--demo` flag disables audio entirely; missing asset files produce warnings rather than crashes
- **Handedness-correct thumb detection** — MediaPipe's mirrored hand labels require opposite x-direction logic for the thumb; handled explicitly in `count_fingers()`
- **Priority-aware animation system** — lower-priority HUD notifications are suppressed when a more-important action fires, preventing visual clutter

---

## Tech Stack

- **Python 3.9+**
- **MediaPipe 0.10+** — Hands, FaceMesh
- **OpenCV 4.8+** — capture, drawing, display
- **NumPy** — frame compositing, overlay blending
- **pygame** — low-latency audio mixing

---

## Installation

```bash
git clone https://github.com/KIM-ocean-HZ/gesture-music-controller.git
cd gesture-music-controller
pip install -r requirements.txt
```

### Audio files

The `.wav` loops are not included in the repository (they exceed GitHub's size limits).  
Place **five audio loops per set** in the `assets/` folder using these names:

```
assets/
├── loop1.wav  … loop5.wav      # HousePiano set
├── guitar1.wav … guitar5.wav   # BluesGuitar set
├── funk1.wav  … funk5.wav      # FunkGuitar set
├── startup.wav                  # played on launch
└── confirm.wav                  # played on sound-set switch
```

Any 8-second stereo WAV at 44 100 Hz will work.  
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
| **Swipe right** | Queue targeted loop — starts on the next beat |
| **Swipe left** | Stop targeted loop — stops on the next beat |
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

## Project Structure

```
gesture-music-controller/
├── main.py            # Full gesture controller (AudioEngine, GestureEngine, Renderer)
├── demo_simple.py     # Audio-free hand-tracking demo
├── requirements.txt
├── .gitignore
└── assets/            # Place your .wav files here (excluded from git)
    └── .gitkeep
```

---

## Roadmap

- [ ] Export finger-gesture classifier to ONNX for edge deployment
- [ ] Quantise MediaPipe models for on-device mobile inference
- [ ] Add custom gesture mapping via a JSON config file
- [ ] Two-hand support — one hand selects loop, other hand controls FX

---

## Author

**Hanze JIN** — BSc Computer Science, University of Nottingham  
[GitHub: KIM-ocean-HZ](https://github.com/KIM-ocean-HZ)
