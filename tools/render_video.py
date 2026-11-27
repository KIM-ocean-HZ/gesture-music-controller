#!/usr/bin/env python3
"""Render a demo video from a live session: picture from --record-screen,
sound mixed from the original WAV loops at the times the frame log says
each loop was played and stopped.

  python tools/render_video.py --screen recordings/take.screen.mp4 \
      --log recordings/take.jsonl --start 95 --end 171.2 --out demo.mp4

--start/--end are seconds from the first logged frame. The soundtrack is
rendered, not recorded from the speakers: it leaves out audio-device
latency and anything else that was audible in the room. Loops stop with a
5 ms fade to avoid a digital click; otherwise samples are summed and
clipped like pygame's mixer. Needs ffmpeg on PATH.
"""

import argparse
import csv
import json
import statistics
import subprocess
import sys
import wave
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
RATE, FPS = 44100, 30


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path)) as w:
        if w.getframerate() != RATE or w.getsampwidth() != 2:
            raise ValueError(f"{path.name}: need 16-bit {RATE} Hz")
        data = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768
        data = data.reshape(-1, w.getnchannels())
    return np.repeat(data, 2, axis=1) if data.shape[1] == 1 else data[:, :2]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--screen", type=Path, required=True)
    ap.add_argument("--log", type=Path, required=True)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    import main as app  # sound-set file names
    lines = [json.loads(s) for s in args.log.read_text().splitlines() if s.strip()]
    header, frames = lines[0].get("header", {}), [r for r in lines if "header" not in r]
    cfg = app.Config(sound_set=header.get("sound_set", "HousePiano"))
    loops = [read_wav(app.ASSETS_DIR / f) for f in cfg.sound_sets[cfg.sound_set]]
    t0 = frames[0]["t"]
    end = args.end if args.end is not None else frames[-1]["t"] - t0

    # Each frame is shown, and its starts/stops happen, after processing:
    # place both at capture time + (play() time - read time).
    delays = [p - r["perf_read"] for r in frames for p in r.get("perf_play", [])]
    delay = statistics.median(delays) if delays else 0.0

    # Loop instances [start, stop) in seconds from t0, and the volume over time.
    open_at, spans = {}, []
    for r in frames:
        t = r["t"] - t0
        for loop in r.get("stopped", []):
            if loop in open_at:
                spans.append((loop, open_at.pop(loop), t + delay))
        for loop, p in zip(r.get("played", []), r.get("perf_play", [])):
            open_at[loop] = t + (p - r["perf_read"])
    spans += [(loop, s, end + 1.0) for loop, s in open_at.items()]
    vol_t = np.array([r["t"] - t0 + delay for r in frames])
    vol_v = np.array([r.get("volume", cfg.default_volume) for r in frames])

    n = int(round((end - args.start) * RATE))
    mix = np.zeros((n, 2), np.float32)
    fade = int(0.005 * RATE)
    for loop, s, e in spans:
        a, b = max(s, args.start), min(e, end)
        if b <= a:
            continue
        src = loops[loop - 1]
        i0, i1 = int(round((a - args.start) * RATE)), int(round((b - args.start) * RATE))
        offset = int(round((a - s) * RATE))
        idx = (offset + np.arange(i1 - i0)) % len(src)
        seg = src[idx].copy()
        if e <= end and len(seg) > fade:
            seg[-fade:] *= np.linspace(1, 0, fade, dtype=np.float32)[:, None]
        mix[i0:i1] += seg
    times = args.start + np.arange(n) / RATE
    mix *= vol_v[np.clip(np.searchsorted(vol_t, times, side="right") - 1, 0, len(vol_v) - 1)][:, None]
    clipped = float(np.mean(np.abs(mix) > 1.0))
    pcm = (np.clip(mix, -1, 1) * 32767).astype("<i2")
    wav_path = args.out.with_suffix(".wav")
    with wave.open(str(wav_path), "wb") as w:
        w.setnchannels(2), w.setsampwidth(2), w.setframerate(RATE)
        w.writeframes(pcm.tobytes())

    # Picture: constant 30 fps from the variable-rate screen recording.
    stamps = [float(row["t"]) - t0 + delay
              for row in csv.DictReader(open(app.timestamps_path(args.screen)))]
    cap = cv2.VideoCapture(str(args.screen))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ff = subprocess.Popen(
        ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{w}x{h}", "-r", str(FPS), "-i", "-", "-i", str(wav_path),
         "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-b:a", "192k", "-shortest", str(args.out)],
        stdin=subprocess.PIPE)
    k, current, ok = -1, None, True
    for j in range(int((end - args.start) * FPS)):
        tau = args.start + j / FPS
        while ok and k + 1 < len(stamps) and stamps[k + 1] <= tau:
            ok, img = cap.read()
            if ok:
                k, current = k + 1, img
        if current is not None:
            ff.stdin.write(current.tobytes())
    ff.stdin.close()
    if ff.wait():
        sys.exit("ffmpeg failed")
    wav_path.unlink()
    print(f"{args.out}: {end - args.start:.1f} s, {len(spans)} loop spans, "
          f"processing offset {delay * 1000:.0f} ms, clipped samples {clipped:.4%}")


if __name__ == "__main__":
    main()
