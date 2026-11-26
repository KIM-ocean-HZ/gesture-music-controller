#!/usr/bin/env python3
"""Replay unlabelled failure clips without inventing a scripted ground truth.

Example (run from the repository root):
  python bench/diagnose.py --source-root recordings/repair-r1-evidence/before-source \
      --out results/r1-before --clip recordings/repair-r1-evidence/clips/*.mp4

Frame logs include landmarks for diagnosis and stay under git-ignored logs/.
Counts are observations, not accuracy, false-trigger or latency measurements.
"""

import argparse
import hashlib
import importlib.util
import json
import os
from collections import Counter
from dataclasses import asdict
from pathlib import Path


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path,
                        default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--clip", type=Path, nargs="+", required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"refusing to overwrite evidence: {args.out}")
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    source = args.source_root.resolve() / "main.py"
    spec = importlib.util.spec_from_file_location("diagnostic_app", source)
    app = importlib.util.module_from_spec(spec)
    # dataclasses and annotations can need the module during execution.
    import sys
    sys.modules[spec.name] = app
    spec.loader.exec_module(app)

    class DiagnosticController(app.GestureMusicController):
        def _handle_gesture(self, result, fw, now, *args, **kwargs):
            info = super()._handle_gesture(result, fw, now, *args, **kwargs)
            if result.multi_hand_landmarks:
                info["landmarks"] = [[p.x, p.y, p.z] for p in
                                     result.multi_hand_landmarks[0].landmark]
                if result.multi_handedness:
                    info["handedness_score"] = result.multi_handedness[0].classification[0].score
            return info

    (args.out / "logs").mkdir(parents=True)
    summary = {
        "purpose": "unlabelled failure diagnosis; no accuracy or false-trigger claim",
        "source_sha256": sha256(source),
        "source_root": str(args.source_root),
        "harness_sha256": sha256(Path(__file__)),
        "config": asdict(app.Config()),
        "clips": [],
    }
    for path in args.clip:
        path = path.resolve()
        sidecar = app.timestamps_path(path)
        video_source = app.VideoFileSource(path)
        width = int(video_source.cap.get(app.cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video_source.cap.get(app.cv2.CAP_PROP_FRAME_HEIGHT))
        cfg = app.Config(frame_width=width, frame_height=height)
        log_path = args.out / "logs" / (path.stem + ".jsonl")
        log = app.FrameLog(log_path, app.run_header(cfg, "replay", path))
        DiagnosticController(cfg, demo_mode=True, source=video_source,
                             frame_log=log, headless=True).run()
        records = [json.loads(s) for s in log_path.read_text().splitlines()]
        frames = [r for r in records if "header" not in r]
        if not frames:
            raise ValueError(f"empty replay: {path}")
        start = frames[0]["t"]
        events = [{"frame": r["i"], "time_s": round(r["t"] - start, 3),
                   "direction": r["swipe"], "fingers": r["fingers"],
                   "action": r.get("action")}
                  for r in frames if "swipe" in r]
        item = {
            "clip": path.name, "sha256": sha256(path),
            "timestamps_sha256": sha256(sidecar),
            "resolution": [width, height], "frames": len(frames),
            "capture_span_s": frames[-1]["t"] - start,
            "frames_without_detection": sum(not r["hand"] for r in frames),
            "detected_finger_histogram": dict(Counter(r["fingers"] for r in frames if r["hand"])),
            "swipes": events,
        }
        volumes = [r["volume"] for r in frames if "volume" in r]
        if volumes:
            item["volume"] = {"min": min(volumes), "max": max(volumes), "last": volumes[-1]}
        summary["clips"].append(item)
        print(f"{path.name}: {len(frames)} frames, {len(events)} swipe events", flush=True)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
