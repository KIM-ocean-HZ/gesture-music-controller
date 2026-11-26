#!/usr/bin/env python3
"""
Measurement harness for Gesture Music Controller.

  replay   Run every clip in a manifest through the current main.py (headless,
           no audio) and write per-frame logs plus summary.json.
  compare  Print two summary.json files side by side (before → after).
  latency  Software latency from a *live* frame log (main.py --log).

Definitions are in docs/MEASUREMENT.md.  Typical use:

  python bench/bench.py replay --label baseline
  python bench/bench.py replay --label lowlight-v1
  python bench/bench.py compare results/baseline results/lowlight-v1
  python bench/bench.py latency recordings/live-normal.jsonl
"""

import argparse
import csv
import json
import logging
import os
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

REPO = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Metric helpers (pure functions — covered by tests/test_bench.py)
# ---------------------------------------------------------------------------
def parse_intervals(text: str) -> List[Tuple[float, float]]:
    """'1.5-3;10-12.5' → [(1.5, 3.0), (10.0, 12.5)] (seconds from clip start)."""
    out = []
    for part in (text or "").split(";"):
        part = part.strip()
        if part:
            a, b = part.split("-")
            out.append((float(a), float(b)))
    return out


def swipe_token(rec: dict) -> str:
    return ("R" if rec["swipe"] == "right" else "L") + str(rec["fingers"])


def align(expected: Sequence[str], detected: Sequence[str]) -> Dict[str, int]:
    """
    Levenshtein alignment of the intended gesture sequence against the
    detected one.  Returns counts of hits (exact token), substitutions
    (a gesture fired but with the wrong direction or finger count), misses
    (intended, nothing fired) and extras (fired, nothing intended).
    """
    n, m = len(expected), len(detected)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if expected[i - 1] == detected[j - 1] else 1
            d[i][j] = min(d[i - 1][j - 1] + cost, d[i - 1][j] + 1, d[i][j - 1] + 1)

    counts = {"hits": 0, "substitutions": 0, "misses": 0, "extras": 0}
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            cost = 0 if expected[i - 1] == detected[j - 1] else 1
            if d[i][j] == d[i - 1][j - 1] + cost:
                counts["hits" if cost == 0 else "substitutions"] += 1
                i, j = i - 1, j - 1
                continue
        if i > 0 and d[i][j] == d[i - 1][j] + 1:
            counts["misses"] += 1
            i -= 1
        else:
            counts["extras"] += 1
            j -= 1
    return counts


def clip_metrics(frames: List[dict], kind: str, expected: Sequence[str] = (),
                 absent: Sequence[Tuple[float, float]] = ()) -> dict:
    """Metrics for one replayed clip.  `frames` excludes the header line."""
    if not frames:
        raise ValueError("empty frame log")
    t0 = frames[0]["t"]
    span = frames[-1]["t"] - t0
    fps = (len(frames) - 1) / span if span > 0 else 0.0
    duration = span + (1.0 / fps if fps else 0.0)

    def hand_expected(f):
        t = f["t"] - t0
        return not any(a <= t <= b for a, b in absent)

    lost = present = dropouts = 0
    prev_lost = False
    for f in frames:
        if not hand_expected(f):
            prev_lost = False
            continue
        present += 1
        is_lost = not f.get("hand", False)
        lost += is_lost
        if is_lost and not prev_lost:
            dropouts += 1
        prev_lost = is_lost

    swipes = [swipe_token(f) for f in frames if "swipe" in f]
    out = {
        "kind": kind,
        "frames": len(frames),
        "duration_s": round(duration, 3),
        "fps": round(fps, 2),
        "hand_expected_frames": present,
        "lost_frames": lost,
        "dropouts": dropouts,
        "swipes": len(swipes),
        "detected": " ".join(swipes),
    }
    if kind == "scripted":
        out["expected"] = len(expected)
        out.update(align(list(expected), swipes))
    return out


def aggregate(clips: List[dict]) -> dict:
    """Pool clip metrics: frames for tracking loss, minutes for rates."""
    present = sum(c["hand_expected_frames"] for c in clips)
    lost = sum(c["lost_frames"] for c in clips)
    minutes = sum(c["duration_s"] for c in clips) / 60
    idle = [c for c in clips if c["kind"] == "idle"]
    scripted = [c for c in clips if c["kind"] == "scripted"]
    agg = {
        "clips": len(clips),
        "minutes": round(minutes, 2),
        "tracking_loss_pct": round(100 * lost / present, 2) if present else None,
        "dropouts_per_min": round(sum(c["dropouts"] for c in clips) / minutes, 2)
        if minutes else None,
    }
    idle_min = sum(c["duration_s"] for c in idle) / 60
    if idle:
        agg["idle_minutes"] = round(idle_min, 2)
        agg["false_triggers_per_min"] = round(
            sum(c["swipes"] for c in idle) / idle_min, 2) if idle_min else None
    if scripted:
        exp = sum(c["expected"] for c in scripted)
        s_min = sum(c["duration_s"] for c in scripted) / 60
        agg["expected_gestures"] = exp
        agg["hit_rate_pct"] = round(100 * sum(c["hits"] for c in scripted) / exp, 1) if exp else None
        agg["substitutions"] = sum(c["substitutions"] for c in scripted)
        agg["misses"] = sum(c["misses"] for c in scripted)
        agg["extras_per_min"] = round(
            sum(c["extras"] for c in scripted) / s_min, 2) if s_min else None
    return agg


def software_latency(frames: List[dict]) -> List[float]:
    """
    For every immediate loop start ("start" action) in a live log: time from
    the triggering frame's read to the play() call, in ms.  Queued loops are
    excluded — they wait for the master-loop boundary by design.
    """
    out = []
    for k, f in enumerate(frames):
        if f.get("action") != "start":
            continue
        loop = f["fingers"]
        # From the triggering frame itself: since 2026-11-26 the loop is
        # played in that frame; older logs play it on a later frame.
        for g in frames[k:]:
            if loop in g.get("played", []):
                p = g["perf_play"][g["played"].index(loop)]
                out.append((p - f["perf_read"]) * 1000)
                break
    return out


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def read_log(path: Path) -> Tuple[dict, List[dict]]:
    with open(path) as fh:
        lines = [json.loads(line) for line in fh if line.strip()]
    header = lines[0].get("header", {}) if lines else {}
    return header, [r for r in lines if "header" not in r]


def read_manifest(path: Path) -> List[dict]:
    with open(path, newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if r.get("clip", "").strip()
                and not r["clip"].startswith("#")]
    for r in rows:
        r["kind"] = r.get("kind", "").strip()
        if r["kind"] not in ("idle", "scripted"):
            raise ValueError(f"{r['clip']}: kind must be idle or scripted")
        r["expected"] = (r.get("expected") or "").split()
        r["absent"] = parse_intervals(r.get("absent", ""))
    return rows


def git_state() -> dict:
    def run(*args):
        return subprocess.run(["git", *args], cwd=REPO, capture_output=True,
                              text=True).stdout.strip()
    return {
        "commit": run("rev-parse", "--short", "HEAD"),
        "dirty": bool(run("status", "--porcelain", "--untracked-files=no")),
    }


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
def cmd_replay(args):
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    sys.path.insert(0, str(REPO))
    import main as app  # noqa: E402  (needs the env var above)
    logging.getLogger().setLevel(logging.WARNING)

    rows = read_manifest(args.manifest)
    out_dir = args.out or REPO / "results" / args.label
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    state = git_state()
    if state["dirty"]:
        print("warning: working tree has uncommitted changes; "
              "summary records the commit with dirty=true", file=sys.stderr)

    clips = []
    for r in rows:
        clip = (args.manifest.parent / r["clip"]).resolve()
        log_path = out_dir / "logs" / (Path(r["clip"]).stem + ".jsonl")
        cfg = app.Config()
        frame_log = app.FrameLog(log_path, app.run_header(cfg, "replay", clip))
        app.GestureMusicController(
            cfg, demo_mode=True, source=app.VideoFileSource(clip),
            frame_log=frame_log, headless=True,
        ).run()
        _, frames = read_log(log_path)
        m = clip_metrics(frames, r["kind"], r["expected"], r["absent"])
        m.update(clip=r["clip"], condition=r.get("condition", "").strip() or "-")
        clips.append(m)
        print(f"  {r['clip']}: {m['frames']} frames, "
              f"{m['lost_frames']} lost, {m['swipes']} swipes", file=sys.stderr)

    by_cond = defaultdict(list)
    for c in clips:
        by_cond[c["condition"]].append(c)
    summary = {
        "label": args.label,
        **state,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "manifest": str(args.manifest),
        "by_condition": {k: aggregate(v) for k, v in sorted(by_cond.items())},
        "overall": aggregate(clips),
        "clips": clips,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print_summary(summary)
    print(f"\nwrote {out_dir / 'summary.json'}")


METRICS = [
    ("tracking_loss_pct",      "tracking loss %"),
    ("dropouts_per_min",       "dropouts / min"),
    ("false_triggers_per_min", "idle false triggers / min"),
    ("hit_rate_pct",           "scripted hit rate %"),
    ("substitutions",          "wrong loop / direction"),
    ("misses",                 "missed gestures"),
    ("extras_per_min",         "scripted extras / min"),
    ("minutes",                "minutes of video"),
]


def print_summary(s: dict):
    print(f"\n{s['label']}  (commit {s['commit']}{' dirty' if s['dirty'] else ''})")
    for cond, agg in list(s["by_condition"].items()) + [("ALL", s["overall"])]:
        print(f"  [{cond}]")
        for key, name in METRICS:
            if agg.get(key) is not None:
                print(f"    {name:28s} {agg[key]}")


def cmd_compare(args):
    a = json.loads((args.before / "summary.json").read_text())
    b = json.loads((args.after / "summary.json").read_text())
    if a["manifest"] != b["manifest"]:
        print("warning: the two runs used different manifests", file=sys.stderr)
    print(f"{'':30s}{a['label']} ({a['commit']})  →  {b['label']} ({b['commit']})")
    conds = list(dict.fromkeys(list(a["by_condition"]) + list(b["by_condition"])))
    for cond in conds + ["ALL"]:
        ga = a["overall"] if cond == "ALL" else a["by_condition"].get(cond, {})
        gb = b["overall"] if cond == "ALL" else b["by_condition"].get(cond, {})
        print(f"[{cond}]")
        for key, name in METRICS:
            va, vb = ga.get(key), gb.get(key)
            if va is None and vb is None:
                continue
            print(f"  {name:28s}{str(va):>10s}  →  {str(vb):<10s}")


def cmd_latency(args):
    header, frames = read_log(args.log)
    if header.get("source") != "live":
        sys.exit("latency needs a live log (main.py --log without --input).")
    ms = software_latency(frames)
    intervals = [(g["perf_read"] - f["perf_read"]) * 1000
                 for f, g in zip(frames, frames[1:])]
    print(f"frames: {len(frames)}, median frame interval "
          f"{statistics.median(intervals):.1f} ms" if intervals else "no frames")
    if not ms:
        print("no immediate loop starts in this log "
              "(start a loop while no other loop is playing)")
        return
    ms.sort()
    p90 = ms[min(len(ms) - 1, int(round(0.9 * (len(ms) - 1))))]
    print(f"immediate starts: {len(ms)}")
    print(f"software latency (frame read → play call), ms: "
          f"median {statistics.median(ms):.1f}, p90 {p90:.1f}, "
          f"min {ms[0]:.1f}, max {ms[-1]:.1f}")
    print("excludes camera capture and audio output latency — see docs/MEASUREMENT.md")


def main(argv: Optional[List[str]] = None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("replay", help="replay all clips in a manifest")
    r.add_argument("--label", required=True,
                   help="name for this run, e.g. baseline or lowlight-v1")
    r.add_argument("--manifest", type=Path, default=REPO / "recordings" / "clips.csv")
    r.add_argument("--out", type=Path, help="default: results/<label>")
    r.set_defaults(func=cmd_replay)

    c = sub.add_parser("compare", help="compare two replay results")
    c.add_argument("before", type=Path)
    c.add_argument("after", type=Path)
    c.set_defaults(func=cmd_compare)

    lat = sub.add_parser("latency", help="software latency from a live log")
    lat.add_argument("log", type=Path)
    lat.set_defaults(func=cmd_latency)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
