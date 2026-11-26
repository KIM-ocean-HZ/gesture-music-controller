"""Unit tests for the measurement maths and the swipe detector.

Run:  python -m unittest discover tests
"""

import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "bench"))
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import bench  # noqa: E402
import main as app  # noqa: E402


def frames(spec, fps=10.0, t0=100.0):
    """Build frame records from a compact spec: list of dicts merged per frame."""
    return [{"i": i, "t": t0 + i / fps, "hand": True, **extra}
            for i, extra in enumerate(spec)]


class AlignTest(unittest.TestCase):
    def test_perfect(self):
        self.assertEqual(bench.align(["R1", "L1"], ["R1", "L1"]),
                         {"hits": 2, "substitutions": 0, "misses": 0, "extras": 0})

    def test_wrong_finger_is_substitution(self):
        c = bench.align(["R1", "R2"], ["R1", "R3"])
        self.assertEqual((c["hits"], c["substitutions"]), (1, 1))

    def test_miss_and_extra(self):
        c = bench.align(["R1", "R2", "L1"], ["R1", "L4", "R2", "L1", "L1"])
        self.assertEqual(c["hits"], 3)
        self.assertEqual(c["misses"] + c["substitutions"], 0)
        self.assertEqual(c["extras"], 2)

    def test_nothing_detected(self):
        self.assertEqual(bench.align(["R1", "R2"], [])["misses"], 2)


class ClipMetricsTest(unittest.TestCase):
    def test_tracking_loss_and_dropouts(self):
        spec = [{}, {"hand": False}, {"hand": False}, {}, {"hand": False}, {}]
        m = bench.clip_metrics(frames(spec), "idle")
        self.assertEqual((m["lost_frames"], m["dropouts"]), (3, 2))
        self.assertAlmostEqual(m["duration_s"], 0.6, places=6)

    def test_absent_interval_is_excluded(self):
        spec = [{}, {"hand": False}, {"hand": False}, {}]
        m = bench.clip_metrics(frames(spec), "idle", absent=[(0.05, 0.25)])
        self.assertEqual((m["hand_expected_frames"], m["lost_frames"]), (2, 0))

    def test_scripted(self):
        spec = [{"swipe": "right", "fingers": 1}, {},
                {"swipe": "left", "fingers": 2}, {}]
        m = bench.clip_metrics(frames(spec), "scripted", expected=["R1", "L1"])
        self.assertEqual(m["detected"], "R1 L2")
        self.assertEqual((m["hits"], m["substitutions"]), (1, 1))

    def test_aggregate_rates(self):
        idle = bench.clip_metrics(
            frames([{"swipe": "right", "fingers": 1}] + [{}] * 599), "idle")
        agg = bench.aggregate([idle])
        self.assertEqual(agg["false_triggers_per_min"], 1.0)
        self.assertEqual(agg["tracking_loss_pct"], 0.0)


class LatencyTest(unittest.TestCase):
    def test_start_to_play(self):
        log = [
            {"perf_read": 1.000, "action": "start", "fingers": 2},
            {"perf_read": 1.033, "played": [2], "perf_play": [1.040]},
        ]
        self.assertEqual([round(x, 3) for x in bench.software_latency(log)], [40.0])

    def test_played_in_the_triggering_frame(self):
        log = [{"perf_read": 2.000, "action": "start", "fingers": 3,
                "played": [3], "perf_play": [2.012]}]
        self.assertEqual([round(x, 3) for x in bench.software_latency(log)], [12.0])


class RunLoopLatencyTest(unittest.TestCase):
    """The real frame loop, with a fake camera and fake hand landmarks."""

    def test_start_from_silence_plays_in_the_triggering_frame(self):
        import tempfile
        import numpy as np
        from test_gestures import hand, POSES
        one = next(p for p in POSES if p["expected"] == 1)
        xs = [0.4] * 16 + [0.4 + 0.7 * i / 30 for i in range(1, 12)]

        class Camera:
            live, nominal_fps, i = True, 30.0, 0

            def read(self):
                if self.i >= len(xs):
                    return False, None, 0.0
                self.i += 1
                return True, np.zeros((72, 128, 3), np.uint8), (self.i - 1) / 30

            def release(self):
                pass

        camera = Camera()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "live.jsonl"
            cfg = app.Config()
            c = app.GestureMusicController(
                cfg, demo_mode=True, source=camera, headless=True,
                frame_log=app.FrameLog(path, app.run_header(cfg, "live", None)))
            c._mp_hands.close()
            c._mp_face.close()
            c._mp_hands = SimpleNamespace(close=lambda: None, process=lambda rgb: SimpleNamespace(
                multi_hand_landmarks=[hand(one, x=xs[camera.i - 1], y=0.65)], multi_handedness=[]))
            c._mp_face = SimpleNamespace(close=lambda: None,
                                         process=lambda rgb: SimpleNamespace(multi_face_landmarks=None))
            c.run()
            _, frames = bench.read_log(path)
        starts = [f for f in frames if f.get("action") == "start"]
        self.assertEqual(len(starts), 1)
        self.assertEqual(starts[0].get("played"), [1])
        self.assertEqual(len(bench.software_latency(frames)), 1)


class ManifestTest(unittest.TestCase):
    def test_example_manifest_parses(self):
        rows = bench.read_manifest(REPO / "bench" / "clips.example.csv")
        self.assertTrue(rows)
        self.assertTrue(all(r["kind"] in ("idle", "scripted") for r in rows))


if __name__ == "__main__":
    unittest.main()
