"""R1 regressions: visible finger poses and timestamped wrist trajectories."""

import copy
import json
import math
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import main as app

POSES = json.loads((ROOT / "tests/fixtures/finger_poses.json").read_text())["poses"]


def hand(pose, x=None, y=None):
    points = copy.deepcopy(pose["landmarks"])
    dx = 0 if x is None else x - points[0][0]
    dy = 0 if y is None else y - points[0][1]
    return SimpleNamespace(landmark=[SimpleNamespace(x=p[0] + dx, y=p[1] + dy, z=p[2])
                                     for p in points])


class FingerCountTest(unittest.TestCase):
    def test_reviewed_frames(self):
        for pose in POSES:
            with self.subTest(frame=pose["frame"], view=pose["view"]):
                self.assertEqual(app.count_fingers(hand(pose), pose["handedness"], 1280, 720),
                                 pose["expected"])

    def test_mirror_rotation_and_handedness_do_not_change_count(self):
        for pose in POSES:
            for angle in (0, math.pi / 2, math.pi):
                for mirror in (-1, 1):
                    lm = hand(pose)
                    for p in lm.landmark:
                        x, y = (p.x - 0.5) * mirror, (p.y - 0.5) * 720 / 1280
                        p.x = 0.5 + x * math.cos(angle) - y * math.sin(angle)
                        p.y = 0.5 + (x * math.sin(angle) + y * math.cos(angle)) * 1280 / 720
                    for label in ("Left", "Right"):
                        with self.subTest(frame=pose["frame"], angle=angle, mirror=mirror, label=label):
                            self.assertEqual(app.count_fingers(lm, label, 1280, 720), pose["expected"])

    def test_degenerate_and_nonfinite_landmarks_are_not_fingers(self):
        lm = hand(POSES[0])
        for p in lm.landmark:
            p.x = p.y = p.z = 0
        self.assertEqual(app.count_fingers(lm), 0)
        lm.landmark[4].x = float("nan")
        self.assertEqual(app.count_fingers(lm), 0)


class SwipeTest(unittest.TestCase):
    def setUp(self):
        self.g = app.GestureEngine(app.Config())
        self.one = next(p for p in POSES if p["expected"] == 1)
        self.five = next(p for p in POSES if p["expected"] == 5)
        self.events = []

    def feed(self, t, x, pose=None, y=0.65, width=1280, height=720):
        self.g.update_hand(hand(pose or self.one, x=x, y=y), "Right", width, t, height)
        event = self.g.detect_swipe(t)
        if event:
            self.events.append((event, self.g.swipe_fingers))

    def ready(self, fps=30, pose=None, x=0.4):
        for i in range(round(0.5 * fps) + 1):
            self.feed(i / fps, x, pose=pose)
        self.assertGreater(self.g.finger_count, 0)
        return i / fps

    def test_both_directions_at_multiple_frame_rates_and_resolutions(self):
        for fps in (15, 30, 60):
            for width, height in ((640, 360), (1280, 720), (1920, 1080)):
                for sign in (-1, 1):
                    with self.subTest(fps=fps, width=width, direction=sign):
                        self.g = app.GestureEngine(app.Config())
                        self.events = []
                        start = self.ready(fps=fps)
                        for i in range(1, round(0.35 * fps) + 1):
                            self.feed(start + i / fps, 0.4 + sign * 0.7 * i / fps,
                                      width=width, height=height)
                        self.assertEqual(self.events, [("right" if sign > 0 else "left", 1)])

    def test_raw_count_flicker_during_stroke_does_not_change_selected_loop(self):
        self.ready()
        for i in range(1, 12):
            self.feed(0.5 + i / 30, 0.4 + 0.7 * i / 30, pose=self.five)
        self.assertEqual(self.g.raw_finger_count, 5)
        self.assertEqual(self.events, [("right", 1)])

    def test_zero_finger_frame_during_stroke_keeps_selected_loop(self):
        self.ready()
        collapsed = copy.deepcopy(self.one)
        collapsed["landmarks"][4] = collapsed["landmarks"][17][:]
        for mcp in (5, 9, 13, 17):
            collapsed["landmarks"][mcp + 3] = collapsed["landmarks"][mcp + 1][:]
        for i in range(1, 12):
            self.feed(0.5 + i / 30, 0.4 + 0.7 * i / 30, pose=collapsed)
        self.assertEqual(self.g.raw_finger_count, 0)
        self.assertEqual(self.events, [("right", 1)])

    def test_short_tracking_gap_holds_selection_but_cannot_emit(self):
        self.ready()
        self.g.hand_lost(0.6)
        self.assertEqual(self.g.finger_count, 1)
        self.assertTrue(self.g.tracking_held)
        self.assertIsNone(self.g.detect_swipe(0.6))
        # Reappearing elsewhere must not invent a swipe across the gap.
        self.feed(0.65, 0.8, pose=self.five)
        self.assertEqual(self.events, [])
        self.assertEqual(self.g.finger_count, 1)
        for i in range(1, 10):
            self.feed(0.65 + i / 30, 0.8 - 0.7 * i / 30, pose=self.five)
        self.assertEqual(self.events, [("left", 1)])

    def test_long_loss_clears_selection_and_motion(self):
        self.ready()
        self.g.hand_lost(0.8)
        self.assertEqual(self.g.finger_count, 0)
        self.feed(0.85, 0.8)
        self.assertEqual(self.events, [])
        self.assertEqual(self.g.finger_count, 0)

    def test_gap_between_processed_frames_and_clock_reversal_reset(self):
        self.ready()
        self.feed(1.0, 0.9)
        self.assertEqual(self.g.finger_count, 0)
        self.assertEqual(self.events, [])
        self.feed(0.9, 0.1)
        self.assertEqual(self.events, [])

    def test_idle_jitter_slow_drift_and_vertical_motion_do_not_fire(self):
        for kind in ("jitter", "slow", "vertical"):
            self.g = app.GestureEngine(app.Config())
            self.events = []
            self.ready()
            for i in range(1, 61):
                dt = i / 30
                x = 0.4 + (0.004 * math.sin(i) if kind == "jitter" else 0.1 * dt)
                y = 0.65 - (0.5 * dt if kind == "vertical" else 0)
                self.feed(0.5 + dt, x, y=y)
            self.assertEqual(self.events, [], kind)

    def test_single_landmark_jump_does_not_fire(self):
        self.ready()
        for i in range(1, 15):
            self.feed(0.5 + i / 30, 0.75)
        self.assertEqual(self.events, [])

    def test_sustained_stroke_only_fires_once_even_after_cooldown(self):
        self.ready(x=0.1)
        for i in range(1, 46):
            self.feed(0.5 + i / 30, 0.1 + 0.5 * i / 30)
        self.assertEqual(self.events, [("right", 1)])

    def test_new_selection_can_be_made_after_motion_stops(self):
        self.ready()
        for i in range(1, 31):
            self.feed(0.5 + i / 30, 0.4, pose=self.five)
        self.assertEqual(self.g.finger_count, 5)
        for i in range(1, 12):
            self.feed(1.5 + i / 30, 0.4 + 0.7 * i / 30, pose=self.five)
        self.assertEqual(self.events, [("right", 5)])

    def test_cooldown_applies_even_if_a_swipe_has_no_audio_action(self):
        self.ready()
        for i in range(1, 10):
            self.feed(0.5 + i / 30, 0.4 + 0.7 * i / 30)
        for i in range(1, 8):
            self.feed(0.8 + i / 30, 0.61 - 0.7 * i / 30)
        self.assertEqual(self.events, [("right", 1)])

    def test_selection_updates_while_the_hand_moves_slowly(self):
        self.ready()
        for i in range(1, 10):    # show 5 while drifting right at 0.2 widths/s
            self.feed(0.5 + i / 30, 0.4 + 0.2 * i / 30, pose=self.five)
        self.assertEqual(self.g.finger_count, 5)
        for i in range(1, 12):    # the same movement speeds up into a swipe
            self.feed(0.8 + i / 30, 0.46 + 0.7 * i / 30, pose=self.five)
        self.assertEqual(self.events, [("right", 5)])

    def test_target_returns_after_a_long_loss_while_the_hand_moves(self):
        self.ready()
        self.g.hand_lost(0.6)
        self.g.hand_lost(1.0)     # longer than the grace period: cleared
        self.assertEqual(self.g.finger_count, 0)
        for i in range(8):        # back in view, drifting at 0.2 widths/s
            self.feed(1.5 + i / 30, 0.3 + 0.2 * i / 30, pose=self.five)
        self.assertEqual(self.g.finger_count, 5)


if __name__ == "__main__":
    unittest.main()
