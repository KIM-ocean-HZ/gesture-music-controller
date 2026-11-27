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

    def test_only_a_recorded_action_starts_the_cooldown(self):
        # Wind-up to the left, then the intended swipe right straight after.
        self.ready()
        for i in range(1, 10):
            self.feed(0.5 + i / 30, 0.4 - 0.7 * i / 30)
        for i in range(1, 12):
            self.feed(0.8 + i / 30, 0.19 + 0.7 * i / 30)
        self.assertEqual(self.events, [("left", 1), ("right", 1)])
        # Once the caller records an action, the return stroke is ignored.
        self.g.record_action(1.2)
        self.events = []
        for i in range(1, 12):
            self.feed(1.2 + i / 30, 0.45 - 0.7 * i / 30)
        self.assertEqual(self.events, [])

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


class SoundSetTest(unittest.TestCase):
    def test_demo_mode_reads_loop_length_from_the_wav_header(self):
        import tempfile
        import wave
        from unittest import mock
        with tempfile.TemporaryDirectory() as tmp:
            with wave.open(str(Path(tmp) / "loop2.wav"), "wb") as w:
                w.setnchannels(2)
                w.setsampwidth(2)
                w.setframerate(44100)
                w.writeframes(b"\0" * 4 * 44100 * 4)          # 4.0 s
            with mock.patch.object(app, "ASSETS_DIR", Path(tmp)):
                audio = app.AudioEngine(app.Config(), demo_mode=True)
                audio.set_sound_set("HousePiano")
                self.assertAlmostEqual(audio.loop_length(1), 4.0)
                self.assertEqual(audio.loop_length(0), 8.0)  # loop1.wav missing

    def test_controller_starts_in_the_configured_set(self):
        c = app.GestureMusicController(app.Config(sound_set="FunkGuitar"), demo_mode=True,
                                       source=SimpleNamespace(release=lambda: None),
                                       headless=True)
        try:
            self.assertEqual(c._SET_NAMES[c._set_idx], "FunkGuitar")
            self.assertEqual(c.audio._active_set, "FunkGuitar")
        finally:
            c._mp_hands.close()
            c._mp_face.close()


def controller():
    """A controller without camera, window or audio, for gesture → loop tests."""
    c = object.__new__(app.GestureMusicController)
    c.cfg = app.Config()
    c.audio = app.AudioEngine(c.cfg, demo_mode=True)
    c.volume_control = app.PinchVolumeControl(c.cfg)
    c.gesture = app.GestureEngine(c.cfg)
    c.loop_states = [{"status": "inactive"} for _ in range(5)]
    c.master_clock, c.master_length = None, 0.0
    c.quant_queue, c.stop_queue = [], []
    c.anim = SimpleNamespace(add=lambda *args, **kwargs: None)
    return c


class ControllerSwipeTest(unittest.TestCase):
    def setUp(self):
        self.c = controller()
        self.one = next(p for p in POSES if p["expected"] == 1)
        self.events = []

    def frame(self, t, x):
        # Same order as run(): gesture first, then due starts/stops.
        result = SimpleNamespace(multi_hand_landmarks=[hand(self.one, x=x, y=0.65)],
                                 multi_handedness=[])
        info = self.c._handle_gesture(result, 1280, t, 720)
        self.c._process_queues(t)
        if "swipe" in info:
            self.events.append((info["swipe"], info["fingers"], info.get("action")))

    def stroke(self, t, x, direction, frames=11, speed=0.7):
        for i in range(1, frames + 1):
            self.frame(t + i / 30, x + direction * speed * i / 30)
        return t + frames / 30, x + direction * speed * frames / 30

    def rest(self, t, x, seconds):
        for i in range(1, round(seconds * 30) + 1):
            self.frame(t + i / 30, x)
        return t + round(seconds * 30) / 30

    def loop_two_playing(self):
        """Master clock running (8 s loops) with loop 2 playing."""
        self.c.master_clock, self.c.master_length = 0.0, 8.0
        self.c.loop_states[1]["status"] = "locked"

    def test_left_swipe_cancels_a_queued_loop(self):
        self.loop_two_playing()
        t = self.rest(0, 0.4, 0.5)
        t, x = self.stroke(t, 0.4, +1)            # R1: queued for t = 8 s
        t = self.rest(t, x, 0.8)
        t, x = self.stroke(t, x, -1)              # L1 before the boundary
        self.rest(8.0, x, 0.3)
        self.assertEqual(self.events, [("right", 1, "queue"), ("left", 1, "cancel_start")])
        self.assertEqual(self.c.loop_states[0]["status"], "inactive")
        self.assertEqual(self.c.quant_queue, [])

    def test_right_swipe_keeps_a_stopping_loop(self):
        self.loop_two_playing()
        self.c.loop_states[0]["status"] = "locked"
        t = self.rest(0, 0.6, 0.5)
        t, x = self.stroke(t, 0.6, -1)            # L1: stops at t = 8 s
        t = self.rest(t, x, 0.8)
        t, x = self.stroke(t, x, +1)              # R1 before the boundary
        self.rest(8.0, x, 0.3)
        self.assertEqual(self.events, [("left", 1, "stop"), ("right", 1, "cancel_stop")])
        self.assertEqual(self.c.loop_states[0]["status"], "locked")
        self.assertEqual(self.c.stop_queue, [])

    def test_stopped_loops_are_reported_for_the_log(self):
        self.c.loop_states[0]["status"] = "stopping"
        self.c.stop_queue = [{"index": 0, "stop_time": 8.0}]
        self.c._process_queues(7.9)
        self.assertEqual(self.c.last_stopped, [])
        self.c._process_queues(8.0)
        self.assertEqual(self.c.last_stopped, [0])
        self.assertEqual(self.c.loop_states[0]["status"], "inactive")

    def test_return_stroke_that_outlasts_the_cooldown_does_not_fire(self):
        t = self.rest(-1 / 30, 0.4, 0.5)
        t, x = self.stroke(t, 0.4, +1)            # start from silence
        self.stroke(t, x, -1, frames=36, speed=0.4)  # 1.2 s swing back
        self.assertEqual(self.events, [("right", 1, "start")])
        self.assertEqual(self.c.loop_states[0]["status"], "locked")

    def test_wind_up_before_a_swipe_does_not_block_it(self):
        for i in range(16):
            self.frame(i / 30, 0.4)
        for i in range(1, 10):    # wind-up left: loop 1 is not playing, no-op
            self.frame(0.5 + i / 30, 0.4 - 0.7 * i / 30)
        for i in range(1, 12):    # the intended swipe right
            self.frame(0.8 + i / 30, 0.19 + 0.7 * i / 30)
        self.assertEqual(self.events, [("left", 1, None), ("right", 1, "start")])
        self.assertEqual(self.c.loop_states[0]["status"], "locked")

    def test_return_stroke_after_a_start_does_not_stop_the_loop(self):
        for i in range(16):
            self.frame(i / 30, 0.4)
        for i in range(1, 12):
            self.frame(0.5 + i / 30, 0.4 + 0.7 * i / 30)
        for i in range(1, 12):    # hand swings back at once
            self.frame(0.87 + i / 30, 0.65 - 0.7 * i / 30)
        self.assertEqual(self.events, [("right", 1, "start")])
        self.assertEqual(self.c.loop_states[0]["status"], "locked")


if __name__ == "__main__":
    unittest.main()
