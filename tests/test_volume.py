"""Regression for normal gestures/hand withdrawal accidentally muting audio."""

import unittest
from types import SimpleNamespace

from test_gestures import app, hand, POSES


def pinch(spread, x=0.5, scale=1.0):
    """A stationary palm with independently adjustable thumb/index spacing."""
    points = [SimpleNamespace(x=x, y=0.5, z=0.0) for _ in range(21)]
    points[0].y = 0.65
    points[5].x, points[17].x = x - 0.05 * scale, x + 0.05 * scale
    points[4].x, points[8].x = x - 0.05 * scale * spread, x + 0.05 * scale * spread
    return SimpleNamespace(landmark=points)


class VolumeControlTest(unittest.TestCase):
    def setUp(self):
        self.v = app.PinchVolumeControl(app.Config())

    def hold(self, spread, start=0, count=10, **kwargs):
        for i in range(count):
            self.v.update(pinch(spread, **kwargs), start + i / 30, 1280, 720)

    def test_disabled_gestures_never_change_volume(self):
        self.hold(1.5)
        self.hold(0, start=0.5)
        self.v.hand_lost()
        self.assertEqual(self.v.value, 0.8)

    def test_entering_with_closed_fingers_does_not_jump_to_zero(self):
        self.v.set_enabled(True, 0.65)
        self.hold(0)
        self.assertAlmostEqual(self.v.value, 0.65)

    def test_relative_deliberate_adjustment_and_lock(self):
        self.v.set_enabled(True, 0.6)
        self.hold(1.0)
        self.hold(1.3, start=0.4)
        self.assertAlmostEqual(self.v.value, 0.8)
        self.v.set_enabled(False)
        self.hold(0, start=0.8)
        self.assertAlmostEqual(self.v.value, 0.8)

    def test_last_noisy_closed_frames_before_loss_do_not_mute(self):
        self.v.set_enabled(True, 0.6)
        self.hold(1.0)
        self.hold(0, start=0.34, count=2)
        self.v.hand_lost()
        self.assertFalse(self.v.enabled)
        self.assertAlmostEqual(self.v.value, 0.6)
        self.hold(0, start=0.5)
        self.assertAlmostEqual(self.v.value, 0.6)

    def test_withdrawal_motion_and_palm_scale_change_freeze_adjustment(self):
        self.v.set_enabled(True, 0.6)
        self.hold(1.0)
        for i in range(1, 8):
            self.v.update(pinch(0, x=0.5 + i * 0.03), 0.3 + i / 30, 1280, 720)
        self.assertAlmostEqual(self.v.value, 0.6)
        self.v.set_enabled(True, 0.6)
        self.hold(1.0, start=1.0)
        for i in range(1, 8):
            self.v.update(pinch(0, scale=0.85 ** i), 1.3 + i / 30, 1280, 720)
        self.assertAlmostEqual(self.v.value, 0.6)

    def test_palm_relative_spacing_is_independent_of_distance(self):
        for scale in (0.5, 1.0, 2.0):
            self.v = app.PinchVolumeControl(app.Config())
            self.v.set_enabled(True, 0.6)
            self.hold(1.0, scale=scale)
            self.hold(1.3, start=0.4, scale=scale)
            self.assertAlmostEqual(self.v.value, 0.8)

    def test_partly_out_of_frame_hand_cannot_write_volume(self):
        self.v.set_enabled(True, 0.6)
        self.hold(1.0)
        for i in range(10):
            lm = pinch(0)
            lm.landmark[20].x = 1.1
            self.v.update(lm, 0.4 + i / 30, 1280, 720)
        self.assertAlmostEqual(self.v.value, 0.6)


class ControllerVolumeTest(unittest.TestCase):
    def setUp(self):
        self.c = object.__new__(app.GestureMusicController)
        self.c.cfg = app.Config()
        self.c.audio = app.AudioEngine(self.c.cfg, demo_mode=True)
        self.c.volume_control = app.PinchVolumeControl(self.c.cfg)
        self.c.gesture = app.GestureEngine(self.c.cfg)
        self.c.loop_states = [{"status": "inactive"} for _ in range(5)]
        self.c.master_clock = None
        self.c.quant_queue, self.c.stop_queue = [], []
        self.c.anim = SimpleNamespace(add=lambda *args, **kwargs: None)

    def frame(self, lm, t):
        result = SimpleNamespace(multi_hand_landmarks=[lm] if lm else [], multi_handedness=[])
        return self.c._handle_gesture(result, 1280, t, 720)

    def test_normal_loop_selection_and_withdrawal_leave_audio_unchanged(self):
        self.c.audio.set_volume(0.65)
        pose = next(p for p in POSES if p["expected"] == 1)
        for i in range(16):
            self.frame(hand(pose, x=0.4), i / 30)
        events = []
        for i in range(1, 12):
            info = self.frame(hand(pose, x=0.4 + 0.7 * i / 30), 0.5 + i / 30)
            if "swipe" in info:
                events.append((info["swipe"], info["fingers"], info.get("action")))
        self.frame(None, 1.0)
        self.assertEqual(events, [("right", 1, "start")])
        self.assertAlmostEqual(self.c.audio.master_volume, 0.65)

    def test_volume_mode_blocks_swipes_and_loss_locks_last_level(self):
        self.c.audio.set_volume(0.65)
        self.c._handle_key(ord("v"), None)
        self.assertTrue(self.c.volume_control.enabled)
        pose = next(p for p in POSES if p["expected"] == 1)
        for i in range(16):
            self.frame(hand(pose, x=0.4), i / 30)
        for i in range(1, 12):
            info = self.frame(hand(pose, x=0.4 + 0.7 * i / 30), 0.5 + i / 30)
            self.assertNotIn("swipe", info)
        last_volume = self.c.audio.master_volume
        info = self.frame(None, 1.0)
        self.assertFalse(info["volume_adjusting"])
        self.assertAlmostEqual(self.c.audio.master_volume, last_volume)
        self.assertEqual(self.c.quant_queue, [])


if __name__ == "__main__":
    unittest.main()
