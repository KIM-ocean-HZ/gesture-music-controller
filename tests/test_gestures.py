"""R1 regressions: finger counts on reviewed palm/back poses."""

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


if __name__ == "__main__":
    unittest.main()
