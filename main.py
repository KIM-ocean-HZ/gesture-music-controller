#!/usr/bin/env python3
"""
Gesture Music Controller
========================
Real-time, hands-free music loop station driven by webcam-based hand gesture
and head-motion recognition (MediaPipe + OpenCV).

Controls
--------
Gesture (fingers shown)  1–5  →  target loop 1–5
Swipe right              →  queue / start targeted loop on next beat
Swipe left               →  stop targeted loop on next beat
V, then pinch           →  adjust volume; V again locks the current level

Keyboard:
  1 / 2       Artistic / Reality visual mode
  p / g / f   HousePiano / BluesGuitar / FunkGuitar sound set
  s           Save screenshot
  h           Toggle help overlay
  v           Toggle volume adjustment (locked by default)
  q           Quit

Usage:
  python main.py                         # default camera, 1280×720
  python main.py --camera 1              # second webcam
  python main.py --demo                  # no audio required (visual only)
  python main.py --width 640 --height 480 --swipe-threshold 0.35

Measurement (see docs/MEASUREMENT.md):
  python main.py --demo --record recordings/clip.mp4     # save raw camera frames
  python main.py --input recordings/clip.mp4 --log out.jsonl   # headless replay
  python main.py --log live.jsonl                        # per-frame log, live
"""

import argparse
import csv
import hashlib
import json
import logging
import math
import random
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

# ---------------------------------------------------------------------------
# Optional pygame — gracefully disabled in demo mode
# ---------------------------------------------------------------------------
try:
    import pygame
    _PYGAME_OK = True
except ImportError:
    _PYGAME_OK = False

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ASSETS_DIR = Path(__file__).parent / "assets"

# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------
@dataclass
class Config:
    # Camera
    camera_index: int = 0
    frame_width: int = 1280
    frame_height: int = 720

    # Gesture thresholds
    swipe_velocity_threshold: float = 0.35  # frame widths / second
    swipe_min_distance: float = 0.09     # fraction of image width
    swipe_window_s: float = 0.45
    swipe_min_duration_s: float = 0.06
    velocity_smoothing_s: float = 0.06   # time constant, independent of FPS
    selection_settle_s: float = 0.12
    tracking_grace_s: float = 0.22       # hold selection, never guess lost motion
    motion_release_speed: float = 0.12  # frame widths / second
    motion_release_s: float = 0.12
    action_cooldown: float = 0.8         # minimum seconds between actions

    # Head-nod
    nod_cooldown: float = 0.15
    nod_amplitude_threshold: float = 0.004
    head_history_len: int = 4

    # Volume control via pinch gesture
    default_volume: float = 0.8
    volume_settle_s: float = 0.15
    volume_stability_tolerance: float = 0.08  # spread in palm widths
    volume_spread_range: float = 1.5         # palm widths per full volume range
    volume_motion_limit: float = 0.25        # image widths / second

    # UI
    dashboard_height: int = 120
    dashboard_alpha: float = 0.75
    fps_smoothing: int = 30

    # Sound sets: keys are display names, values are filename lists
    sound_sets: Dict[str, List[str]] = field(default_factory=lambda: {
        "HousePiano":  [f"loop{i}.wav"   for i in range(1, 6)],
        "BluesGuitar": [f"guitar{i}.wav" for i in range(1, 6)],
        "FunkGuitar":  [f"funk{i}.wav"   for i in range(1, 6)],
    })
    startup_sound: str = "startup.wav"
    confirm_sound: str = "confirm.wav"


# ---------------------------------------------------------------------------
# Colour palettes
# ---------------------------------------------------------------------------
LOOP_COLORS: List[Tuple[int, int, int]] = [
    (255, 191,   0),   # amber
    (  0, 165, 255),   # sky-blue
    (218, 112, 214),   # orchid
    (203, 192, 255),   # lavender
    (255,   0, 102),   # rose
]

ARTISTIC_PALETTE: List[Tuple[int, int, int]] = [
    (255,   0,   0), (255, 128,   0), (255, 255,   0),
    (  0, 255,   0), (  0, 255, 255), (  0, 128, 255),
    (  0,   0, 255), (128,   0, 255), (255,   0, 255),
]

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------
def adjust_brightness(
    color: Tuple[int, int, int], factor: float
) -> Tuple[int, int, int]:
    return tuple(min(255, int(c * factor)) for c in color)


def landmark_px(lm, w: int, h: int) -> Tuple[int, int]:
    return int(lm.x * w), int(lm.y * h)


def dist_norm(a, b) -> float:
    """Euclidean distance between two MediaPipe landmarks (normalised coords)."""
    return math.hypot(a.x - b.x, a.y - b.y)


# ---------------------------------------------------------------------------
# Finger counting in hand geometry, independent of palm/back and handedness
# ---------------------------------------------------------------------------
_TIP_IDS = [4, 8, 12, 16, 20]


def count_fingers(hand_landmarks, handedness_label: str = "",
                  frame_width: int = 1, frame_height: int = 1) -> int:
    """Count extension using joint shape, not screen-up or thumb x direction.

    MediaPipe x/z use approximately image-width units; y uses image height.
    Correct the aspect ratio before comparing angles/distances. The handedness
    argument is retained for callers but does not decide whether a thumb is up.
    These geometric rules still depend on the quality of the detected landmarks.
    """
    points = np.array([(p.x, p.y * frame_height / frame_width, p.z)
                       for p in hand_landmarks.landmark], dtype=float)
    if points.shape != (21, 3) or not np.isfinite(points).all():
        return 0

    def distance(a, b):
        return float(np.linalg.norm(points[a] - points[b]))

    def straight(a, joint, b, degrees):
        u, v = points[a] - points[joint], points[b] - points[joint]
        size = float(np.linalg.norm(u) * np.linalg.norm(v))
        return size > 1e-12 and float(np.dot(u, v)) / size < math.cos(math.radians(degrees))

    palm_width = distance(5, 17)
    across = points[5, :2] - points[17, :2]          # little → index knuckle
    palm_width_2d = float(np.linalg.norm(across))
    if palm_width < 1e-6 or palm_width_2d < 1e-6:
        return 0
    # A tucked thumb can be straight too: require it to reach away from the
    # little-finger knuckle, measured against this hand's own palm width, and
    # to stick out sideways past the index knuckle by > 0.6 palm widths. A
    # thumb folded beside or behind the palm (back-of-hand "4") is straight
    # and far enough from the little finger, but does not stick out sideways.
    thumb = (straight(2, 3, 4, 135) and distance(4, 17) > 1.25 * palm_width
             and float(np.dot(points[4, :2] - points[5, :2], across)) > 0.6 * palm_width_2d ** 2)
    # For the four fingers retain the visible tip-past-knuckle rule, but
    # measure it along the palm's own axis instead of screen-up. Requiring
    # perfectly straight 3-D joints rejects naturally relaxed raised fingers;
    # the small palm-relative margin rejects nearly coincident noisy points.
    axis = points[9, :2] - points[0, :2]
    axis_length = float(np.linalg.norm(axis))
    if axis_length < 1e-6:
        return int(thumb)
    margin = 0.04 * palm_width_2d
    axis /= axis_length
    fingers = sum(float(np.dot(points[mcp + 3, :2] - points[mcp + 1, :2], axis)) > margin
                  for mcp in (5, 9, 13, 17))
    return int(thumb) + fingers


# ---------------------------------------------------------------------------
# Audio engine
# ---------------------------------------------------------------------------
class AudioEngine:
    """
    Wraps pygame.mixer.  Provides beat-quantised multi-set loop management
    and degrades gracefully when pygame is unavailable or in demo mode.
    """

    def __init__(self, cfg: Config, demo_mode: bool = False):
        self.cfg = cfg
        self.demo_mode = demo_mode or not _PYGAME_OK
        self.master_volume: float = cfg.default_volume
        self._sounds: Dict[str, List[Optional[object]]] = {}
        self._startup: Optional[object] = None
        self._confirm: Optional[object] = None
        self._active_set: str = ""

        if self.demo_mode:
            log.warning("Audio disabled (demo mode or pygame unavailable).")
            return

        try:
            pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
            pygame.mixer.set_num_channels(20)
        except pygame.error as exc:
            log.error("Pygame mixer init failed: %s — running without audio.", exc)
            self.demo_mode = True
            return

        self._load_all()

    # ------------------------------------------------------------------
    def _load_all(self):
        missing: List[str] = []
        for set_name, files in self.cfg.sound_sets.items():
            sounds: List[Optional[object]] = []
            for fname in files:
                path = ASSETS_DIR / fname
                if not path.exists():
                    missing.append(fname)
                    sounds.append(None)
                    continue
                try:
                    snd = pygame.mixer.Sound(str(path))
                    snd.set_volume(self.master_volume)
                    sounds.append(snd)
                except pygame.error as exc:
                    log.warning("Cannot load %s: %s", path.name, exc)
                    sounds.append(None)
            self._sounds[set_name] = sounds

        for fname, attr in [
            (self.cfg.startup_sound, "_startup"),
            (self.cfg.confirm_sound, "_confirm"),
        ]:
            path = ASSETS_DIR / fname
            if path.exists():
                try:
                    setattr(self, attr, pygame.mixer.Sound(str(path)))
                except pygame.error:
                    pass

        if missing:
            log.warning(
                "Audio files not found in assets/ (loops will be silent): %s",
                ", ".join(missing),
            )

    # ------------------------------------------------------------------
    @property
    def current_sounds(self) -> List[Optional[object]]:
        return self._sounds.get(self._active_set, [])

    def set_sound_set(self, name: str):
        self._active_set = name

    def play_loop(self, idx: int):
        if self.demo_mode:
            return
        sounds = self.current_sounds
        if idx < len(sounds) and sounds[idx]:
            sounds[idx].set_volume(self.master_volume)
            sounds[idx].play(loops=-1)

    def stop_loop(self, idx: int):
        if self.demo_mode:
            return
        sounds = self.current_sounds
        if idx < len(sounds) and sounds[idx]:
            sounds[idx].stop()

    def stop_all(self):
        if self.demo_mode:
            return
        for sounds in self._sounds.values():
            for s in sounds:
                if s:
                    s.stop()

    def play_confirm(self):
        if not self.demo_mode and self._confirm:
            self._confirm.play()

    def play_startup(self):
        if not self.demo_mode and self._startup:
            self._startup.play()

    def set_volume(self, volume: float):
        self.master_volume = max(0.0, min(1.0, volume))
        if self.demo_mode:
            return
        for sounds in self._sounds.values():
            for s in sounds:
                if s:
                    s.set_volume(self.master_volume)

    def loop_length(self, idx: int) -> float:
        """Return audio duration in seconds; fall back to 8 s if unavailable."""
        if self.demo_mode:
            return 8.0
        sounds = self.current_sounds
        if idx < len(sounds) and sounds[idx]:
            length = sounds[idx].get_length()
            return length if length > 0 else 8.0
        return 8.0

    def quit(self):
        if not self.demo_mode and _PYGAME_OK:
            pygame.mixer.quit()


# ---------------------------------------------------------------------------
# Explicit volume control, separate from loop-selection gestures
# ---------------------------------------------------------------------------
class PinchVolumeControl:
    """Relative pinch adjustment, enabled explicitly and committed when steady.

    Selection/swiping cannot write volume while this mode is off. Reacquiring a
    hand or entering the mode anchors to the current value, never to absolute
    finger spacing. Fast withdrawal, edge clipping and loss discard a pending
    adjustment instead of committing the last noisy frame.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.enabled = False
        self.value = cfg.default_volume
        self._reset_measurement()

    def _reset_measurement(self):
        self._anchor_spread = None
        self._anchor_value = self.value
        self._samples: deque = deque()
        self._previous = None

    def set_enabled(self, enabled: bool, current_volume: Optional[float] = None):
        if current_volume is not None:
            self.value = max(0.0, min(1.0, current_volume))
        self.enabled = enabled
        self._reset_measurement()

    def hand_lost(self):
        self.set_enabled(False)

    def update(self, hand_landmarks, now: float, width: int, height: int) -> float:
        if not self.enabled:
            return self.value
        lm = hand_landmarks.landmark
        if any(not math.isfinite(p.x) or not math.isfinite(p.y)
               or not (0.02 < p.x < 0.98 and 0.02 < p.y < 0.98) for p in lm):
            self._reset_measurement()
            return self.value
        points = np.array([(p.x, p.y * height / width) for p in lm])
        palm = float(np.linalg.norm(points[5] - points[17]))
        if palm < 0.02:
            self._reset_measurement()
            return self.value
        spread = float(np.linalg.norm(points[4] - points[8])) / palm
        previous = self._previous
        sample = (now, points[0], palm)
        self._previous = sample
        if previous is not None:
            dt = now - previous[0]
            if (dt <= 0 or dt > self.cfg.tracking_grace_s
                    or float(np.linalg.norm(points[0] - previous[1])) / dt > self.cfg.volume_motion_limit
                    or abs(palm - previous[2]) / (previous[2] * dt) > 1.2):
                self._reset_measurement()
                self._previous = sample
                return self.value
        if self._anchor_spread is None:
            self._anchor_spread = spread
            self._anchor_value = self.value
        self._samples.append((now, spread))
        # Retain the sample immediately before the dwell window as well, so
        # irregular FPS does not make the required time span unreachable.
        while len(self._samples) > 1 and now - self._samples[1][0] >= self.cfg.volume_settle_s:
            self._samples.popleft()
        spreads = [s for _, s in self._samples]
        if (len(spreads) >= 3 and now - self._samples[0][0] >= self.cfg.volume_settle_s
                and max(spreads) - min(spreads) <= self.cfg.volume_stability_tolerance):
            settled_spread = float(np.median(spreads))
            value = self._anchor_value + (settled_spread - self._anchor_spread) / self.cfg.volume_spread_range
            self.value = max(0.0, min(1.0, value))
            if value <= 0 or value >= 1:
                self._anchor_value, self._anchor_spread = self.value, settled_spread
        return self.value


# ---------------------------------------------------------------------------
# Gesture / state engine
# ---------------------------------------------------------------------------
class GestureEngine:
    """
    Converts raw MediaPipe landmark streams into high-level gesture events.

    Settle a loop selection, then latch it for a horizontal stroke. Motion uses
    capture timestamps and image-width units, not pixels per processed frame.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._action_ts: float = -math.inf
        self._reset_hand()

        # Head nod
        self._head_y: deque = deque(maxlen=cfg.head_history_len)
        self._last_nod_ts: float = 0.0

    def _clear_stroke(self):
        self._motion: deque = deque()
        self._stroke_target = 0
        self._stroke_sign = 0
        self._stroke_fired = False
        self._still_since: Optional[float] = None

    def _reset_hand(self):
        self._last_seen: Optional[float] = None
        self._last_sample = None
        self._tracked = False
        self.tracking_held = False
        self.ema_velocity = 0.0
        self.raw_finger_count = 0
        self.finger_count = 0
        self.swipe_fingers = 0
        self._candidate_count = 0
        self._candidate_since: Optional[float] = None
        self._clear_stroke()

    def _settle_selection(self, now):
        if self.raw_finger_count != self._candidate_count or self._candidate_since is None:
            self._candidate_count = self.raw_finger_count
            self._candidate_since = now
        if now - self._candidate_since >= self.cfg.selection_settle_s:
            self.finger_count = self._candidate_count

    def update_hand(
        self,
        hand_landmarks,
        handedness_label: str,
        frame_width: int,
        current_time: float,
        frame_height: int = 720,
    ) -> Tuple[int, float]:
        """
        Process one hand frame.
        Returns (finger_count, ema_velocity_x).
        Positive velocity → rightward swipe; negative → leftward.
        """
        if self._last_seen is not None and (
            current_time <= self._last_seen
            or current_time - self._last_seen > self.cfg.tracking_grace_s
        ):
            self._reset_hand()
        self.raw_finger_count = count_fingers(
            hand_landmarks, handedness_label, frame_width, frame_height)
        lm = hand_landmarks.landmark
        sample = (current_time, lm[0].x, lm[0].y * frame_height / frame_width)
        previous = self._last_sample if self._tracked else None
        self._tracked = True
        self.tracking_held = False
        self._last_seen = current_time
        self._last_sample = sample
        if previous is None:
            # After a dropout keep the selection briefly, but do not mistake
            # the unobserved jump to a re-detected wrist for a swipe.
            self.ema_velocity = 0.0
            self._clear_stroke()
            self._candidate_since = None
            self._settle_selection(current_time)
        else:
            dt = current_time - previous[0]
            raw_velocity = (sample[1] - previous[1]) / dt
            alpha = -math.expm1(-dt / self.cfg.velocity_smoothing_s)
            self.ema_velocity += alpha * (raw_velocity - self.ema_velocity)
            sign = (1 if self.ema_velocity > self.cfg.motion_release_speed else
                    -1 if self.ema_velocity < -self.cfg.motion_release_speed else 0)
            if sign:
                self._still_since = None
                if sign != self._stroke_sign:
                    self._clear_stroke()
                    self._stroke_sign = sign
                    self._stroke_target = self.finger_count
                    self._motion.append(previous)
                self._motion.append(sample)
                while len(self._motion) > 1 and current_time - self._motion[0][0] > self.cfg.swipe_window_s:
                    self._motion.popleft()
            else:
                if self._still_since is None:
                    self._still_since = current_time
                if current_time - self._still_since >= self.cfg.motion_release_s:
                    self._clear_stroke()
                if not self._stroke_sign:
                    self._settle_selection(current_time)

        return self.finger_count, self.ema_velocity

    def hand_lost(self, current_time: float):
        self._tracked = False
        self.raw_finger_count = 0
        self.ema_velocity = 0.0
        self._candidate_since = None
        self._clear_stroke()
        if self._last_seen is None or current_time - self._last_seen > self.cfg.tracking_grace_s:
            self._reset_hand()
        else:
            self.tracking_held = self.finger_count > 0

    def cooldown_ok(self, current_time: float) -> bool:
        return current_time - self._action_ts > self.cfg.action_cooldown

    def record_action(self, current_time: float):
        self._action_ts = current_time

    def detect_swipe(self, current_time: float) -> Optional[str]:
        """
        Return "right" / "left" when a swipe fires on this frame, else None.

        A settled target stays fixed through a stroke even if finger landmarks
        flicker. Require observed horizontal travel, speed and several samples.
        Emit once per stroke; a missing hand can never emit a command.
        """
        if (not self._tracked or self._stroke_target <= 0 or self._stroke_fired
                or not self.cooldown_ok(current_time) or len(self._motion) < 3):
            return None
        first, last = self._motion[0], self._motion[-1]
        dx, dy = last[1] - first[1], last[2] - first[2]
        steps = [b[1] - a[1] for a, b in zip(self._motion, list(self._motion)[1:])]
        travel = sum(abs(step) for step in steps)
        if (last[0] - first[0] < self.cfg.swipe_min_duration_s
                or abs(dx) < self.cfg.swipe_min_distance
                or abs(self.ema_velocity) < self.cfg.swipe_velocity_threshold
                or dx * self._stroke_sign <= 0
                or abs(dx) < 1.3 * abs(dy)
                or abs(dx) < 0.8 * travel
                or sum(step * self._stroke_sign > 0.002 for step in steps) < 2):
            return None
        direction = "right" if dx > 0 else "left"
        self.swipe_fingers = self._stroke_target
        self._stroke_fired = True
        self.record_action(current_time)
        return direction

    def check_nod(self, face_landmarks, current_time: float) -> Optional[float]:
        """
        Detect a downward head nod via nose-tip Y velocity reversal.
        Returns nod intensity in [0, 1] if detected, else None.
        """
        nose_y = face_landmarks[1].y
        self._head_y.append(nose_y)
        if len(self._head_y) < self._head_y.maxlen:
            return None

        vel   = self._head_y[-1] - self._head_y[-2]
        p_vel = self._head_y[-2] - self._head_y[-3]
        amp   = max(self._head_y) - min(self._head_y)

        if (
            p_vel <= 0 < vel
            and amp > self.cfg.nod_amplitude_threshold
            and current_time - self._last_nod_ts > self.cfg.nod_cooldown
        ):
            self._last_nod_ts = current_time
            return min(1.0, amp / 0.08) * 0.8
        return None


# ---------------------------------------------------------------------------
# Animation manager
# ---------------------------------------------------------------------------
class AnimationManager:
    """Priority-aware, type-aware animation queue."""

    def __init__(self):
        self._anims: List[dict] = []

    def add(self, anim_type: str, now: Optional[float] = None, **kwargs):
        entry = {
            "type":       anim_type,
            "text":       None,
            "start_time": time.time() if now is None else now,
            "duration":   1.2,
            "priority":   99,
            "color":      (255, 255, 255),
            "intensity":  1.0,
            "direction":  None,
        }
        entry.update(kwargs)

        # Only one integrated_hud at a time (lower priority wins = more important)
        if anim_type == "integrated_hud":
            existing = next(
                (a for a in self._anims if a["type"] == "integrated_hud"), None
            )
            if existing and entry["priority"] >= existing["priority"]:
                return
            self._anims = [a for a in self._anims if a["type"] != "integrated_hud"]

        self._anims.append(entry)

    def tick(self, current_time: float) -> List[dict]:
        """Remove expired animations and return the live list."""
        self._anims = [
            a for a in self._anims
            if current_time - a["start_time"] < a["duration"]
        ]
        return self._anims


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------
class Renderer:
    """All OpenCV drawing logic, stateless per call."""

    _HELP_LINES = [
        ("GESTURE CONTROLS",                  False),
        ("  Show 1–5 fingers  target loop",   True),
        ("  Swipe RIGHT        queue loop",    True),
        ("  Swipe LEFT         stop loop",     True),
        ("  V then pinch       adjust volume", True),
        ("",                                   True),
        ("KEYBOARD",                           False),
        ("  1 / 2   Artistic / Reality mode", True),
        ("  p / g / f   Piano / Guitar / Funk",True),
        ("  s  screenshot   h  help   q  quit",True),
        ("  v  adjust / lock volume",          True),
    ]

    def __init__(self, cfg: Config):
        self.cfg = cfg

    # ------------------------------------------------------------------
    def draw_dashboard(
        self,
        canvas: np.ndarray,
        loop_states: List[dict],
        beat_progress: float,
        current_time: float,
        set_name: str,
        master_volume: float,
    ) -> np.ndarray:
        h, w = canvas.shape[:2]
        dy = h - self.cfg.dashboard_height
        ov = np.zeros_like(canvas, dtype=np.uint8)
        cv2.rectangle(ov, (0, dy), (w, h), (10, 10, 10), -1)

        n = len(LOOP_COLORS)
        for i, color in enumerate(LOOP_COLORS):
            xc = (w // (n + 1)) * (i + 1)
            yc = dy + self.cfg.dashboard_height // 2
            r  = int(self.cfg.dashboard_height * 0.35)
            status = loop_states[i]["status"]

            if status == "locked":
                end = 270 + int(beat_progress * 360)
                cv2.ellipse(ov, (xc, yc), (r, r), 0, 270, end, color, -1)
                # Beat-sync glow pulse
                pulse = abs(math.sin(beat_progress * math.pi))
                glow  = adjust_brightness(color, 1.0 + 0.45 * pulse)
                cv2.circle(ov, (xc, yc), r, glow, 3)
            else:
                cv2.circle(ov, (xc, yc), r, adjust_brightness(color, 0.45), 2)

            if status in ("queued", "stopping") and int(current_time * 5) % 2:
                cv2.circle(ov, (xc, yc), 10, (255, 255, 255), -1)

            # Loop number
            cv2.putText(ov, str(i + 1), (xc - 7, yc + 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (210, 210, 210), 2)

        # Sound-set label
        cv2.putText(ov, set_name, (w - 270, dy + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (180, 205, 255), 2)

        # Volume bar
        bx, by = w - 38, dy + 14
        bh = self.cfg.dashboard_height - 28
        cv2.rectangle(ov, (bx, by), (bx + 14, by + bh), (55, 55, 55), -1)
        filled = int(bh * master_volume)
        cv2.rectangle(ov, (bx, by + bh - filled), (bx + 14, by + bh),
                      (90, 210, 90), -1)
        cv2.putText(ov, "VOL", (bx - 3, by + bh + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (150, 150, 150), 1)

        return cv2.addWeighted(ov, self.cfg.dashboard_alpha, canvas, 1, 0)

    # ------------------------------------------------------------------
    def draw_hand_skeleton(
        self, canvas: np.ndarray, hand_lm, fw: int, fh: int
    ) -> None:
        for i, lm in enumerate(hand_lm.landmark):
            cv2.circle(canvas, landmark_px(lm, fw, fh), 5,
                       ARTISTIC_PALETTE[i % len(ARTISTIC_PALETTE)], -1)
        for con in mp.solutions.hands.HAND_CONNECTIONS:
            p1 = landmark_px(hand_lm.landmark[con[0]], fw, fh)
            p2 = landmark_px(hand_lm.landmark[con[1]], fw, fh)
            cv2.line(canvas, p1, p2,
                     ARTISTIC_PALETTE[con[0] % len(ARTISTIC_PALETTE)], 2)

    def draw_face_mesh(
        self, canvas: np.ndarray, face_lm, fw: int, fh: int
    ) -> None:
        for i, con in enumerate(mp.solutions.face_mesh.FACEMESH_TESSELATION):
            p1 = landmark_px(face_lm.landmark[con[0]], fw, fh)
            p2 = landmark_px(face_lm.landmark[con[1]], fw, fh)
            cv2.line(canvas, p1, p2,
                     ARTISTIC_PALETTE[i % len(ARTISTIC_PALETTE)], 1)

    # ------------------------------------------------------------------
    def draw_target_indicator(
        self, canvas: np.ndarray, finger_count: int, fw: int,
        tracking_held: bool = False,
        volume_adjusting: bool = False, volume: float = 0.8,
    ) -> None:
        if volume_adjusting:
            title = f"VOLUME: {volume:.0%}"
            hint = "Adjust pinch, hold briefly to set. V to lock."
            for text, y, scale in ((title, 50, 0.9), (hint, 75, 0.55)):
                (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
                cv2.putText(canvas, text, ((fw - tw) // 2, y),
                            cv2.FONT_HERSHEY_SIMPLEX, scale, (100, 230, 230), 2)
            return
        if finger_count <= 0:
            cv2.putText(canvas, "Hold 1-5 fingers still to select a loop",
                        (max(12, fw // 2 - 230), 50), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (210, 210, 210), 2)
            cv2.putText(canvas, "V: adjust / lock volume",
                        (max(12, fw // 2 - 100), 75), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (210, 210, 210), 1)
            return
        text  = f"TARGET: LOOP {finger_count}"
        color = LOOP_COLORS[finger_count - 1]
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 1.0, 2)
        cv2.putText(canvas, text, ((fw - tw) // 2, 50),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, color, 2)
        hint = ("Tracking briefly lost - selection held" if tracking_held else
                "Swipe right to start / left to stop | V: volume")
        (hw, _), _ = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.putText(canvas, hint, ((fw - hw) // 2, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (210, 210, 210), 1)

    def draw_fps(self, canvas: np.ndarray, fps: float) -> None:
        cv2.putText(canvas, f"FPS: {fps:.0f}", (14, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (160, 160, 160), 2)

    # ------------------------------------------------------------------
    def draw_help_overlay(self, canvas: np.ndarray) -> np.ndarray:
        h, w = canvas.shape[:2]
        dark = canvas.copy()
        cv2.rectangle(dark, (18, 18), (w - 18, h - 18), (0, 0, 0), -1)
        canvas = cv2.addWeighted(dark, 0.72, canvas, 0.28, 0)
        y = 72
        for text, is_detail in self._HELP_LINES:
            color = (255, 200, 0) if not is_detail else (215, 215, 215)
            scale = 0.68 if is_detail else 0.88
            thick = 1 if is_detail else 2
            cv2.putText(canvas, text, (52, y),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)
            y += 40 if is_detail else 52
        return canvas

    # ------------------------------------------------------------------
    def draw_animations(
        self,
        canvas: np.ndarray,
        anims: List[dict],
        active_colors: List[Tuple[int, int, int]],
        visual_mode: str,
        fw: int,
        fh: int,
        now: float,
    ) -> np.ndarray:
        ov = np.zeros_like(canvas, dtype=np.uint8)

        for anim in anims:
            progress = (now - anim["start_time"]) / anim["duration"]
            atype    = anim["type"]

            if atype == "integrated_hud":
                self._draw_hud_panel(ov, anim, progress, fw, fh)

            elif atype == "color_splash":
                alpha  = math.sin(progress * math.pi) * 0.65
                splash = np.full(canvas.shape, anim["color"], dtype=np.uint8)
                ov     = cv2.addWeighted(ov, 1.0, splash, alpha, 0)

            elif atype == "set_indicator":
                alpha = math.sin(progress * math.pi) * 0.85
                fs, tk = 2.5, 4
                (tw, th), _ = cv2.getTextSize(
                    anim["text"], cv2.FONT_HERSHEY_DUPLEX, fs, tk
                )
                tmp = ov.copy()
                cv2.putText(tmp, anim["text"],
                            ((fw - tw) // 2, (fh + th) // 2),
                            cv2.FONT_HERSHEY_DUPLEX, fs, anim["color"], tk)
                ov = cv2.addWeighted(ov, 1.0, tmp, alpha, 0)

            elif atype == "rhythmic_flash":
                if active_colors:
                    col = random.choice(active_colors)
                    ov  = cv2.addWeighted(
                        ov, 1.0,
                        np.full(canvas.shape, col, dtype=np.uint8), 0.85, 0
                    )
                else:
                    alpha = math.sin(progress * math.pi) * 0.45
                    ov    = cv2.addWeighted(
                        ov, 1.0,
                        np.full(canvas.shape, (180, 180, 180), dtype=np.uint8),
                        alpha, 0,
                    )

        if visual_mode == "artistic":
            return cv2.add(canvas, ov)
        return cv2.addWeighted(canvas, 1.0, ov, 0.9, 0)

    # ------------------------------------------------------------------
    def _draw_hud_panel(
        self, ov: np.ndarray, anim: dict, progress: float, fw: int, fh: int
    ) -> None:
        font, tk, fs = cv2.FONT_HERSHEY_DUPLEX, 3, 2.0
        (tw, th), _  = cv2.getTextSize(anim["text"], font, fs, tk)
        aw, ah, pad  = 60, 50, 40
        bw, bh       = tw + aw + pad * 3, th + pad * 2

        # Cubic ease-in / ease-out slide
        if progress < 0.2:
            p  = 1 - (1 - progress / 0.2) ** 3
            x0 = -bw if anim["direction"] == "right" else fw
            cx = int(x0 + ((fw - bw) // 2 - x0) * p)
        elif progress > 0.8:
            p  = ((progress - 0.8) / 0.2) ** 3
            x0 = (fw - bw) // 2
            xt = fw if anim["direction"] == "right" else -bw
            cx = int(x0 + (xt - x0) * p)
        else:
            cx = (fw - bw) // 2
        cy = (fh - bh) // 2

        panel = ov.copy()
        cv2.rectangle(panel, (cx, cy), (cx + bw, cy + bh),
                      adjust_brightness(anim["color"], 0.55), -1)

        ty = cy + pad + th
        if anim["direction"] == "right":
            tx  = cx + pad
            arr = np.array([
                [cx + bw - pad - aw, cy + (bh - ah) // 2],
                [cx + bw - pad,      cy + bh // 2],
                [cx + bw - pad - aw, cy + (bh + ah) // 2],
            ])
        else:
            tx  = cx + pad * 2 + aw
            arr = np.array([
                [cx + pad + aw, cy + (bh - ah) // 2],
                [cx + pad,      cy + bh // 2],
                [cx + pad + aw, cy + (bh + ah) // 2],
            ])

        cv2.putText(panel, anim["text"], (tx, ty), font, fs, (255, 255, 255), tk)
        cv2.fillPoly(panel, [arr], (255, 255, 255))
        np.copyto(ov, cv2.addWeighted(ov, 1.0, panel, 0.9, 0))


# ---------------------------------------------------------------------------
# Frame sources, recording and per-frame logging (measurement support)
# ---------------------------------------------------------------------------
def timestamps_path(video_path: Path) -> Path:
    """Sidecar CSV that stores each recorded frame's capture time."""
    return video_path.with_suffix(".frames.csv")


class CameraSource:
    """Live webcam.  Timestamps are wall-clock seconds taken right after read."""

    live = True

    def __init__(self, cfg: Config):
        cap = cv2.VideoCapture(cfg.camera_index, cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
            cap = cv2.VideoCapture(cfg.camera_index)
        if not cap.isOpened():
            log.error("Cannot open camera index %d.", cfg.camera_index)
            sys.exit(1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  cfg.frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.frame_height)
        time.sleep(0.8)   # allow camera to warm up
        self.cap = cap
        self.nominal_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    def read(self) -> Tuple[bool, Optional[np.ndarray], float]:
        ret, frame = self.cap.read()
        return ret, frame, time.time()

    def release(self):
        self.cap.release()


class VideoFileSource:
    """
    Replays a recorded clip frame by frame (no frames are dropped).
    Timestamps come from the clip's .frames.csv sidecar when present, so
    time-based logic (cooldowns, loop boundaries) sees the recorded timing;
    otherwise they are derived from the container's frame rate.
    """

    live = False

    def __init__(self, path: Path):
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Cannot open video {path}")
        self.nominal_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._idx = 0
        self._times: List[float] = []
        ts_file = timestamps_path(path)
        if ts_file.exists():
            with open(ts_file, newline="") as fh:
                self._times = [float(row["t"]) for row in csv.DictReader(fh)]
        else:
            log.warning("No %s — using %.1f fps for timestamps.",
                        ts_file.name, self.nominal_fps)

    def read(self) -> Tuple[bool, Optional[np.ndarray], float]:
        ret, frame = self.cap.read()
        if self._idx < len(self._times):
            t = self._times[self._idx]
        else:
            t = self._idx / self.nominal_fps
        self._idx += 1
        return ret, frame, t

    def release(self):
        self.cap.release()


class ClipRecorder:
    """Writes raw (un-mirrored) camera frames plus a capture-time sidecar."""

    def __init__(self, path: Path, fps: float, size: Tuple[int, int]):
        path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = "MJPG" if path.suffix.lower() == ".avi" else "mp4v"
        self.writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*fourcc), fps, size
        )
        if not self.writer.isOpened():
            raise RuntimeError(f"Cannot write video {path}")
        self._ts_fh = open(timestamps_path(path), "w", newline="")
        self._ts = csv.writer(self._ts_fh)
        self._ts.writerow(["frame", "t"])
        self._idx = 0
        log.info("Recording to %s", path)

    def write(self, frame: np.ndarray, t: float):
        self.writer.write(frame)
        self._ts.writerow([self._idx, f"{t:.6f}"])
        self._idx += 1

    def close(self):
        self.writer.release()
        self._ts_fh.close()


class FrameLog:
    """
    One JSON object per processed frame (JSON Lines).  The first line is a
    header describing the run.  `perf_read` / `play` use time.perf_counter()
    and are only meaningful for live runs (software latency).
    """

    def __init__(self, path: Path, header: dict):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(path, "w")
        self.write({"header": header})

    def write(self, record: dict):
        self._fh.write(json.dumps(record) + "\n")

    def close(self):
        self._fh.close()


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------
class GestureMusicController:
    """
    Top-level controller.  Owns the main loop, wires together AudioEngine,
    GestureEngine, AnimationManager, and Renderer.
    """

    _SET_NAMES = ["HousePiano", "BluesGuitar", "FunkGuitar"]
    _SET_KEYS  = [ord("p"),     ord("g"),      ord("f")]

    def __init__(
        self,
        cfg: Config,
        demo_mode: bool = False,
        source=None,
        recorder: Optional[ClipRecorder] = None,
        frame_log: Optional[FrameLog] = None,
        headless: bool = False,
    ):
        self.cfg        = cfg
        self.demo_mode  = demo_mode
        self.visual_mode: str  = "reality"
        self.show_help: bool   = False
        self.recorder   = recorder
        self.frame_log  = frame_log
        self.headless   = headless

        # Sub-systems
        self.audio    = AudioEngine(cfg, demo_mode)
        self.volume_control = PinchVolumeControl(cfg)
        self.gesture  = GestureEngine(cfg)
        self.anim     = AnimationManager()
        self.renderer = Renderer(cfg)

        # Loop state machine: status ∈ {inactive, queued, locked, stopping}
        self.loop_states = [{"status": "inactive"} for _ in range(5)]
        self.quant_queue: List[dict] = []
        self.stop_queue:  List[dict] = []
        self.master_clock:  Optional[float] = None
        self.master_length: float = 0.0

        # Active sound set
        self._set_idx: int = 0
        self.audio.set_sound_set(self._SET_NAMES[0])

        # FPS tracking
        self._frame_times: deque = deque(maxlen=cfg.fps_smoothing)

        # MediaPipe models
        self._mp_hands = mp.solutions.hands.Hands(
            model_complexity=0, max_num_hands=1,
            min_detection_confidence=0.7, min_tracking_confidence=0.7,
        )
        self._mp_face = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5,
        )

        self.source = source if source is not None else CameraSource(cfg)

    # ------------------------------------------------------------------
    def _next_beat(self, now: float) -> float:
        if self.master_clock is None:
            return now
        cycles = math.ceil((now - self.master_clock) / self.master_length)
        return self.master_clock + cycles * self.master_length

    def _action_priority(self, idx: int, kind: str) -> int:
        return idx + (5 if kind == "stopping" else 0)

    # ------------------------------------------------------------------
    def _switch_set(self, name: str):
        if name == self._SET_NAMES[self._set_idx]:
            return
        log.info("Sound set → %s", name)
        self.audio.stop_all()
        for s in self.loop_states:
            s["status"] = "inactive"
        self.quant_queue.clear()
        self.stop_queue.clear()
        self.master_clock = None
        self._set_idx = self._SET_NAMES.index(name)
        self.audio.set_sound_set(name)
        self.audio.play_confirm()
        self.anim.add("set_indicator", text=name, duration=1.5,
                      color=(200, 220, 255))

    # ------------------------------------------------------------------
    def _process_queues(self, now: float) -> List[Tuple[int, float]]:
        """Start / stop due loops.  Returns [(loop index, perf time of play)]."""
        played: List[Tuple[int, float]] = []
        for item in self.quant_queue[:]:
            if now >= item["start_time"]:
                idx = item["index"]
                self.loop_states[idx]["status"] = "locked"
                self.audio.play_loop(idx)
                played.append((idx, time.perf_counter()))
                self.anim.add("color_splash", now=now,
                              color=LOOP_COLORS[idx], duration=0.8)
                self.quant_queue.remove(item)

        for item in self.stop_queue[:]:
            if now >= item["stop_time"]:
                idx = item["index"]
                self.loop_states[idx]["status"] = "inactive"
                self.audio.stop_loop(idx)
                self.stop_queue.remove(item)
        return played

    # ------------------------------------------------------------------
    def _handle_gesture(self, hand_results, fw: int, now: float,
                        fh: Optional[int] = None) -> dict:
        """Update gesture state and loops.  Returns a summary for the log."""
        info: dict = {"hand": False, "volume": round(self.audio.master_volume, 4),
                      "volume_adjusting": self.volume_control.enabled}
        if not hand_results.multi_hand_landmarks:
            if self.volume_control.enabled:
                self.volume_control.hand_lost()
                self.gesture._reset_hand()
            self.gesture.hand_lost(now)
            info.update(fingers=self.gesture.finger_count, raw_fingers=0,
                        tracking_held=self.gesture.tracking_held, volume_adjusting=False)
            return info

        hand_lm = hand_results.multi_hand_landmarks[0]
        label   = "Right"
        if hand_results.multi_handedness:
            label = hand_results.multi_handedness[0].classification[0].label

        finger_count, velocity = self.gesture.update_hand(
            hand_lm, label, fw, now, fh or self.cfg.frame_height)
        info.update(hand=True, handedness=label, fingers=finger_count,
                    raw_fingers=self.gesture.raw_finger_count,
                    tracking_held=False, vel=round(velocity, 4))

        if self.volume_control.enabled:
            self.audio.set_volume(self.volume_control.update(hand_lm, now, fw, fh or self.cfg.frame_height))
            self.gesture._clear_stroke()
            info["volume"] = round(self.audio.master_volume, 4)
            return info

        swipe = self.gesture.detect_swipe(now)
        if swipe is None:
            return info
        info["swipe"] = swipe
        finger_count = self.gesture.swipe_fingers
        info["fingers"] = finger_count
        idx = finger_count - 1

        if swipe == "right" and self.loop_states[idx]["status"] == "inactive":
            if self.master_clock is None:
                self.master_clock  = now
                self.master_length = self.audio.loop_length(idx)
                self.quant_queue.append({"index": idx, "start_time": now})
                info["action"] = "start"
            else:
                self.quant_queue.append(
                    {"index": idx, "start_time": self._next_beat(now)}
                )
                info["action"] = "queue"
            self.loop_states[idx]["status"] = "queued"
            self.anim.add(
                "integrated_hud",
                now=now,
                text=f"LOOP {idx + 1} QUEUED",
                direction="right",
                color=LOOP_COLORS[idx],
                priority=self._action_priority(idx, "queued"),
            )

        elif swipe == "left" and self.loop_states[idx]["status"] == "locked":
            self.stop_queue.append(
                {"index": idx, "stop_time": self._next_beat(now)}
            )
            self.loop_states[idx]["status"] = "stopping"
            info["action"] = "stop"
            self.anim.add(
                "integrated_hud",
                now=now,
                text=f"LOOP {idx + 1} STOPPING",
                direction="left",
                color=LOOP_COLORS[idx],
                priority=self._action_priority(idx, "stopping"),
            )
        return info

    # ------------------------------------------------------------------
    def _handle_face(self, face_results, now: float):
        if self.visual_mode != "artistic" or not face_results.multi_face_landmarks:
            return
        face_lm   = face_results.multi_face_landmarks[0].landmark
        intensity = self.gesture.check_nod(face_lm, now)
        if intensity is not None:
            self.anim.add("rhythmic_flash", now=now,
                          duration=0.15, intensity=intensity)

    # ------------------------------------------------------------------
    def _handle_key(self, key: int, canvas: np.ndarray) -> bool:
        """Return False to signal quit."""
        if key == ord("q"):
            return False
        elif key == ord("1"):
            self.visual_mode = "artistic"
            log.info("Visual mode → Artistic")
        elif key == ord("2"):
            self.visual_mode = "reality"
            log.info("Visual mode → Reality")
        elif key == ord("h"):
            self.show_help = not self.show_help
        elif key == ord("v"):
            self.volume_control.set_enabled(not self.volume_control.enabled, self.audio.master_volume)
            self.gesture._reset_hand()
            log.info("Volume %s at %.0f%%", "adjustment enabled" if self.volume_control.enabled else "locked",
                     self.audio.master_volume * 100)
        elif key == ord("s"):
            fname = f"screenshot_{int(time.time())}.png"
            cv2.imwrite(fname, canvas)
            log.info("Screenshot saved: %s", fname)
        elif key in self._SET_KEYS:
            self._switch_set(self._SET_NAMES[self._SET_KEYS.index(key)])
        return True

    # ------------------------------------------------------------------
    def run(self):
        log.info("Gesture Music Controller started  (demo=%s)", self.demo_mode)
        log.info("Visual: 1 = Artistic | 2 = Reality   Sound: p / g / f")
        log.info("Press h for controls, q to quit.")
        self.audio.play_startup()
        frame_idx = 0

        try:
            while True:
                t0  = time.perf_counter()
                ret, frame, now = self.source.read()
                if not ret:
                    if self.source.live:
                        log.warning("Frame read failed — camera disconnected?")
                    break
                t_read = time.perf_counter()

                if self.recorder:
                    self.recorder.write(frame, now)

                frame = cv2.flip(frame, 1)
                fh, fw = frame.shape[:2]

                # ── Core logic ──────────────────────────────────────────
                played = self._process_queues(now)

                rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                hand_res = self._mp_hands.process(rgb)
                face_res = self._mp_face.process(rgb)

                info = self._handle_gesture(hand_res, fw, now, fh)
                self._handle_face(face_res, now)

                # Auto-reset master clock when all loops go inactive
                if self.master_clock is not None and all(
                    s["status"] == "inactive" for s in self.loop_states
                ):
                    self.master_clock = None
                    self.quant_queue.clear()
                    self.stop_queue.clear()

                if self.frame_log:
                    rec = {"i": frame_idx, "t": round(now, 6), **info}
                    if played:
                        rec["played"] = [i + 1 for i, _ in played]
                    if self.source.live:
                        rec["perf_read"] = round(t_read, 6)
                        if played:
                            rec["perf_play"] = [round(p, 6) for _, p in played]
                    self.frame_log.write(rec)
                frame_idx += 1

                if self.headless:
                    continue

                canvas = (
                    np.zeros_like(frame) if self.visual_mode == "artistic"
                    else frame.copy()
                )

                # ── Rendering ───────────────────────────────────────────
                beat_prog = 0.0
                if self.master_clock and self.master_length > 0:
                    beat_prog = (
                        (now - self.master_clock) % self.master_length
                    ) / self.master_length

                canvas = self.renderer.draw_dashboard(
                    canvas, self.loop_states, beat_prog, now,
                    self._SET_NAMES[self._set_idx],
                    self.audio.master_volume,
                )

                if self.visual_mode == "artistic":
                    if hand_res.multi_hand_landmarks:
                        for hl in hand_res.multi_hand_landmarks:
                            self.renderer.draw_hand_skeleton(canvas, hl, fw, fh)
                    if face_res.multi_face_landmarks:
                        for fl in face_res.multi_face_landmarks:
                            self.renderer.draw_face_mesh(canvas, fl, fw, fh)

                self.renderer.draw_target_indicator(
                    canvas, self.gesture.finger_count, fw, self.gesture.tracking_held,
                    self.volume_control.enabled, self.audio.master_volume,
                )

                active_anims  = self.anim.tick(now)
                active_colors = [
                    LOOP_COLORS[i]
                    for i, s in enumerate(self.loop_states)
                    if s["status"] == "locked"
                ]
                canvas = self.renderer.draw_animations(
                    canvas, active_anims, active_colors,
                    self.visual_mode, fw, fh, now,
                )

                # FPS counter
                self._frame_times.append(time.perf_counter() - t0)
                fps = 1.0 / (sum(self._frame_times) / len(self._frame_times))
                self.renderer.draw_fps(canvas, fps)

                if self.show_help:
                    canvas = self.renderer.draw_help_overlay(canvas)

                cv2.imshow("Gesture Music Controller", canvas)
                key = cv2.waitKey(1) & 0xFF
                if key != 255 and not self._handle_key(key, canvas):
                    break

        finally:
            self._cleanup()

    # ------------------------------------------------------------------
    def _cleanup(self):
        log.info("Shutting down…")
        self.audio.stop_all()
        self.source.release()
        if self.recorder:
            self.recorder.close()
        if self.frame_log:
            self.frame_log.close()
        if not self.headless:
            cv2.destroyAllWindows()
        self._mp_hands.close()
        self._mp_face.close()
        self.audio.quit()
        log.info("Done.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Gesture Music Controller — hands-free webcam loop station"
    )
    p.add_argument("--camera",          type=int, default=0,  metavar="N",
                   help="Camera device index (default: 0)")
    p.add_argument("--width",           type=int, default=1280,
                   help="Capture width  (default: 1280)")
    p.add_argument("--height",          type=int, default=720,
                   help="Capture height (default: 720)")
    p.add_argument("--demo",            action="store_true",
                   help="Disable audio — no .wav files required")
    p.add_argument("--swipe-threshold", type=float, default=0.35, metavar="WIDTHS_PER_S",
                   help="Swipe speed in frame widths/second (default: 0.35; "
                        "R1 replaces the old px/frame units)")
    p.add_argument("--record",          type=Path, metavar="FILE",
                   help="Save raw camera frames to FILE (.mp4 or .avi) "
                        "plus a .frames.csv of capture times")
    p.add_argument("--input",           type=Path, metavar="FILE",
                   help="Replay a recorded clip instead of the camera "
                        "(headless unless --show; audio off)")
    p.add_argument("--show",            action="store_true",
                   help="With --input: also open the preview window")
    p.add_argument("--log",             type=Path, metavar="FILE",
                   help="Write one JSON line per processed frame to FILE")
    return p.parse_args()


def run_header(cfg: Config, source_kind: str, input_path: Optional[Path]) -> dict:
    """Describe a run for the first line of a frame log."""
    return {
        "source": source_kind,
        "input": str(input_path) if input_path else None,
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "swipe_velocity_threshold": cfg.swipe_velocity_threshold,
        "velocity_units": "frame_widths_per_second",
        "swipe_min_distance": cfg.swipe_min_distance,
        "swipe_window_s": cfg.swipe_window_s,
        "swipe_min_duration_s": cfg.swipe_min_duration_s,
        "velocity_smoothing_s": cfg.velocity_smoothing_s,
        "selection_settle_s": cfg.selection_settle_s,
        "tracking_grace_s": cfg.tracking_grace_s,
        "motion_release_speed": cfg.motion_release_speed,
        "motion_release_s": cfg.motion_release_s,
        "action_cooldown": cfg.action_cooldown,
        "volume_mode": "explicit_v_toggle",
        "volume_settle_s": cfg.volume_settle_s,
        "volume_spread_range": cfg.volume_spread_range,
        "volume_stability_tolerance": cfg.volume_stability_tolerance,
        "volume_motion_limit": cfg.volume_motion_limit,
        "mediapipe": mp.__version__,
        "opencv": cv2.__version__,
    }


if __name__ == "__main__":
    args = _parse_args()
    if not math.isfinite(args.swipe_threshold) or args.swipe_threshold <= 0:
        sys.exit("--swipe-threshold must be a positive speed in frame widths/second.")
    cfg  = Config(
        camera_index             = args.camera,
        frame_width              = args.width,
        frame_height             = args.height,
        swipe_velocity_threshold = args.swipe_threshold,
    )
    if args.input and args.record:
        sys.exit("--record needs the live camera; it cannot be used with --input.")

    if args.input:
        source = VideoFileSource(args.input)
    else:
        source = CameraSource(cfg)

    recorder = None
    if args.record:
        ok, probe, _ = source.read()   # first frame fixes the clip size
        if not ok:
            sys.exit("Camera returned no frame; nothing to record.")
        h, w = probe.shape[:2]
        recorder = ClipRecorder(args.record, source.nominal_fps, (w, h))

    frame_log = None
    if args.log:
        frame_log = FrameLog(
            args.log,
            run_header(cfg, "replay" if args.input else "live", args.input),
        )

    GestureMusicController(
        cfg,
        demo_mode = args.demo or bool(args.input),
        source    = source,
        recorder  = recorder,
        frame_log = frame_log,
        headless  = bool(args.input) and not args.show,
    ).run()
