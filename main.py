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
Pinch (thumb + index)    →  adjust master volume (spread = louder)

Keyboard:
  1 / 2       Artistic / Reality visual mode
  p / g / f   HousePiano / BluesGuitar / FunkGuitar sound set
  s           Save screenshot
  h           Toggle help overlay
  q           Quit

Usage:
  python main.py                         # default camera, 1280×720
  python main.py --camera 1              # second webcam
  python main.py --demo                  # no audio required (visual only)
  python main.py --width 640 --height 480 --swipe-threshold 35

Measurement (see docs/MEASUREMENT.md):
  python main.py --demo --record recordings/clip.mp4     # save raw camera frames
  python main.py --input recordings/clip.mp4 --log out.jsonl   # headless replay
  python main.py --log live.jsonl                        # per-frame log, live
"""

import argparse
import csv
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
    swipe_velocity_threshold: int = 45   # px/frame to register a swipe
    action_cooldown: float = 0.8         # minimum seconds between actions
    ema_alpha: float = 0.4               # EMA weight for velocity smoothing

    # Head-nod
    nod_cooldown: float = 0.15
    nod_amplitude_threshold: float = 0.004
    head_history_len: int = 4

    # Volume control via pinch gesture
    default_volume: float = 0.8
    pinch_min_dist: float = 0.04         # normalised distance → vol = 0
    pinch_max_dist: float = 0.22         # normalised distance → vol = 1

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
# Finger counting — handedness-aware (correct thumb detection on both hands)
# ---------------------------------------------------------------------------
_TIP_IDS = [4, 8, 12, 16, 20]


def count_fingers(hand_landmarks, handedness_label: str) -> int:
    """Return 0-5 extended fingers, handling mirrored thumb for left hand."""
    lm = hand_landmarks.landmark
    fingers: List[int] = []
    # Thumb: check x-direction relative to hand orientation
    if handedness_label == "Right":
        fingers.append(1 if lm[_TIP_IDS[0]].x > lm[_TIP_IDS[0] - 1].x else 0)
    else:
        fingers.append(1 if lm[_TIP_IDS[0]].x < lm[_TIP_IDS[0] - 1].x else 0)
    # Four fingers: tip y < knuckle-2 y  (smaller y = higher in frame)
    for i in range(1, 5):
        fingers.append(
            1 if lm[_TIP_IDS[i]].y < lm[_TIP_IDS[i] - 2].y else 0
        )
    return sum(fingers)


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
# Gesture / state engine
# ---------------------------------------------------------------------------
class GestureEngine:
    """
    Converts raw MediaPipe landmark streams into high-level gesture events.

    Swipe detection uses an exponential moving average (EMA) over raw wrist
    displacement to suppress high-frequency noise while preserving intent.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._last_wrist_x: Optional[int] = None
        self.ema_velocity: float = 0.0
        self._action_ts: float = 0.0
        self.finger_count: int = 0
        self.pinch_volume: float = cfg.default_volume

        # Head nod
        self._head_y: deque = deque(maxlen=cfg.head_history_len)
        self._last_nod_ts: float = 0.0

    def update_hand(
        self,
        hand_landmarks,
        handedness_label: str,
        frame_width: int,
    ) -> Tuple[int, float]:
        """
        Process one hand frame.
        Returns (finger_count, ema_velocity_x).
        Positive velocity → rightward swipe; negative → leftward.
        """
        self.finger_count = count_fingers(hand_landmarks, handedness_label)

        wrist_x = int(
            hand_landmarks.landmark[
                mp.solutions.hands.HandLandmark.WRIST
            ].x
            * frame_width
        )
        if self._last_wrist_x is not None:
            raw_vel = wrist_x - self._last_wrist_x
            self.ema_velocity = (
                self.cfg.ema_alpha * raw_vel
                + (1 - self.cfg.ema_alpha) * self.ema_velocity
            )
        else:
            self.ema_velocity = 0.0
        self._last_wrist_x = wrist_x

        # Pinch volume: distance between thumb tip (4) and index tip (8)
        lm = hand_landmarks.landmark
        pinch_d = dist_norm(lm[4], lm[8])
        t = (pinch_d - self.cfg.pinch_min_dist) / (
            self.cfg.pinch_max_dist - self.cfg.pinch_min_dist
        )
        self.pinch_volume = max(0.0, min(1.0, t))

        return self.finger_count, self.ema_velocity

    def hand_lost(self):
        self._last_wrist_x = None
        self.ema_velocity = 0.0
        self.finger_count = 0

    def cooldown_ok(self, current_time: float) -> bool:
        return current_time - self._action_ts > self.cfg.action_cooldown

    def record_action(self, current_time: float):
        self._action_ts = current_time

    def detect_swipe(self, current_time: float) -> Optional[str]:
        """
        Return "right" / "left" when a swipe fires on this frame, else None.

        A swipe needs at least one extended finger, |EMA velocity| above the
        threshold, and the cooldown to have elapsed.  Every swipe that fires
        restarts the cooldown, whether or not it changes a loop, so the same
        motion cannot fire twice.
        """
        if self.finger_count <= 0 or not self.cooldown_ok(current_time):
            return None
        thr = self.cfg.swipe_velocity_threshold
        if self.ema_velocity > thr:
            direction = "right"
        elif self.ema_velocity < -thr:
            direction = "left"
        else:
            return None
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
        ("  Pinch thumb+index  volume",        True),
        ("",                                   True),
        ("KEYBOARD",                           False),
        ("  1 / 2   Artistic / Reality mode", True),
        ("  p / g / f   Piano / Guitar / Funk",True),
        ("  s  screenshot   h  help   q  quit",True),
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
        self, canvas: np.ndarray, finger_count: int, fw: int
    ) -> None:
        if finger_count <= 0:
            return
        text  = f"TARGET: LOOP {finger_count}"
        color = LOOP_COLORS[finger_count - 1]
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 1.0, 2)
        cv2.putText(canvas, text, ((fw - tw) // 2, 50),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, color, 2)

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
    def _handle_gesture(self, hand_results, fw: int, now: float) -> dict:
        """Update gesture state and loops.  Returns a summary for the log."""
        info: dict = {"hand": False}
        if not hand_results.multi_hand_landmarks:
            self.gesture.hand_lost()
            return info

        hand_lm = hand_results.multi_hand_landmarks[0]
        label   = "Right"
        if hand_results.multi_handedness:
            label = hand_results.multi_handedness[0].classification[0].label

        finger_count, velocity = self.gesture.update_hand(hand_lm, label, fw)
        self.audio.set_volume(self.gesture.pinch_volume)
        info.update(hand=True, handedness=label, fingers=finger_count,
                    vel=round(velocity, 2))

        swipe = self.gesture.detect_swipe(now)
        if swipe is None:
            return info
        info["swipe"] = swipe
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

                info = self._handle_gesture(hand_res, fw, now)
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
                    canvas, self.gesture.finger_count, fw
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
    p.add_argument("--swipe-threshold", type=int, default=45,  metavar="PX",
                   help="Swipe velocity threshold in px/frame (default: 45)")
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
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "swipe_velocity_threshold": cfg.swipe_velocity_threshold,
        "action_cooldown": cfg.action_cooldown,
        "ema_alpha": cfg.ema_alpha,
        "mediapipe": mp.__version__,
        "opencv": cv2.__version__,
    }


if __name__ == "__main__":
    args = _parse_args()
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
