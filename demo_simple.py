#!/usr/bin/env python3
"""
demo_simple.py — Minimal hand-tracking demo (no audio required).

Displays real-time hand landmark detection and finger counting from a webcam.
Use this to verify your MediaPipe and OpenCV installation before running the
full controller.

Usage:
    python demo_simple.py              # default camera (index 0)
    python demo_simple.py --camera 1   # second camera

Press 'q' to quit.
"""

import argparse
import time

import cv2
import mediapipe as mp
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_TIP_IDS     = [4, 8, 12, 16, 20]
_FINGER_NAMES = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
_TIP_COLORS  = [
    (80,  80,  255),   # thumb  – blue
    (80,  200, 80),    # index  – green
    (255, 180, 50),    # middle – amber
    (200, 80,  200),   # ring   – purple
    (50,  200, 220),   # pinky  – cyan
]
_INACTIVE_COLOR = (70, 70, 70)


# ---------------------------------------------------------------------------
# Handedness-aware finger counting
# ---------------------------------------------------------------------------
def count_fingers(hand_lm, handedness_label: str):
    """
    Return a list of 5 ints (0 = folded, 1 = extended), one per finger.
    Handles the mirrored thumb for left-hand detection correctly.
    """
    lm     = hand_lm.landmark
    result = []
    # Thumb
    if handedness_label == "Right":
        result.append(1 if lm[_TIP_IDS[0]].x > lm[_TIP_IDS[0] - 1].x else 0)
    else:
        result.append(1 if lm[_TIP_IDS[0]].x < lm[_TIP_IDS[0] - 1].x else 0)
    # Other four fingers: tip y < two-joint-below y
    for i in range(1, 5):
        result.append(
            1 if lm[_TIP_IDS[i]].y < lm[_TIP_IDS[i] - 2].y else 0
        )
    return result


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------
def draw_finger_tips(frame, hand_lm, fingers, fw, fh):
    for i, (tip_id, extended) in enumerate(zip(_TIP_IDS, fingers)):
        lm   = hand_lm.landmark[tip_id]
        px   = int(lm.x * fw)
        py   = int(lm.y * fh)
        col  = _TIP_COLORS[i] if extended else _INACTIVE_COLOR
        cv2.circle(frame, (px, py), 14, col, -1)
        cv2.circle(frame, (px, py), 14, (255, 255, 255), 2)


def draw_wrist_label(frame, hand_lm, label, total, fw, fh):
    wrist = hand_lm.landmark[0]
    wx    = int(wrist.x * fw)
    wy    = int(wrist.y * fh)
    word  = "finger" if total == 1 else "fingers"
    text  = f"{label}: {total} {word}"
    cv2.putText(frame, text, (wx - 65, wy + 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)


def draw_legend(frame, fingers, fw):
    """Small horizontal legend showing each finger's name and state."""
    panel_h = 38
    panel   = np.zeros((panel_h, fw, 3), dtype=np.uint8)
    n       = len(_FINGER_NAMES)
    col_w   = fw // n
    for i, name in enumerate(_FINGER_NAMES):
        col   = _TIP_COLORS[i] if fingers[i] else _INACTIVE_COLOR
        x_ctr = col_w * i + col_w // 2
        cv2.circle(panel, (x_ctr, panel_h // 2), 8, col, -1)
        tw, _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[:2]
        cv2.putText(panel, name, (x_ctr - tw[0] // 2, panel_h - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    return np.vstack([panel, frame])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Minimal hand-tracking demo")
    parser.add_argument("--camera", type=int, default=0,
                        metavar="N", help="Camera device index (default: 0)")
    args = parser.parse_args()

    # MediaPipe setup — detect up to 2 hands simultaneously
    mp_hands  = mp.solutions.hands
    hands     = mp_hands.Hands(
        model_complexity       = 0,
        max_num_hands          = 2,
        min_detection_confidence = 0.7,
        min_tracking_confidence  = 0.7,
    )
    draw_utils = mp.solutions.drawing_utils
    hand_style = draw_utils.DrawingSpec(
        color=(60, 120, 10), thickness=2, circle_radius=3)
    conn_style = draw_utils.DrawingSpec(
        color=(80, 230, 120), thickness=2, circle_radius=2)

    # Camera
    cap = cv2.VideoCapture(args.camera, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    time.sleep(0.8)

    if not cap.isOpened():
        print("ERROR: Cannot open camera.")
        print("  1. Check camera index (try --camera 1)")
        print("  2. Ensure camera permission is granted in System Settings")
        return

    print("Hand Tracking Demo  —  press q to quit")

    prev_t = time.perf_counter()
    # Default fingers list when no hand is visible
    no_hand_fingers = [0, 0, 0, 0, 0]

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame read failed — camera may have disconnected.")
            break

        frame = cv2.flip(frame, 1)
        fh, fw = frame.shape[:2]
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results  = hands.process(rgb)
        all_fins = no_hand_fingers  # used for legend; overwritten if hand found

        if results.multi_hand_landmarks:
            for hand_lm, handedness in zip(
                results.multi_hand_landmarks,
                results.multi_handedness or [],
            ):
                label   = handedness.classification[0].label
                fingers = count_fingers(hand_lm, label)
                total   = sum(fingers)
                all_fins = fingers  # last hand wins for legend

                # MediaPipe skeleton
                draw_utils.draw_landmarks(frame, hand_lm,
                                          mp_hands.HAND_CONNECTIONS,
                                          hand_style, conn_style)
                # Highlighted tip dots
                draw_finger_tips(frame, hand_lm, fingers, fw, fh)
                # Wrist count label
                draw_wrist_label(frame, hand_lm, label, total, fw, fh)
        else:
            # Prompt when no hand detected
            msg = "Show your hand to the camera"
            (mw, _), _ = cv2.getTextSize(
                msg, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            cv2.putText(frame, msg, ((fw - mw) // 2, fh // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (140, 140, 140), 2)

        # FPS
        now   = time.perf_counter()
        fps   = 1.0 / max(now - prev_t, 1e-6)
        prev_t = now
        cv2.putText(frame, f"FPS: {fps:.0f}", (14, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (160, 160, 160), 2)

        # Legend strip at top
        display = draw_legend(frame, all_fins, fw)

        cv2.imshow("Hand Tracking Demo  (q to quit)", display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()


if __name__ == "__main__":
    main()
