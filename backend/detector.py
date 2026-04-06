"""
Drowsiness Detection — Multi-Signal Fusion Edition
===================================================
Improvements over the original:
  • PERCLOS metric  — % eye closure over a rolling window (more robust than a raw counter)
  • Head pose       — detects nodding/drooping even when eyes stay open
  • Blink rate      — tracks microsleeps and abnormally low blink frequency
  • Adaptive EAR    — calibrates to each user's eye geometry at startup
  • Multi-signal fusion — weighted drowsiness score 0-1 across all three signals
  • Tiered alerts   — Warning / Alert / Critical with escalating responses
  • Time-based      — all windows are in seconds, not assumed frame counts
  • Cross-platform  — robust audio fallback chain

Dependencies:
    pip install opencv-python mediapipe numpy scipy
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import threading
import platform
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

# ─── MediaPipe setup ──────────────────────────────────────────────────────────

_mp_face   = mp.solutions.face_mesh
_face_mesh = _mp_face.FaceMesh(
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# Landmark index sets (MediaPipe 468-point mesh)
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]

# Nose/chin/forehead for head-pose PnP
POSE_POINTS = {
    "nose_tip":     1,
    "chin":         152,
    "left_eye_l":   33,
    "right_eye_r":  263,
    "left_mouth":   61,
    "right_mouth":  291,
}

# ─── Config ───────────────────────────────────────────────────────────────────

@dataclass
class Config:
    # Calibration
    calibration_seconds: float = 3.0        # How long to collect baseline EAR
    ear_closed_ratio:    float = 0.75       # Fraction of baseline EAR that means "closed"

    # PERCLOS (EAR-based)
    perclos_window:      float = 3.0        # Rolling window length (seconds)
    perclos_warn:        float = 0.20       # 20 % of window eyes closed → warning
    perclos_alert:       float = 0.45       # 45 %
    perclos_critical:    float = 0.70       # 70 %

    # Head pose
    pitch_warn:          float = 15.0       # Degrees of downward nod
    pitch_alert:         float = 25.0
    pitch_critical:      float = 35.0

    # Blink rate (blinks per minute)
    blink_window:        float = 60.0       # BPM measurement window
    blink_low_warn:      float = 10.0       # Below 10 BPM is suspicious
    blink_low_alert:     float = 6.0
    blink_high_warn:     float = 25.0       # Above 25 BPM = micro-sleep attempts
    blink_high_alert:    float = 35.0

    # Signal weights (must sum to 1.0)
    w_perclos:           float = 0.40
    w_pose:              float = 0.35
    w_blink:             float = 0.25

    # Alert thresholds on fused score
    score_warn:          float = 0.35
    score_alert:         float = 0.60
    score_critical:      float = 0.80

    # Alert cooldown so we don't spam sound
    alert_cooldown:      float = 4.0


# ─── EAR ──────────────────────────────────────────────────────────────────────

def eye_aspect_ratio(eye: np.ndarray) -> float:
    """
    Eye Aspect Ratio (Soukupová & Čech, 2016).
    eye: (6, 2) array of landmark coords.
    """
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)


# ─── Head pose ────────────────────────────────────────────────────────────────

# Generic 3-D face model (mm), same for every person
_MODEL_POINTS = np.array([
    [0.0,    0.0,    0.0  ],   # nose tip
    [0.0,   -63.6, -12.5 ],   # chin
    [-43.3,  32.7, -26.0 ],   # left eye left corner
    [43.3,   32.7, -26.0 ],   # right eye right corner
    [-28.9, -28.9, -24.1 ],   # left mouth corner
    [28.9,  -28.9, -24.1 ],   # right mouth corner
], dtype=np.float64)

def _camera_matrix(w: int, h: int) -> np.ndarray:
    f = w  # Approximate focal length
    return np.array([[f, 0, w / 2],
                     [0, f, h / 2],
                     [0, 0, 1   ]], dtype=np.float64)

def head_pitch_yaw(landmarks, w: int, h: int) -> tuple[float, float]:
    """
    Returns (pitch, yaw) in degrees.
    Positive pitch  = looking down (head drop = drowsiness signal).
    Positive yaw    = looking right.
    """
    idx = list(POSE_POINTS.values())
    image_points = np.array(
        [[landmarks[i].x * w, landmarks[i].y * h] for i in idx],
        dtype=np.float64,
    )
    cam   = _camera_matrix(w, h)
    dist  = np.zeros((4, 1))
    ok, rvec, _ = cv2.solvePnP(
        _MODEL_POINTS, image_points, cam, dist,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return 0.0, 0.0
    rmat, _ = cv2.Rodrigues(rvec)
    proj     = np.hstack([rmat, np.zeros((3, 1))])
    _, _, _, _, _, _, euler = cv2.decomposeProjectionMatrix(proj)
    pitch = float(euler[0])
    yaw   = float(euler[1])
    return pitch, yaw


# ─── Blink detector ───────────────────────────────────────────────────────────

class BlinkDetector:
    def __init__(self, ear_threshold: float):
        self.threshold    = ear_threshold
        self._was_closed  = False
        self._blink_times: deque = deque()   # timestamps of completed blinks

    def update(self, ear: float, now: float, window: float) -> float:
        """Register current EAR, return current blinks-per-minute."""
        closed = ear < self.threshold
        if self._was_closed and not closed:
            self._blink_times.append(now)
        self._was_closed = closed

        # Prune old blinks
        cutoff = now - window
        while self._blink_times and self._blink_times[0] < cutoff:
            self._blink_times.popleft()

        elapsed = min(now - (self._blink_times[0] if self._blink_times else now), window)
        if elapsed < 5:
            return 15.0  # Not enough data yet — return a neutral rate
        return len(self._blink_times) / elapsed * 60.0


# ─── PERCLOS tracker ──────────────────────────────────────────────────────────

class PerclosTracker:
    def __init__(self, window_seconds: float):
        self._window = window_seconds
        # Each entry: (timestamp, is_closed)
        self._samples: deque = deque()

    def update(self, is_closed: bool, now: float) -> float:
        """Add sample, return fraction of window that eyes were closed."""
        self._samples.append((now, is_closed))
        cutoff = now - self._window
        while self._samples and self._samples[0][0] < cutoff:
            self._samples.popleft()
        if not self._samples:
            return 0.0
        closed_count = sum(1 for _, c in self._samples if c)
        return closed_count / len(self._samples)


# ─── Signal fusion ────────────────────────────────────────────────────────────

def _perclos_score(perclos: float, cfg: Config) -> float:
    if perclos < cfg.perclos_warn:
        return 0.0
    elif perclos < cfg.perclos_alert:
        return np.interp(perclos, [cfg.perclos_warn, cfg.perclos_alert], [0.0, 0.5])
    elif perclos < cfg.perclos_critical:
        return np.interp(perclos, [cfg.perclos_alert, cfg.perclos_critical], [0.5, 0.9])
    return 1.0


def _pose_score(pitch: float, cfg: Config) -> float:
    p = abs(pitch)
    if p < cfg.pitch_warn:
        return 0.0
    elif p < cfg.pitch_alert:
        return np.interp(p, [cfg.pitch_warn, cfg.pitch_alert], [0.0, 0.5])
    elif p < cfg.pitch_critical:
        return np.interp(p, [cfg.pitch_alert, cfg.pitch_critical], [0.5, 0.9])
    return 1.0


def _blink_score(bpm: float, cfg: Config) -> float:
    # Low blink rate
    if bpm < cfg.blink_low_alert:
        return 0.8
    if bpm < cfg.blink_low_warn:
        return np.interp(bpm, [cfg.blink_low_alert, cfg.blink_low_warn], [0.8, 0.3])
    # High blink rate (microsleep attempts)
    if bpm > cfg.blink_high_alert:
        return 0.9
    if bpm > cfg.blink_high_warn:
        return np.interp(bpm, [cfg.blink_high_warn, cfg.blink_high_alert], [0.3, 0.9])
    return 0.0


def fuse_signals(perclos: float, pitch: float, bpm: float, cfg: Config) -> float:
    ps = _perclos_score(perclos, cfg)
    hs = _pose_score(pitch, cfg)
    bs = _blink_score(bpm, cfg)
    return cfg.w_perclos * ps + cfg.w_pose * hs + cfg.w_blink * bs


# ─── Alert level ─────────────────────────────────────────────────────────────

@dataclass
class AlertLevel:
    name:  str
    color: tuple   # BGR
    sound_times: int = 1   # How many beeps

LEVELS = {
    "ok":       AlertLevel("OK",       (0,   200, 0  ), 0),
    "warning":  AlertLevel("WARNING",  (0,   200, 255), 1),
    "alert":    AlertLevel("ALERT",    (0,   130, 255), 2),
    "critical": AlertLevel("CRITICAL", (0,   0,   255), 3),
}

def score_to_level(score: float, cfg: Config) -> str:
    if score >= cfg.score_critical:
        return "critical"
    elif score >= cfg.score_alert:
        return "alert"
    elif score >= cfg.score_warn:
        return "warning"
    return "ok"


# ─── Audio ────────────────────────────────────────────────────────────────────

def _play_alert(n: int = 1):
    """Non-blocking alert sound — tries multiple methods."""
    def _play():
        for _ in range(n):
            try:
                import winsound
                winsound.Beep(880, 400)
                continue
            except ImportError:
                pass
            try:
                import subprocess
                if platform.system() == "Darwin":
                    subprocess.run(["afplay", "/System/Library/Sounds/Funk.aiff"],
                                   check=False, capture_output=True)
                else:
                    subprocess.run(["paplay", "/usr/share/sounds/freedesktop/stereo/alarm-clock-elapsed.oga"],
                                   check=False, capture_output=True)
            except Exception:
                # Last resort: terminal bell
                print("\a", end="", flush=True)
            time.sleep(0.3)

    threading.Thread(target=_play, daemon=True).start()


# ─── Calibration ─────────────────────────────────────────────────────────────

def calibrate(cap: cv2.VideoCapture, cfg: Config) -> float:
    """
    Collect EAR samples for `cfg.calibration_seconds` with eyes open.
    Returns the user's personal open-eye EAR baseline.
    """
    samples = []
    deadline = time.time() + cfg.calibration_seconds
    while time.time() < deadline:
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.flip(frame, 1)
        h, w   = frame.shape[:2]
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = _face_mesh.process(rgb)

        remaining = deadline - time.time()
        bar_w = int(w * (1 - remaining / cfg.calibration_seconds))
        cv2.rectangle(frame, (0, h - 6), (bar_w, h), (0, 230, 120), -1)
        cv2.putText(frame, "Keep eyes open — calibrating...",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 230, 120), 2)

        if result.multi_face_landmarks:
            mesh = _landmarks_to_array(result.multi_face_landmarks[0].landmark, w, h)
            ear  = (eye_aspect_ratio(mesh[LEFT_EYE]) +
                    eye_aspect_ratio(mesh[RIGHT_EYE])) / 2.0
            samples.append(ear)

        cv2.imshow("Drowsiness Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    if not samples:
        return 0.28   # Sensible fallback
    baseline = float(np.median(samples))
    print(f"[calibration] baseline EAR = {baseline:.4f}")
    return baseline


def _landmarks_to_array(landmarks, w: int, h: int) -> np.ndarray:
    return np.array([[p.x * w, p.y * h] for p in landmarks], dtype=np.float32)


# ─── Overlay helpers ─────────────────────────────────────────────────────────

def _draw_eye_contour(frame, pts, color):
    hull = cv2.convexHull(pts.astype(np.int32))
    cv2.drawContours(frame, [hull], -1, color, 1)


def _score_bar(frame, score: float, level: str, x=20, y=80, w=200, h=18):
    fill = LEVELS[level].color
    cv2.rectangle(frame, (x, y), (x + w, y + h), (60, 60, 60), -1)
    cv2.rectangle(frame, (x, y), (x + int(w * score), y + h), fill, -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (180, 180, 180), 1)
    cv2.putText(frame, f"{score:.2f}", (x + w + 6, y + 13),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)


# ─── Main loop ────────────────────────────────────────────────────────────────

def detect_drowsiness(source: int = 0, cfg: Optional[Config] = None):
    """
    Generator that yields MJPEG frames suitable for a Flask /video_feed route,
    or can be run standalone (call with show=True or just iterate frames).
    """
    if cfg is None:
        cfg = Config()

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source {source}")

    # ── Calibrate ──
    ear_baseline = calibrate(cap, cfg)
    ear_closed   = ear_baseline * cfg.ear_closed_ratio

    # ── Trackers ──
    perclos  = PerclosTracker(cfg.perclos_window)
    blinker  = BlinkDetector(ear_closed)

    last_alert_time = 0.0
    prev_level      = "ok"

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        now  = time.time()
        h, w = frame.shape[:2]
        frame = cv2.flip(frame, 1)
        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = _face_mesh.process(rgb)

        score = 0.0
        level = "ok"

        if result.multi_face_landmarks:
            lm   = result.multi_face_landmarks[0].landmark
            mesh = _landmarks_to_array(lm, w, h)

            # ── EAR ──
            left_pts  = mesh[LEFT_EYE]
            right_pts = mesh[RIGHT_EYE]
            avg_ear   = (eye_aspect_ratio(left_pts) +
                         eye_aspect_ratio(right_pts)) / 2.0
            is_closed = avg_ear < ear_closed

            # ── PERCLOS ──
            perc = perclos.update(is_closed, now)

            # ── Blink rate ──
            bpm = blinker.update(avg_ear, now, cfg.blink_window)

            # ── Head pose ──
            pitch, yaw = head_pitch_yaw(lm, w, h)

            # ── Fuse ──
            score = fuse_signals(perc, pitch, bpm, cfg)
            level = score_to_level(score, cfg)

            # ── Overlays ──
            eye_color = (0, 0, 220) if is_closed else (0, 220, 100)
            _draw_eye_contour(frame, left_pts,  eye_color)
            _draw_eye_contour(frame, right_pts, eye_color)

            # HUD
            cv2.putText(frame, f"EAR:{avg_ear:.3f}  PERCLOS:{perc:.2f}",
                        (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
            cv2.putText(frame, f"Pitch:{pitch:+.1f}  BPM:{bpm:.1f}",
                        (20, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        # ── Score bar ──
        _score_bar(frame, score, level)

        # ── Alert banner ──
        al = LEVELS[level]
        if level != "ok":
            cv2.rectangle(frame, (0, h - 50), (w, h), al.color, -1)
            cv2.putText(frame, f"⚠  {al.name}  ⚠",
                        (w // 2 - 80, h - 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        # ── Sound (throttled) ──
        if level != "ok" and (level != prev_level or now - last_alert_time > cfg.alert_cooldown):
            if al.sound_times > 0:
                _play_alert(al.sound_times)
            last_alert_time = now
        prev_level = level

        # ── Yield MJPEG frame ──
        _, jpeg = cv2.imencode(".jpg", frame)
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")

    cap.release()


# ─── Standalone entry point ───────────────────────────────────────────────────

def run_standalone():
    """Run the detector in a local window (no Flask needed)."""
    cfg = Config()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: cannot open camera.")
        return

    ear_baseline = calibrate(cap, cfg)
    ear_closed   = ear_baseline * cfg.ear_closed_ratio

    perclos  = PerclosTracker(cfg.perclos_window)
    blinker  = BlinkDetector(ear_closed)
    last_alert = 0.0
    prev_level = "ok"

    print("\nDetector running — press Q to quit.\n")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        now  = time.time()
        h, w = frame.shape[:2]
        frame = cv2.flip(frame, 1)
        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = _face_mesh.process(rgb)

        score = 0.0
        level = "ok"

        if result.multi_face_landmarks:
            lm   = result.multi_face_landmarks[0].landmark
            mesh = _landmarks_to_array(lm, w, h)

            avg_ear   = (eye_aspect_ratio(mesh[LEFT_EYE]) +
                         eye_aspect_ratio(mesh[RIGHT_EYE])) / 2.0
            is_closed = avg_ear < ear_closed
            perc      = perclos.update(is_closed, now)
            bpm       = blinker.update(avg_ear, now, cfg.blink_window)
            pitch, _  = head_pitch_yaw(lm, w, h)
            score     = fuse_signals(perc, pitch, bpm, cfg)
            level     = score_to_level(score, cfg)

            eye_color = (0, 0, 220) if is_closed else (0, 220, 100)
            _draw_eye_contour(frame, mesh[LEFT_EYE],  eye_color)
            _draw_eye_contour(frame, mesh[RIGHT_EYE], eye_color)

            cv2.putText(frame, f"EAR:{avg_ear:.3f}  PERCLOS:{perc:.2f}",
                        (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
            cv2.putText(frame, f"Pitch:{pitch:+.1f} deg  BPM:{bpm:.1f}",
                        (20, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        _score_bar(frame, score, level)

        al = LEVELS[level]
        if level != "ok":
            cv2.rectangle(frame, (0, h - 50), (w, h), al.color, -1)
            cv2.putText(frame, f"DROWSY — {al.name}",
                        (w // 2 - 90, h - 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        if level != "ok" and (level != prev_level or now - last_alert > cfg.alert_cooldown):
            if al.sound_times > 0:
                _play_alert(al.sound_times)
            last_alert = now
        prev_level = level

        cv2.imshow("Drowsiness Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_standalone()