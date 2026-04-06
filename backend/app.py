"""
Flask server for the drowsiness detector.
Exposes:
  GET /           — HTML dashboard
  GET /video      — MJPEG stream
  GET /metrics    — JSON snapshot of current detector state
"""

import threading
import time
from flask import Flask, Response, jsonify
from drowsiness_detector import (
    detect_drowsiness, Config,
    PerclosTracker, BlinkDetector,
    eye_aspect_ratio, head_pitch_yaw, fuse_signals, score_to_level,
    _landmarks_to_array, _face_mesh,
    LEFT_EYE, RIGHT_EYE,
    calibrate,
)
import cv2
import numpy as np

app = Flask(__name__)

# ─── Shared state (updated by the metrics thread) ─────────────────────────────

_state = {
    "level":      "ok",
    "score":      0.0,
    "ear":        0.0,
    "perclos":    0.0,
    "pitch":      0.0,
    "blink_bpm":  0.0,
    "face_found": False,
    "calibrated": False,
}
_state_lock = threading.Lock()

_cfg = Config()
_cap = None
_ear_closed = 0.20   # will be set after calibration


def _metrics_loop():
    """Background thread: reads frames, updates _state without encoding MJPEG."""
    global _cap, _ear_closed

    _cap = cv2.VideoCapture(0)
    if not _cap.isOpened():
        return

    # Run calibration
    baseline = calibrate(_cap, _cfg)
    _ear_closed = baseline * _cfg.ear_closed_ratio

    perclos = PerclosTracker(_cfg.perclos_window)
    blinker = BlinkDetector(_ear_closed)

    with _state_lock:
        _state["calibrated"] = True

    while _cap.isOpened():
        ok, frame = _cap.read()
        if not ok:
            time.sleep(0.03)
            continue

        now  = time.time()
        h, w = frame.shape[:2]
        frame = cv2.flip(frame, 1)
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = _face_mesh.process(rgb)

        if result.multi_face_landmarks:
            lm   = result.multi_face_landmarks[0].landmark
            mesh = _landmarks_to_array(lm, w, h)

            avg_ear   = (eye_aspect_ratio(mesh[LEFT_EYE]) +
                         eye_aspect_ratio(mesh[RIGHT_EYE])) / 2.0
            is_closed = avg_ear < _ear_closed
            perc      = perclos.update(is_closed, now)
            bpm       = blinker.update(avg_ear, now, _cfg.blink_window)
            pitch, _  = head_pitch_yaw(lm, w, h)
            score     = fuse_signals(perc, pitch, bpm, _cfg)
            level     = score_to_level(score, _cfg)

            with _state_lock:
                _state.update(
                    level=level,
                    score=round(score, 3),
                    ear=round(avg_ear, 3),
                    perclos=round(perc, 3),
                    pitch=round(pitch, 1),
                    blink_bpm=round(bpm, 1),
                    face_found=True,
                )
        else:
            with _state_lock:
                _state["face_found"] = False

        time.sleep(0.02)   # ~50 Hz metric updates


# Start metrics thread once on import
_metrics_thread = threading.Thread(target=_metrics_loop, daemon=True)
_metrics_thread.start()


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/video")
def video():
    """MJPEG stream with overlaid HUD."""
    return Response(
        detect_drowsiness(source=0, cfg=_cfg),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/metrics")
def metrics():
    """JSON snapshot — polled by the frontend every 200 ms."""
    with _state_lock:
        data = dict(_state)
    return jsonify(data)


@app.route("/")
def home():
    """Serve the dashboard HTML inline (no template folder needed)."""
    with open("index.html", "r") as f:
        return f.read()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, threaded=True)