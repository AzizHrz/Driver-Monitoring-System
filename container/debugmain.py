"""
FILE:    debugmain.py  (v2.0 — debug/diagnostic version of main.py)
WHERE:   Runs INSIDE the Docker container (Python 3.11) on the Raspberry Pi
PARTNER: bridge.py must be running on the Pi HOST before starting this

PURPOSE: Identical to main.py but with structured debug logging added.
         Use this file when diagnosing issues with the pipeline.
         Switch back to main.py for clean production output.

DEBUG FEATURES ADDED (vs main.py):
  - Heartbeat log every 5 seconds showing:
      state, emotion, eyes_closed_frames, is_drowsy, frames_received, frames_annotated
  - State transition log on every change:
      *** STATE  NO_FACE → VERIFYING ***
  - FaceNet similarity score logged during verification
  - Drowsy alert trigger and clear events logged
  - Emotion change events logged
  - /status endpoint at http://<pi-ip>:5001/status returns JSON:
      { frames_received, frames_annotated, annotated_ready, raw_frame_ready }
  - Small debug counter overlay on bottom-right of the browser stream
  - All startup steps logged with ✓/✗ symbols

CONTROL:
  - Debug logging is ON by default
  - To disable: set environment variable DEBUG=false when running docker run
      docker run -e DEBUG=false ... driver-monitor:pi python3 debugmain.py

HOW TO USE FOR DIAGNOSIS:
  1. If frames_received stays 0  → bridge.py is not streaming / not reachable
  2. If frames_annotated stays 0 → the main loop is stuck (IR gate, AI crash)
  3. If state stays VERIFYING    → owner embeddings not matching (check owner*.jpeg)
  4. If no drowsy alert fires    → eye model not detecting closed eyes (check model path)

EVERYTHING ELSE is identical to main.py — same AI pipeline, same REST API,
same MJPEG stream on port 5001, same bridge command format.

USAGE:
  Terminal 1 (Pi host):      python3 bridge.py
  Terminal 2 (Pi container): docker run ... driver-monitor:pi python3 debugmain.py
"""

import base64
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import requests
from deepface import DeepFace
from flask import Flask, Response, jsonify
from tensorflow.keras.models import load_model

import config
import connect

# ── Debug ─────────────────────────────────────────────────────────────────────
DEBUG_MODE      = os.getenv("DEBUG", "true").lower() == "true"
DEBUG_LOG_EVERY = 5.0   # seconds between heartbeat logs

_last_heartbeat = 0.0


def dbg(msg: str, force: bool = False):
    if DEBUG_MODE or force:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def log_state(old: str, new: str, extra: str = ""):
    line = f"*** STATE  {old} → {new}"
    if extra:
        line += f"  ({extra})"
    print(f"[{time.strftime('%H:%M:%S')}] {line} ***", flush=True)


# ── Thread pool for cloud I/O ─────────────────────────────────────────────────
_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="cloud")

def _async(fn, *args):
    _pool.submit(fn, *args)


# ── State machine ─────────────────────────────────────────────────────────────
class DS:
    NO_FACE        = "NO_FACE"
    VERIFYING      = "VERIFYING"
    VERIFIED       = "VERIFIED"
    UNKNOWN_PERSON = "UNKNOWN_PERSON"


class DriverStateMachine:
    def __init__(self):
        self.state                = DS.NO_FACE
        self.verification_counter = 0
        self.eyes_closed_frames   = 0
        self.unknown_alert_start  = 0.0
        self.emotion              = ""
        self.countdown            = None

    def transition(self, new_state: str, extra: str = ""):
        if new_state != self.state:
            log_state(self.state, new_state, extra)
        self.state = new_state

    def reset_to_no_face(self):
        self.transition(DS.NO_FACE)
        self.verification_counter = 0
        self.eyes_closed_frames   = 0
        self.emotion              = ""
        self.countdown            = None

    @property
    def is_drowsy(self) -> bool:
        return self.eyes_closed_frames > config.DROWSY_FRAME_THRESHOLD


# ── Load models ───────────────────────────────────────────────────────────────
print("[main] Loading models …", flush=True)

eye_model = None
try:
    eye_model = load_model(config.EYE_MODEL_PATH)
    print(f"[main] ✓ eye model loaded", flush=True)
except Exception as e:
    print(f"[main] ✗ eye model FAILED: {e}", flush=True)

face_cascade   = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
left_eye_casc  = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_lefteye_2splits.xml")
right_eye_casc = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_righteye_2splits.xml")

print("[main] ✓ Haar cascades loaded", flush=True)

owner_embeddings = []
for path in config.OWNER_IMG_PATHS:
    img = cv2.imread(path)
    if img is not None:
        try:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            emb = DeepFace.represent(
                img_rgb, model_name=config.FACENET_MODEL,
                enforce_detection=False)[0]["embedding"]
            owner_embeddings.append(emb)
            print(f"[main] ✓ {path}", flush=True)
        except Exception as e:
            print(f"[main] ✗ embedding {path}: {e}", flush=True)
    else:
        print(f"[main] ✗ not found: {path}", flush=True)

owner_mean = np.mean(owner_embeddings, axis=0) if owner_embeddings else None
print(f"[main] {len(owner_embeddings)}/3 owner embeddings loaded", flush=True)
print("[main] ✓ All models ready.\n", flush=True)


# ── Shared buffers ────────────────────────────────────────────────────────────
_raw_lock      = threading.Lock()
_raw_frame     = None
_raw_jpg_bytes = None

_ann_lock = threading.Lock()
_ann_jpg  = None      # annotated JPEG bytes served to browser

_frames_received  = 0
_frames_annotated = 0


# ── MJPEG stream reader (background thread) ───────────────────────────────────
def _stream_reader():
    global _raw_frame, _raw_jpg_bytes, _frames_received
    url = f"http://localhost:{config.BRIDGE_PORT}/stream"
    while True:
        try:
            print(f"[stream] Connecting → {url}", flush=True)
            resp = requests.get(url, stream=True, timeout=10)
            print(f"[stream] ✓ Connected HTTP {resp.status_code}", flush=True)
            buf = b""
            for chunk in resp.iter_content(chunk_size=4096):
                buf += chunk
                s = buf.find(b"\xff\xd8")
                e = buf.find(b"\xff\xd9")
                if s != -1 and e > s:
                    jpg   = buf[s: e + 2]
                    buf   = buf[e + 2:]
                    frame = cv2.imdecode(
                        np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if frame is not None:
                        with _raw_lock:
                            _raw_frame     = frame
                            _raw_jpg_bytes = jpg
                        _frames_received += 1
        except Exception as ex:
            print(f"[stream] ERROR: {ex} — retry in 3 s", flush=True)
            time.sleep(3)


# ── OpenCV drawing ────────────────────────────────────────────────────────────
_FONT = cv2.FONT_HERSHEY_SIMPLEX

def _put(img, text, pos, color=config.COLOR_WHITE, scale=0.65, thick=2):
    cv2.putText(img, text, pos, _FONT, scale, color, thick)

def _draw_status_bar(img, sm: DriverStateMachine):
    h, w = img.shape[:2]
    bar = img.copy()
    cv2.rectangle(bar, (0, 0), (w, 44), (0, 0, 0), -1)
    cv2.addWeighted(bar, 0.55, img, 0.45, 0, img)
    labels = {
        DS.NO_FACE:        ("NO FACE DETECTED",   config.COLOR_BLUE),
        DS.VERIFYING:      (f"Verifying ({sm.verification_counter}/{config.VERIFY_MAX_FAILS})",
                            config.COLOR_ORANGE),
        DS.VERIFIED:       ("VERIFIED",            config.COLOR_GREEN),
        DS.UNKNOWN_PERSON: (
            f"UNKNOWN! ({sm.countdown}s)" if sm.countdown else "UNKNOWN PERSON!",
            config.COLOR_RED),
    }
    label, color = labels.get(sm.state, ("", config.COLOR_WHITE))
    _put(img, label, (10, 30), color, scale=0.75, thick=2)
    if sm.emotion:
        _put(img, f"Emotion: {sm.emotion}", (w - 240, 30), config.COLOR_WHITE)

def _draw_face_box(img, x, y, w, h, sm: DriverStateMachine):
    colors = {
        DS.VERIFIED:       config.COLOR_GREEN,
        DS.UNKNOWN_PERSON: config.COLOR_RED,
        DS.VERIFYING:      config.COLOR_ORANGE,
        DS.NO_FACE:        config.COLOR_BLUE,
    }
    color = colors.get(sm.state, config.COLOR_WHITE)
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
    _put(img, sm.state.replace("_", " "), (x, y - 10), color)
    if sm.emotion:
        _put(img, f"Emotion: {sm.emotion}", (x, y - 30), config.COLOR_WHITE, scale=0.6)
    fh = img.shape[0]
    if sm.is_drowsy:
        cv2.rectangle(img, (0, fh - 48), (320, fh), config.COLOR_DARK_RED, -1)
        _put(img, "  DANGER - DROWSY!", (6, fh - 14), config.COLOR_WHITE, scale=0.75, thick=2)
    elif sm.eyes_closed_frames > 0:
        cv2.rectangle(img, (0, fh - 48), (310, fh), config.COLOR_DARK_BLUE, -1)
        _put(img, f"  EYES CLOSED ({sm.eyes_closed_frames})", (6, fh - 14),
             config.COLOR_WHITE, scale=0.65)

def _draw_debug_info(img):
    """Small counter panel — visible in browser stream."""
    if not DEBUG_MODE:
        return
    h, w = img.shape[:2]
    lines = [f"recv:{_frames_received}", f"ann:{_frames_annotated}"]
    y = h - 40
    for line in lines:
        _put(img, line, (w - 130, y), (160, 160, 160), scale=0.45, thick=1)
        y += 16

def _annotate(frame, sm, face_rect=None):
    out = frame.copy()
    if face_rect:
        _draw_face_box(out, *face_rect, sm)
    _draw_status_bar(out, sm)
    _draw_debug_info(out)
    return out


# ── Monitor Flask server ──────────────────────────────────────────────────────
monitor_app = Flask(__name__)

def _gen():
    """Yield annotated MJPEG frames."""
    while True:
        with _ann_lock:
            jpg = _ann_jpg
        if jpg is None:
            time.sleep(0.05)
            continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")

@monitor_app.route("/")
def index():
    return """<!DOCTYPE html>
<html>
<head>
  <title>Driver Monitor</title>
  <style>
    body{background:#111;display:flex;flex-direction:column;
         align-items:center;justify-content:center;min-height:100vh;
         margin:0;color:#eee;font-family:monospace;}
    img{border:2px solid #0f0;border-radius:6px;max-width:95vw;max-height:90vh;}
    h2{color:#0f0;margin-bottom:14px;}
    p{color:#666;font-size:11px;margin-top:6px;}
  </style>
</head>
<body>
  <h2>&#x1F697; Driver Monitoring System</h2>
  <img src="/stream"/>
  <p>Live annotated stream from Raspberry Pi</p>
</body>
</html>"""

@monitor_app.route("/stream")
def stream():
    return Response(_gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

@monitor_app.route("/status")
def status():
    return jsonify({
        "frames_received":  _frames_received,
        "frames_annotated": _frames_annotated,
        "annotated_ready":  _ann_jpg is not None,
        "raw_frame_ready":  _raw_frame is not None,
    })

def _run_monitor():
    monitor_app.run(host="0.0.0.0", port=config.MONITOR_PORT,
                    threaded=True, use_reloader=False)


# ── Bridge I/O helpers ────────────────────────────────────────────────────────
def _send_command(sm: DriverStateMachine):
    if sm.is_drowsy:
        l1, l2 = "Danger!!!", "DROWSY ALERT"
    elif sm.eyes_closed_frames > 0:
        l1, l2 = "EYES CLOSED!", ""
    elif sm.state == DS.VERIFIED:
        l1, l2 = "VERIFIED", f"Emotion:{sm.emotion}" if sm.emotion else ""
    elif sm.state == DS.UNKNOWN_PERSON:
        l1, l2 = "UNKNOWN PERSON!", f"Alert! {sm.countdown}s"
    elif sm.state == DS.VERIFYING:
        l1, l2 = "Verifying...", f"{sm.verification_counter}/{config.VERIFY_MAX_FAILS}"
    else:
        l1, l2 = "NO FACE", ""
    try:
        requests.post(
            f"http://localhost:{config.BRIDGE_PORT}/command",
            json={"state": sm.state, "drowsy": sm.is_drowsy,
                  "eyes_closed": sm.eyes_closed_frames > 0,
                  "lcd_line1": l1, "lcd_line2": l2},
            timeout=1)
    except Exception as e:
        dbg(f"[cmd] {e}")

def _get_ir() -> bool:
    try:
        return requests.get(
            f"http://localhost:{config.BRIDGE_PORT}/ir_status",
            timeout=1).json().get("is_active", False)
    except:
        return False

def _wait_for_bridge(timeout: int = 30):
    print("[main] Waiting for bridge …", flush=True)
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if requests.get(
                f"http://localhost:{config.BRIDGE_PORT}/health",
                timeout=2).status_code == 200:
                print("[main] ✓ Bridge is up.", flush=True)
                return
        except:
            pass
        time.sleep(1)
    raise RuntimeError("Bridge did not respond.")


# ── Main loop ─────────────────────────────────────────────────────────────────
def main():
    global _frames_annotated, _last_heartbeat

    _wait_for_bridge()

    threading.Thread(target=_stream_reader, daemon=True).start()
    threading.Thread(target=_run_monitor,   daemon=True).start()

    pi_ip = "172.20.10.14"
    print(f"\n[main] ✓ Open in browser → http://{pi_ip}:{config.MONITOR_PORT}/", flush=True)
    print(f"[main] ✓ Status          → http://{pi_ip}:{config.MONITOR_PORT}/status\n", flush=True)

    sm          = DriverStateMachine()
    frame_count = 0

    # ── IR gate: wait for person once at startup ──────────────────────────────
    # FIX: was re-checked every frame (blocking HTTP call in hot loop).
    # Now runs ONCE before the main loop, then never again.
    print("[main] Checking IR sensor …", flush=True)
    if not _get_ir():
        print("[main] No person on IR — waiting …", flush=True)
        _send_command(sm)
        while not _get_ir():
            time.sleep(0.5)
    print("[main] ✓ IR active — starting monitoring loop.\n", flush=True)

    _last_heartbeat = time.time()

    while True:
        # ── Grab raw frame ────────────────────────────────────────────────────
        with _raw_lock:
            frame = _raw_frame.copy() if _raw_frame is not None else None
            jpg   = _raw_jpg_bytes

        if frame is None:
            dbg("Waiting for first frame …")
            time.sleep(0.05)
            continue

        frame_count  += 1
        current_time  = time.time()

        # ── Heartbeat log every 5 s ───────────────────────────────────────────
        if current_time - _last_heartbeat >= DEBUG_LOG_EVERY:
            print(
                f"[{time.strftime('%H:%M:%S')}] "
                f"state={sm.state:<15}  emotion={sm.emotion or '-':<10}  "
                f"eyes={sm.eyes_closed_frames}  drowsy={sm.is_drowsy}  "
                f"recv={_frames_received}  ann={_frames_annotated}",
                flush=True)
            _last_heartbeat = current_time

        # ── Face detection ────────────────────────────────────────────────────
        gray      = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces     = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
        face_rect = None

        if len(faces) > 0:
            x, y, w, h = faces[0]
            face_rect  = (x, y, w, h)
            face_roi   = frame[y: y + h, x: x + w]
            gray_face  = gray[y: y + h, x: x + w]

            # ── State transitions ─────────────────────────────────────────────
            if sm.state == DS.NO_FACE:
                sm.transition(DS.VERIFYING)
                sm.verification_counter = 0

            elif sm.state == DS.VERIFYING:
                if frame_count % config.AI_EVERY_N_FRAMES == 0:
                    if owner_mean is not None:
                        try:
                            face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                            emb = DeepFace.represent(
                                face_rgb, model_name=config.FACENET_MODEL,
                                enforce_detection=False)[0]["embedding"]
                            sim = float(np.dot(emb, owner_mean) / (
                                np.linalg.norm(emb) * np.linalg.norm(owner_mean)))
                            dbg(f"Similarity: {sim:.3f} threshold={config.VERIFY_THRESHOLD}")
                            if sim >= config.VERIFY_THRESHOLD:
                                sm.transition(DS.VERIFIED, f"sim={sim:.3f}")
                                sm.verification_counter = 0
                            else:
                                sm.verification_counter += 1
                                dbg(f"Verify fail {sm.verification_counter}/{config.VERIFY_MAX_FAILS}")
                                if sm.verification_counter >= config.VERIFY_MAX_FAILS:
                                    sm.transition(DS.UNKNOWN_PERSON)
                                    sm.unknown_alert_start = current_time
                        except Exception as e:
                            print(f"[verify] ERROR: {e}", flush=True)
                    else:
                        sm.transition(DS.VERIFIED, "no owner images")

            elif sm.state == DS.UNKNOWN_PERSON:
                elapsed      = current_time - sm.unknown_alert_start
                sm.countdown = max(0, int(config.UNKNOWN_ALERT_SEC - elapsed))
                if elapsed >= config.UNKNOWN_ALERT_SEC:
                    sm.transition(DS.VERIFYING, "alert expired")
                    sm.verification_counter = 0
                    sm.countdown = None
                else:
                    if jpg:
                        _async(connect.send_latest_photo_to_firebase,
                               base64.b64encode(jpg).decode("utf-8"))
                    _async(connect.write_to_field, 1, 1)

            elif sm.state == DS.VERIFIED:
                if frame_count % config.AI_EVERY_N_FRAMES == 0:
                    try:
                        face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)

                        # Emotion
                        result = DeepFace.analyze(
                            face_rgb, actions=["emotion"],
                            enforce_detection=False, silent=True)
                        new_emotion = result[0]["dominant_emotion"]
                        if new_emotion != sm.emotion:
                            dbg(f"Emotion: {sm.emotion or '-'} → {new_emotion}", force=True)
                            sm.emotion = new_emotion
                        if sm.emotion in ("sad", "angry"):
                            _async(connect.write_to_field, 3, 1)

                        # Eye state
                        closed = 0
                        if eye_model is not None:
                            for ec in [left_eye_casc, right_eye_casc]:
                                for (ex, ey, ew, eh) in ec.detectMultiScale(gray_face):
                                    eye_img = (
                                        cv2.resize(gray_face[ey:ey+eh, ex:ex+ew], (52, 52))
                                        .astype("float32") / 255.0)
                                    pred = eye_model.predict(
                                        np.expand_dims(eye_img, axis=(0, -1)),
                                        verbose=0)[0][0]
                                    if pred > 0.5:
                                        closed += 1

                        prev_ecf = sm.eyes_closed_frames
                        sm.eyes_closed_frames = sm.eyes_closed_frames + 1 if closed >= 2 else 0

                        if sm.is_drowsy and not (prev_ecf > config.DROWSY_FRAME_THRESHOLD):
                            print(f"[{time.strftime('%H:%M:%S')}] *** DROWSY ALERT! ***",
                                  flush=True)
                        elif prev_ecf > 0 and sm.eyes_closed_frames == 0:
                            dbg("Eyes open — alert cleared.", force=True)

                        _async(connect.write_to_field, 3,
                               2 if sm.is_drowsy else 0)

                    except Exception as e:
                        print(f"[analysis] ERROR: {e}", flush=True)
        else:
            if sm.state != DS.NO_FACE:
                sm.reset_to_no_face()

        # ── Draw ALL overlays ─────────────────────────────────────────────────
        annotated = _annotate(frame, sm, face_rect)

        # ── Encode and publish to browser stream ──────────────────────────────
        ok, enc = cv2.imencode(
            ".jpg", annotated,
            [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_QUALITY])
        if ok:
            with _ann_lock:
                _ann_jpg = enc.tobytes()
            _frames_annotated += 1
        else:
            print("[main] WARNING: imencode failed!", flush=True)

        # ── Send hardware command to bridge ───────────────────────────────────
        _send_command(sm)


if __name__ == "__main__":
    main()