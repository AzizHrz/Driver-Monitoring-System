# main.py
# ─────────────────────────────────────────────────────────────────────────────
# Runs INSIDE the Docker container (Python 3.11).
#
# Responsibilities:
#   • Pull raw MJPEG from bridge   ← http://localhost:5000/stream
#   • Run full AI pipeline
#       - Face detection (Haar)
#       - Face verification (FaceNet cosine similarity)
#       - Emotion analysis (DeepFace)
#       - Eye-state classification (custom eye_state_model.h5)
#   • Draw ALL OpenCV overlays onto each frame
#   • Serve annotated MJPEG stream → http://0.0.0.0:5001/
#     (open this URL in a browser on any device on the network)
#   • POST hardware commands to bridge → http://localhost:5000/command
#   • Write to Firebase and ThingSpeak via connect.py
#
# FIXES vs previous version:
#   1. No cv2.imshow / Qt dependency — display is MJPEG-over-HTTP.
#   2. No time.sleep() on any hot path; throttling is done via frame counters.
#   3. Firebase called through connect.py (properly initialised, error-safe).
#   4. ThingSpeak called through connect.py (rate-limit guard included).
#   5. OpenCV drawing is entirely in this file — bridge.py draws nothing.
#   6. AI (DeepFace) runs every AI_EVERY_N_FRAMES to keep the pipeline smooth.
#   7. Firebase + ThingSpeak calls are dispatched to a background thread pool
#      so they never stall the frame loop.
#   8. Stream reader uses a proper MJPEG parser (SOI/EOI byte markers).
# ─────────────────────────────────────────────────────────────────────────────

import base64
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import requests
from deepface import DeepFace
from flask import Flask, Response
from tensorflow.keras.models import load_model

import config
import connect

# ── Thread pool for non-blocking cloud calls ──────────────────────────────────
_cloud_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="cloud")

def _async(fn, *args, **kwargs):
    """Fire-and-forget: run fn in background thread pool."""
    _cloud_pool.submit(fn, *args, **kwargs)


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

    def reset_to_no_face(self):
        self.state                = DS.NO_FACE
        self.verification_counter = 0
        self.eyes_closed_frames   = 0
        self.emotion              = ""
        self.countdown            = None

    @property
    def is_drowsy(self) -> bool:
        return self.eyes_closed_frames > config.DROWSY_FRAME_THRESHOLD


# ── Load AI models (once at startup) ─────────────────────────────────────────
print("[main] Loading models …")
eye_model        = load_model(config.EYE_MODEL_PATH)
face_cascade     = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
left_eye_casc    = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_lefteye_2splits.xml"
)
right_eye_casc   = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_righteye_2splits.xml"
)

# Owner embeddings
owner_embeddings = []
for path in config.OWNER_IMG_PATHS:
    img = cv2.imread(path)
    if img is not None:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        emb = DeepFace.represent(
            img_rgb, model_name=config.FACENET_MODEL, enforce_detection=False
        )[0]["embedding"]
        owner_embeddings.append(emb)
        print(f"[main] Loaded {path}")
    else:
        print(f"[main] WARNING: {path} not found — verification will be skipped.")

owner_mean = np.mean(owner_embeddings, axis=0) if owner_embeddings else None
print("[main] Models ready.")


# ── Shared frame buffers (raw + annotated) ────────────────────────────────────
_raw_lock        = threading.Lock()
_raw_frame       = None   # BGR numpy array
_raw_jpg_bytes   = None   # original JPEG bytes (for Firebase)

_annotated_lock  = threading.Lock()
_annotated_jpg   = None   # JPEG bytes of annotated frame


# ── MJPEG stream reader (background thread) ───────────────────────────────────
def _stream_reader():
    global _raw_frame, _raw_jpg_bytes
    url = f"http://localhost:{config.BRIDGE_PORT}/stream"
    while True:
        try:
            print(f"[stream] Connecting to {url} …")
            resp = requests.get(url, stream=True, timeout=10)
            buf  = b""
            for chunk in resp.iter_content(chunk_size=4096):
                buf += chunk
                s = buf.find(b"\xff\xd8")   # JPEG start
                e = buf.find(b"\xff\xd9")   # JPEG end
                if s != -1 and e > s:
                    jpg   = buf[s : e + 2]
                    buf   = buf[e + 2 :]
                    frame = cv2.imdecode(
                        np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR
                    )
                    if frame is not None:
                        with _raw_lock:
                            _raw_frame     = frame
                            _raw_jpg_bytes = jpg
        except Exception as ex:
            print(f"[stream] {ex} — reconnecting in 3 s")
            time.sleep(3)


# ── OpenCV drawing helpers ────────────────────────────────────────────────────
_FONT = cv2.FONT_HERSHEY_SIMPLEX

def _put(frame, text, pos, color=config.COLOR_WHITE, scale=0.65, thick=2):
    cv2.putText(frame, text, pos, _FONT, scale, color, thick)

def _draw_status_bar(frame, sm: DriverStateMachine):
    """Translucent top bar with state label + emotion."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 44), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    labels = {
        DS.NO_FACE:        ("NO FACE DETECTED",                      config.COLOR_BLUE),
        DS.VERIFYING:      (f"Verifying… ({sm.verification_counter}/{config.VERIFY_MAX_FAILS})",
                            config.COLOR_ORANGE),
        DS.VERIFIED:       ("VERIFIED",                              config.COLOR_GREEN),
        DS.UNKNOWN_PERSON: (
            f"UNKNOWN PERSON!  ({sm.countdown}s)" if sm.countdown else "UNKNOWN PERSON!",
            config.COLOR_RED,
        ),
    }
    label, color = labels.get(sm.state, ("", config.COLOR_WHITE))
    _put(frame, label, (10, 30), color, scale=0.75, thick=2)

    if sm.emotion:
        _put(frame, f"Emotion: {sm.emotion}", (w - 240, 30), config.COLOR_WHITE)


def _draw_face_box(frame, x, y, w, h, sm: DriverStateMachine):
    """Coloured rectangle + labels around detected face."""
    color_map = {
        DS.VERIFIED:       config.COLOR_GREEN,
        DS.UNKNOWN_PERSON: config.COLOR_RED,
        DS.VERIFYING:      config.COLOR_ORANGE,
        DS.NO_FACE:        config.COLOR_BLUE,
    }
    color = color_map.get(sm.state, config.COLOR_WHITE)
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    # State label above box
    _put(frame, sm.state.replace("_", " "), (x, y - 35), color)

    # Emotion label just above box
    if sm.emotion:
        _put(frame, f"Emotion: {sm.emotion}", (x, y - 10), config.COLOR_WHITE)

    # Drowsy / eyes-closed badge below box
    if sm.eyes_closed_frames > 0:
        fh = frame.shape[0]
        if sm.is_drowsy:
            cv2.rectangle(frame, (0, fh - 48), (320, fh), config.COLOR_DARK_RED, -1)
            _put(frame, "  DANGER — DROWSY!", (6, fh - 14),
                 config.COLOR_WHITE, scale=0.75, thick=2)
        else:
            cv2.rectangle(frame, (0, fh - 48), (310, fh), config.COLOR_DARK_BLUE, -1)
            _put(frame, f"  EYES CLOSED  ({sm.eyes_closed_frames} frames)",
                 (6, fh - 14), config.COLOR_WHITE, scale=0.62)


def _annotate(frame: np.ndarray, sm: DriverStateMachine,
              face_rect=None) -> np.ndarray:
    """Return a copy of frame with all overlays drawn."""
    out = frame.copy()
    if face_rect:
        x, y, w, h = face_rect
        _draw_face_box(out, x, y, w, h, sm)
    _draw_status_bar(out, sm)
    return out


# ── Annotated MJPEG server ────────────────────────────────────────────────────
monitor_app = Flask(__name__)

def _annotated_gen():
    while True:
        with _annotated_lock:
            jpg = _annotated_jpg
        if jpg is None:
            time.sleep(0.05)
            continue
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + jpg +
            b"\r\n"
        )

@monitor_app.route("/stream")
def monitor_stream():
    return Response(
        _annotated_gen(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )

@monitor_app.route("/")
def monitor_index():
    host = "your-pi-ip"
    return f"""<!DOCTYPE html>
<html>
<head>
  <title>Driver Monitoring</title>
  <style>
    body {{ background:#111; display:flex; flex-direction:column;
           align-items:center; justify-content:center; height:100vh;
           margin:0; color:#eee; font-family:monospace; }}
    img  {{ border:2px solid #0f0; border-radius:6px; max-width:95vw; }}
    h2   {{ color:#0f0; margin-bottom:14px; }}
  </style>
</head>
<body>
  <h2>🚗 Driver Monitoring System</h2>
  <img src="/stream"/>
</body>
</html>"""

def _run_monitor_server():
    monitor_app.run(
        host="0.0.0.0",
        port=config.MONITOR_PORT,
        threaded=True,
        use_reloader=False,
    )


# ── Bridge helpers ────────────────────────────────────────────────────────────
def _send_command(sm: DriverStateMachine) -> None:
    """POST current state to bridge for hardware actuation."""
    # Build LCD lines
    if sm.is_drowsy:
        lcd1, lcd2 = "Danger!!!", "DROWSY ALERT"
    elif sm.eyes_closed_frames > 0:
        lcd1, lcd2 = "EYES CLOSED!", ""
    elif sm.state == DS.VERIFIED:
        lcd1 = "VERIFIED"
        lcd2 = f"Emotion:{sm.emotion}" if sm.emotion else ""
    elif sm.state == DS.UNKNOWN_PERSON:
        lcd1, lcd2 = "UNKNOWN PERSON!", f"Alert! {sm.countdown}s"
    elif sm.state == DS.VERIFYING:
        lcd1, lcd2 = "Verifying...", f"{sm.verification_counter}/{config.VERIFY_MAX_FAILS}"
    else:
        lcd1, lcd2 = "NO FACE", ""

    payload = {
        "state":       sm.state,
        "drowsy":      sm.is_drowsy,
        "eyes_closed": sm.eyes_closed_frames > 0,
        "lcd_line1":   lcd1,
        "lcd_line2":   lcd2,
    }
    try:
        requests.post(
            f"http://localhost:{config.BRIDGE_PORT}/command",
            json=payload,
            timeout=1,
        )
    except Exception as e:
        print(f"[cmd] {e}")


def _get_ir() -> bool:
    try:
        r = requests.get(
            f"http://localhost:{config.BRIDGE_PORT}/ir_status", timeout=1
        )
        return r.json().get("is_active", False)
    except:
        return False


def _wait_for_bridge(timeout: int = 30) -> None:
    print("[main] Waiting for bridge …")
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if requests.get(
                f"http://localhost:{config.BRIDGE_PORT}/health", timeout=2
            ).status_code == 200:
                print("[main] Bridge is up.")
                return
        except:
            pass
        time.sleep(1)
    raise RuntimeError("Bridge did not respond within timeout.")


# ── Main processing loop ──────────────────────────────────────────────────────
def main():
    _wait_for_bridge()

    # Start raw stream reader
    threading.Thread(target=_stream_reader, daemon=True).start()

    # Start annotated MJPEG server
    threading.Thread(target=_run_monitor_server, daemon=True).start()
    print(f"[main] Monitor stream → http://<pi-ip>:{config.MONITOR_PORT}/")

    sm          = DriverStateMachine()
    frame_count = 0
    ir_gate_done = False

    while True:
        # ── IR gate: wait for person before starting ──────────────────────────
        if not ir_gate_done and not _get_ir():
            _send_command(sm)
            while not _get_ir():
                time.sleep(0.5)
            ir_gate_done = True
            print("[main] Person detected — starting monitoring.")

        # ── Grab latest raw frame ─────────────────────────────────────────────
        with _raw_lock:
            frame = _raw_frame.copy() if _raw_frame is not None else None
            jpg   = _raw_jpg_bytes

        if frame is None:
            time.sleep(0.05)
            continue

        frame_count  += 1
        current_time  = time.time()
        gray          = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces         = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=4
        )
        face_rect = None

        if len(faces) > 0:
            x, y, w, h = faces[0]
            face_rect   = (x, y, w, h)
            face_roi    = frame[y : y + h, x : x + w]
            gray_face   = gray[y : y + h, x : x + w]

            # ── State transitions ─────────────────────────────────────────────
            if sm.state == DS.NO_FACE:
                sm.state                = DS.VERIFYING
                sm.verification_counter = 0

            elif sm.state == DS.VERIFYING:
                # Throttle: only verify every N frames
                if frame_count % config.AI_EVERY_N_FRAMES == 0:
                    if owner_mean is not None:
                        try:
                            face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                            emb = DeepFace.represent(
                                face_rgb,
                                model_name=config.FACENET_MODEL,
                                enforce_detection=False,
                            )[0]["embedding"]
                            sim = np.dot(emb, owner_mean) / (
                                np.linalg.norm(emb) * np.linalg.norm(owner_mean)
                            )
                            if sim >= config.VERIFY_THRESHOLD:
                                sm.state                = DS.VERIFIED
                                sm.verification_counter = 0
                            else:
                                sm.verification_counter += 1
                                if sm.verification_counter >= config.VERIFY_MAX_FAILS:
                                    sm.state              = DS.UNKNOWN_PERSON
                                    sm.unknown_alert_start = current_time
                        except Exception as e:
                            print(f"[verify] {e}")
                    else:
                        # No owner images — skip straight to verified
                        sm.state = DS.VERIFIED

            elif sm.state == DS.UNKNOWN_PERSON:
                elapsed       = current_time - sm.unknown_alert_start
                sm.countdown  = max(0, int(config.UNKNOWN_ALERT_SEC - elapsed))
                if elapsed >= config.UNKNOWN_ALERT_SEC:
                    sm.state                = DS.VERIFYING
                    sm.verification_counter = 0
                    sm.countdown            = None
                else:
                    # Send photo + ThingSpeak event (non-blocking, rate-limited)
                    if jpg:
                        jpg_b64 = base64.b64encode(jpg).decode("utf-8")
                        _async(connect.send_latest_photo_to_firebase, jpg_b64)
                    _async(connect.write_to_field, 1, 1)

            elif sm.state == DS.VERIFIED:
                if frame_count % config.AI_EVERY_N_FRAMES == 0:
                    try:
                        face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)

                        # Emotion
                        analysis  = DeepFace.analyze(
                            face_rgb,
                            actions=["emotion"],
                            enforce_detection=False,
                            silent=True,
                        )
                        sm.emotion = analysis[0]["dominant_emotion"]
                        if sm.emotion in ("sad", "angry"):
                            _async(connect.write_to_field, 3, 1)

                        # Eye state
                        closed = 0
                        for ec in [left_eye_casc, right_eye_casc]:
                            for (ex, ey, ew, eh) in ec.detectMultiScale(gray_face):
                                eye_img = gray_face[ey : ey + eh, ex : ex + ew]
                                eye_img = (
                                    cv2.resize(eye_img, (52, 52))
                                    .astype("float32") / 255.0
                                )
                                pred = eye_model.predict(
                                    np.expand_dims(eye_img, axis=(0, -1)), verbose=0
                                )[0][0]
                                if pred > 0.5:
                                    closed += 1

                        sm.eyes_closed_frames = (
                            sm.eyes_closed_frames + 1 if closed >= 2 else 0
                        )

                        if sm.is_drowsy:
                            _async(connect.write_to_field, 3, 2)
                        else:
                            _async(connect.write_to_field, 3, 0)

                    except Exception as e:
                        print(f"[analysis] {e}")

        else:
            sm.reset_to_no_face()

        # ── Draw ALL overlays ─────────────────────────────────────────────────
        annotated = _annotate(frame, sm, face_rect)

        # ── Encode and publish to monitor stream ──────────────────────────────
        _, enc = cv2.imencode(
            ".jpg", annotated,
            [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_QUALITY],
        )
        with _annotated_lock:
            _annotated_jpg = enc.tobytes()

        # ── Send hardware command to bridge ───────────────────────────────────
        _send_command(sm)


if __name__ == "__main__":
    main()
