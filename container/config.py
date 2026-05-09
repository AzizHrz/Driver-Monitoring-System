# config.py
# ─────────────────────────────────────────────────────────────────────────────
# Single source of truth for every tunable constant.
# Both bridge.py (host) and main.py (container) import from here.
# In production, override sensitive values via environment variables.
# ─────────────────────────────────────────────────────────────────────────────

import os

# ── Bridge (host-side HTTP server) ────────────────────────────────────────────
BRIDGE_HOST = "0.0.0.0"
BRIDGE_PORT = 5000          # raw MJPEG stream + command endpoint

# ── Monitor (container-side annotated stream) ─────────────────────────────────
MONITOR_PORT = 5001         # annotated MJPEG → open in browser

# ── Camera ────────────────────────────────────────────────────────────────────
CAM_RESOLUTION  = (640, 480)
CAM_FRAMERATE   = 20

# ── GPIO (BCM numbering) ──────────────────────────────────────────────────────
PIN_LED        = 14
PIN_BUZZER     = 18
PIN_LIGHT_LED  = 23
PIN_BLUE_LED   = 24
PIN_IR         = 25

# ── I2C LCD ───────────────────────────────────────────────────────────────────
LCD_BUS     = 2
LCD_ADDR    = 0x27
LCD_BACKLIT = True

# ── AI / model paths ──────────────────────────────────────────────────────────
EYE_MODEL_PATH   = "/app/eye_state_model.h5"
OWNER_IMG_PATHS  = ["/app/owner1.jpg", "/app/owner2.jpg", "/app/owner3.jpg"]
FACENET_MODEL    = "Facenet"
VERIFY_THRESHOLD = 0.5      # cosine similarity ≥ this → owner verified
VERIFY_MAX_FAILS = 3        # consecutive failures → UNKNOWN_PERSON
UNKNOWN_ALERT_SEC = 5       # seconds to show unknown alert before re-verify
DROWSY_FRAME_THRESHOLD = 3  # consecutive closed-eye frames → drowsy
AI_EVERY_N_FRAMES = 5       # run DeepFace every N frames (throttle)

# ── ThingSpeak ────────────────────────────────────────────────────────────────
# Fields:
#   field1 → unknown person detected   (1 = event)
#   field2 → reserved / external input (polled by subscribe_to_field2)
#   field3 → driver state              (0=ok, 1=sad/angry, 2=drowsy)
THINGSPEAK_CHANNEL_ID = os.getenv("TS_CHANNEL_ID", "YOUR_CHANNEL_ID")
THINGSPEAK_WRITE_KEY  = os.getenv("TS_WRITE_KEY",  "YOUR_WRITE_KEY")
THINGSPEAK_READ_KEY   = os.getenv("TS_READ_KEY",   "YOUR_READ_KEY")
THINGSPEAK_POLL_SEC   = 15      # subscribe_to_field2 polling interval

# ── Firebase ──────────────────────────────────────────────────────────────────
# serviceAccountKey.json must be present at this path inside the container.
FIREBASE_CRED_PATH   = os.getenv("FIREBASE_CRED", "/app/serviceAccountKey.json")
FIREBASE_DB_URL      = os.getenv(
    "FIREBASE_DB_URL",
    "https://smart-car-cbf18-default-rtdb.europe-west1.firebasedatabase.app"
)
FIREBASE_PHOTO_NODE  = "pi_photos"   # Realtime DB node for intruder snapshots

# ── OpenCV drawing ────────────────────────────────────────────────────────────
FONT            = "FONT_HERSHEY_SIMPLEX"   # resolved at runtime
COLOR_WHITE     = (255, 255, 255)
COLOR_GREEN     = (0,   255,   0)
COLOR_RED       = (0,     0, 255)
COLOR_ORANGE    = (0,   165, 255)
COLOR_BLUE      = (255, 165,   0)
COLOR_DARK_RED  = (0,     0, 180)
COLOR_DARK_BLUE = (180, 100,   0)
JPEG_QUALITY    = 80
