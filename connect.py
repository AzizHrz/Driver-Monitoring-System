# connect.py
# ─────────────────────────────────────────────────────────────────────────────
# Responsibility: all external cloud I/O (Firebase + ThingSpeak).
#
# FIXES vs original:
#   1. Firebase initialised ONCE via a module-level guard — calling
#      send_latest_photo_to_firebase() from a tight loop no longer raises
#      "app already exists".
#   2. serviceAccountKey.json path comes from config (env-overridable) so
#      the container can find it at /app/serviceAccountKey.json instead of
#      relying on the cwd.
#   3. All network calls are wrapped in try/except — a transient failure no
#      longer crashes the main loop.
#   4. write_to_field now lives ONLY here (it was duplicated in main.py).
#   5. subscribe_to_field2 is a generator so callers control the loop.
#   6. ThingSpeak rate-limit: the free tier allows 1 write per 15 s.
#      A simple last-write timestamp guard prevents flooding.
# ─────────────────────────────────────────────────────────────────────────────

import os
import time
import requests
from datetime import datetime

import firebase_admin
from firebase_admin import credentials, db

import config

# ── Firebase — initialise exactly once ───────────────────────────────────────
_firebase_ready = False

def _init_firebase() -> bool:
    global _firebase_ready
    if _firebase_ready:
        return True
    if firebase_admin._apps:          # already initialised (e.g. re-import)
        _firebase_ready = True
        return True
    cred_path = config.FIREBASE_CRED_PATH
    if not os.path.exists(cred_path):
        print(f"[firebase] ERROR: credential file not found → {cred_path}")
        print("[firebase] Place serviceAccountKey.json at that path and restart.")
        return False
    try:
        cred = credentials.Certificate(cred_path)
        firebase_admin.initialize_app(cred, {"databaseURL": config.FIREBASE_DB_URL})
        _firebase_ready = True
        print("[firebase] Initialised OK.")
        return True
    except Exception as e:
        print(f"[firebase] Init failed: {e}")
        return False

# Attempt init at import time so any path error is visible early.
_init_firebase()


# ── Firebase writes ───────────────────────────────────────────────────────────

def send_latest_photo_to_firebase(encoded_string: str) -> bool:
    """
    Push a base64-encoded JPEG snapshot to Firebase Realtime Database.
    Returns True on success, False on any failure.
    """
    if not _firebase_ready and not _init_firebase():
        print("[firebase] Skipping photo upload — not initialised.")
        return False
    try:
        payload = {
            "date":         datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
            "image_base64": encoded_string,
        }
        ref = db.reference(config.FIREBASE_PHOTO_NODE)
        ref.push(payload)
        print("[firebase] Snapshot pushed OK.")
        return True
    except Exception as e:
        print(f"[firebase] Upload error: {e}")
        return False


# ── ThingSpeak ────────────────────────────────────────────────────────────────
# Rate-limit guard: track last successful write time per field.
_last_write: dict[int, float] = {}
_MIN_WRITE_INTERVAL = 15.0   # seconds — free-tier limit

def write_to_field(field_number: int, value) -> bool:
    """
    Write a value to a ThingSpeak field (1–8).
    Silently skips if called faster than the free-tier rate limit.
    Returns True on success.

    Field mapping (defined in config.py):
        field1 → unknown person event  (value = 1)
        field2 → reserved
        field3 → driver state          (0=ok, 1=negative-emotion, 2=drowsy)
    """
    now = time.monotonic()
    if now - _last_write.get(field_number, 0) < _MIN_WRITE_INTERVAL:
        return False   # rate-limited, silently skip

    url = (
        f"https://api.thingspeak.com/update"
        f"?api_key={config.THINGSPEAK_WRITE_KEY}"
        f"&field{field_number}={value}"
    )
    try:
        resp = requests.get(url, timeout=5)
        if resp.text.strip() == "0":
            print(f"[thingspeak] field{field_number}={value} → FAILED (check key/channel)")
            return False
        _last_write[field_number] = now
        print(f"[thingspeak] field{field_number}={value} → entry {resp.text.strip()}")
        return True
    except requests.RequestException as e:
        print(f"[thingspeak] Network error: {e}")
        return False


def subscribe_to_field2():
    """
    Generator that yields the latest field2 value from ThingSpeak on each
    poll. Caller controls the loop; stops cleanly on KeyboardInterrupt.

    Usage:
        for value in connect.subscribe_to_field2():
            do_something(value)
    """
    url = (
        f"https://api.thingspeak.com/channels/{config.THINGSPEAK_CHANNEL_ID}"
        f"/fields/2/last.json?api_key={config.THINGSPEAK_READ_KEY}"
    )
    print("[thingspeak] Polling field2 ...")
    while True:
        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                value = resp.json().get("field2")
                print(f"[thingspeak] field2 = {value}")
                yield value
            else:
                print(f"[thingspeak] HTTP {resp.status_code}")
                yield None
        except requests.RequestException as e:
            print(f"[thingspeak] Poll error: {e}")
            yield None
        time.sleep(config.THINGSPEAK_POLL_SEC)
