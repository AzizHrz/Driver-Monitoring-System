# 🚗 Driver Monitoring System — v2.0 (Containerized)

[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.13-blue)](https://www.python.org)
[![Docker](https://img.shields.io/badge/docker-ARM64%20%7C%20x86__64-2496ED?logo=docker)](https://www.docker.com)
[![Platform](https://img.shields.io/badge/platform-Raspberry%20Pi%204-red)](https://www.raspberrypi.com)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15.0-FF6F00?logo=tensorflow)](https://tensorflow.org)

> **v2.0 introduces Docker containerization, a bridge/container architecture, browser-based live stream, and production-grade code quality.** The original bare-metal version is preserved as [v1.0](https://github.com/AzizHrz/Driver-Monitoring-System/releases/tag/v1.0).

---

## Table of Contents

- [What changed in v2.0](#what-changed-in-v20)
- [Architecture](#architecture)
- [Repository structure](#repository-structure)
- [Quick start — build directly on the Pi](#quick-start--build-directly-on-the-pi)
- [Cross-build on PC then deploy to Pi](#cross-build-on-pc-then-deploy-to-pi)
- [PC development mode](#pc-development-mode)
- [Viewing the live stream](#viewing-the-live-stream)
- [Hardware test](#hardware-test)
- [Configuration reference](#configuration-reference)
- [Troubleshooting](#troubleshooting)

---

## What changed in v2.0

| Area | v1.0 | v2.0 |
|---|---|---|
| **Dependencies** | Installed globally on host OS | Isolated in Docker container |
| **Camera access** | Direct in main script | `bridge.py` on host, streamed over HTTP |
| **GPIO** | Direct in main script | `bridge.py` on host, commanded via REST |
| **Display** | `cv2.imshow()` — requires monitor | MJPEG stream at `http://<pi-ip>:5001/` |
| **Firebase** | Initialized every call — crashes | Initialized once, background thread pool |
| **ThingSpeak** | Called every frame — rate-limit failures | Rate-limit guard, fire-and-forget |
| **Blocking** | `sleep()` on main thread | Non-blocking everywhere |
| **Config** | Magic strings scattered in files | Single `config.py` |
| **PC dev** | Not possible | `main_pc.py` + USB webcam mock |

---

## Architecture

The core insight: `picamera2`, `gpiozero`, and `libcamera` are compiled against the **host OS Python 3.13** and cannot run inside a Python 3.11 container. Rather than fighting this, we split responsibilities cleanly:

```
┌────────────────────────────────────────────────────────────────┐
│                      Raspberry Pi 4                            │
│                                                                │
│  HOST (Python 3.13)              DOCKER (Python 3.11)          │
│  ┌─────────────────────┐         ┌──────────────────────────┐  │
│  │     bridge.py       │         │        main.py           │  │
│  │                     │ MJPEG   │                          │  │
│  │  picamera2 ─────────┼────────▶│  stream reader           │  │
│  │  IR sensor ─────────┼────────▶│  face detection          │  │
│  │                     │         │  face verification       │  │
│  │  LED ◀──────────────┼─────────┤  emotion analysis        │  │
│  │  Buzzer ◀───────────┼─────────┤  eye state model         │  │
│  │  LCD ◀──────────────┼─ JSON ──┤  OpenCV overlay          │  │
│  │  lightLED ◀─────────┼─────────┤  Firebase upload         │  │
│  │  blueLED ◀──────────┼─────────┤  ThingSpeak logging      │  │
│  └─────────────────────┘         └──────────┬───────────────┘  │
│                                             │ MJPEG :5001      │
└─────────────────────────────────────────────┼──────────────────┘
                                              ▼
                                       Browser / Phone
                                   http://<pi-ip>:5001/
```

### Communication

| Direction | Protocol | Endpoint | Content |
|---|---|---|---|
| bridge → container | HTTP MJPEG | `GET :5000/stream` | Raw camera frames |
| bridge → container | HTTP JSON | `GET :5000/ir_status` | IR sensor state |
| container → bridge | HTTP JSON | `POST :5000/command` | AI results + LCD text |
| container → browser | HTTP MJPEG | `GET :5001/stream` | Annotated frames |

---

## Repository structure

```
Driver-Monitoring-System/
│
├── host/                    ← runs on Pi HOST (Python 3.13, no Docker)
│   ├── bridge.py
│   ├── LCD.py
│   ├── buzzer.py
│   └── test_hardware.py
│
├── container/               ← runs INSIDE Docker container (Python 3.11)
│   ├── main.py
│   ├── debugmain.py
│   ├── connect.py
│   └── config.py
│
├── pc/                      ← PC development mock (no Pi hardware)
│   ├── main_pc.py
│   └── Dockerfile.pc
│
├── docker/                  ← Docker build files
│   ├── Dockerfile.pi
│   └── requirements_pi.txt
│   └── requirements.txt
│
├── models/                  ← AI models and data files
│   ├── eye_state_model.h5
│   ├── embeddings.txt
│   ├── haarcascade_frontalface_default.xml
│   └── shape_predictor_68_face_landmarks.dat
│
├── .gitignore
└── README.md
```
---

## Quick start — build directly on the Pi

> **Recommended if you have a Pi 4 with 4 GB RAM.** We confirmed this works with 2 GB RAM too — it just takes longer.

### Prerequisites

```bash
# On the Pi
sudo apt-get update
sudo apt-get install -y docker.io python3-flask python3-picamera2 \
    python3-gpiozero python3-smbus i2c-tools
sudo usermod -aG docker $USER
# Log out and back in, then verify:
docker run hello-world
```

### Step 1 — Clone the repo

```bash
git clone https://github.com/AzizHrz/Driver-Monitoring-System.git
cd Driver-Monitoring-System
```

### Step 2 — Add credentials

```bash
# Place your Firebase service account file:
cp /path/to/serviceAccountKey.json .

# Edit config.py and set your ThingSpeak keys:
nano config.py
# → THINGSPEAK_CHANNEL_ID, THINGSPEAK_WRITE_KEY, THINGSPEAK_READ_KEY
```

### Step 3 — Add owner photos

```bash
cp /path/to/your/photo1.jpg owner1.jpeg
cp /path/to/your/photo2.jpg owner2.jpeg
cp /path/to/your/photo3.jpg owner3.jpeg
```

### Step 4 — Build the Docker image

```bash
docker build -f Dockerfile.pi -t driver-monitor:pi .
```

> This installs TensorFlow 2.15, DeepFace, dlib-bin, and OpenCV. Expect **20–40 minutes** on a Pi 4. Subsequent rebuilds are fast (Docker layer cache).

### Step 5 — Install host dependencies

```bash
pip3 install flask --break-system-packages
```

### Step 6 — Run

**Terminal 1 — bridge (host):**
```bash
python3 bridge.py
```

**Terminal 2 — container:**
```bash
docker run --rm -it --privileged \
  --device=/dev/i2c-1:/dev/i2c-1 \
  -v $(pwd):/app \
  -p 5001:5001 \
  --network host \
  driver-monitor:pi python3 main.py
```

### Step 7 — Open the stream

On any device on the same network:
```
http://<your-pi-ip>:5001/
```

Find your Pi IP:
```bash
hostname -I
```

---

## Cross-build on PC then deploy to Pi

> **Use this if your Pi has 2 GB RAM and builds are too slow or fail**, or if you want to iterate quickly on the AI code from your PC.

### Why cross-build?

Building TensorFlow and dlib on a Pi 4 with 2 GB RAM takes 30–60 minutes and can run out of memory. Building on a PC with QEMU ARM64 emulation takes the same time but uses your PC's RAM (8–16 GB typically) — much more reliable.

### Prerequisites (PC — Linux/Mac/WSL)

```bash
# Install Docker with buildx support
docker buildx version   # should print buildx version

# Enable QEMU ARM64 emulation
docker run --privileged --rm tonistiigi/binfmt --install arm64
# Verify:
ls /proc/sys/fs/binfmt_misc/ | grep aarch64
```

### Step 1 — Set up a buildx builder

```bash
docker buildx create --use --name pi-builder --platform linux/arm64
docker buildx inspect --bootstrap
```

### Step 2 — Build the ARM64 image

```bash
# Clone on PC
git clone https://github.com/AzizHrz/Driver-Monitoring-System.git
cd Driver-Monitoring-System

# Build — output to local Docker daemon as a .tar
docker buildx build \
  --platform linux/arm64 \
  -f Dockerfile.pi \
  -t driver-monitor:pi \
  --output type=docker \
  .
```

> Expect **30–60 minutes** on first build. The dlib C++ compilation is the longest step.
> **Important**: use `dlib-bin` in `requirements_pi.txt`, not `dlib` — see [Troubleshooting](#troubleshooting).

### Step 3 — Export the image

```bash
docker save driver-monitor:pi | gzip > driver-monitor-pi.tar.gz
ls -lh driver-monitor-pi.tar.gz   # expect ~3–5 GB
```

### Step 4 — Transfer to Pi

```bash
# Replace with your Pi's IP
rsync -avz --progress driver-monitor-pi.tar.gz rahma-pi@172.20.10.14:~/
```

Or with scp:
```bash
scp driver-monitor-pi.tar.gz rahma-pi@172.20.10.14:~/
```

### Step 5 — Load on Pi

```bash
# SSH into Pi
ssh rahma-pi@172.20.10.14

# Load image
docker load < driver-monitor-pi.tar.gz

# Verify
docker images | grep driver-monitor
```

### Step 6 — Run (same as above)

Follow Steps 2–7 from [Quick start](#quick-start--build-directly-on-the-pi).

---

## PC development mode

> Develop and test the AI pipeline on your PC — no Pi hardware needed.

```bash
# Build PC image (x86_64)
docker build -f Dockerfile.pc -t driver-monitor:pc .

# Run with USB webcam
docker run --rm -it \
  --device=/dev/video0:/dev/video0 \
  -v $(pwd):/app \
  -p 5001:5001 \
  driver-monitor:pc python3 main_pc.py
```

`main_pc.py` replaces:
- `picamera2` → `cv2.VideoCapture(0)` (USB webcam)
- `gpiozero` → stub classes that print to console instead of driving GPIO
- `bridge.py` → not needed, runs as a single process

---

## Viewing the live stream

Once the system is running, open in any browser:

```
http://<pi-ip>:5001/          ← full page with annotated video
http://<pi-ip>:5001/stream    ← raw MJPEG (works in VLC too)
http://<pi-ip>:5001/status    ← JSON health check
```

The annotated stream shows:
- **Coloured face bounding box**: green=verified, orange=verifying, red=unknown, blue=no face
- **State label** above the face box
- **Emotion label** (happy, sad, angry, neutral…)
- **Drowsy / eyes closed badge** at the bottom
- **Translucent status bar** at the top with state + emotion
- **Debug counters** (frames received / annotated) — disable by setting `DEBUG=false`

---

## Hardware test

Test all GPIO components independently — no Docker needed:

```bash
python3 test_hardware.py

# Skip LCD if not wired:
python3 test_hardware.py --skip-lcd
```

Tests: LED blink × 5, lightLED blink × 5, blueLED blink × 5, buzzer beep × 3, LCD two-line message, IR sensor state.

---

## Configuration reference

All constants live in `config.py`. Key ones to change for your setup:

```python
# ThingSpeak (or set as env vars)
THINGSPEAK_CHANNEL_ID = "YOUR_CHANNEL_ID"
THINGSPEAK_WRITE_KEY  = "YOUR_WRITE_KEY"
THINGSPEAK_READ_KEY   = "YOUR_READ_KEY"

# Firebase (or set FIREBASE_CRED env var)
FIREBASE_CRED_PATH = "/app/serviceAccountKey.json"
FIREBASE_DB_URL    = "https://your-project.firebaseio.com"

# GPIO pins (BCM)
PIN_LED       = 14
PIN_BUZZER    = 18
PIN_LIGHT_LED = 23
PIN_BLUE_LED  = 24
PIN_IR        = 25

# AI tuning
VERIFY_THRESHOLD       = 0.5   # cosine similarity for face verification
DROWSY_FRAME_THRESHOLD = 3     # consecutive closed-eye frames → drowsy alert
AI_EVERY_N_FRAMES      = 5     # run DeepFace every N frames (CPU throttle)
```

### Environment variable overrides

```bash
docker run ... \
  -e FIREBASE_CRED=/app/serviceAccountKey.json \
  -e FIREBASE_DB_URL=https://your-project.firebaseio.com \
  -e TS_CHANNEL_ID=123456 \
  -e TS_WRITE_KEY=XXXXXXXX \
  -e TS_READ_KEY=YYYYYYYY \
  -e DEBUG=false \
  driver-monitor:pi python3 main.py
```

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'libcamera._libcamera'`

**Cause**: `_libcamera.cpython-313-aarch64-linux-gnu.so` is compiled for Python 3.13 (host) but the container runs Python 3.11. The `.so` binary is ABI-incompatible — renaming it does not work.

**Fix**: This is exactly why v2.0 uses `bridge.py` on the host. The container never imports `libcamera` or `picamera2`. Make sure you are running `bridge.py` on the host and `main.py` inside the container.

---

### `lto-wrapper: fatal error: c++ terminated with signal 11` (dlib build crash)

**Cause**: Building `dlib` from source under QEMU ARM64 emulation fails at the LTO linker step — it spawns 128 parallel jobs that exhaust emulated memory/stack.

**Fix**: Use `dlib-bin==19.24.6` in `requirements_pi.txt` instead of `dlib`. It is a pre-compiled wheel that skips all C++ compilation.

```
# requirements_pi.txt — use this:
dlib-bin==19.24.6

# NOT this:
dlib==19.24.4
```

---

### `ImportError: libblas.so.3: cannot open shared object file`

**Cause**: `numpy` C extensions need BLAS. The system library is missing on the Pi host.

**Fix**:
```bash
sudo apt-get install -y libopenblas0-pthread libblas3
```

---

### ThingSpeak writes return `0` (failure)

**Cause**: ThingSpeak free tier enforces a minimum 15-second interval between writes per channel. Writing every frame causes all writes after the first to fail silently.

**Fix**: Already handled in v2.0 `connect.py` via `_last_write` rate-limit guard. If you still see failures, check that `THINGSPEAK_WRITE_KEY` is correct in `config.py`.

---

### Firebase: `serviceAccountKey.json not found`

**Cause**: Inside the container, the working directory is `/app`. The credential file must exist at `/app/serviceAccountKey.json` — which means it must be present in the folder you mount as `-v $(pwd):/app`.

**Fix**:
```bash
ls $(pwd)/serviceAccountKey.json   # must exist before running docker run
```

---

### Stream at `:5001` loads but video is blank / frozen

**Checklist**:
1. Is `bridge.py` running on the host?  `curl http://localhost:5000/health`
2. Are frames arriving in the container?  `curl http://localhost:5001/status`  → check `frames_received`
3. Is the annotation loop running?  → check `frames_annotated` in the same response
4. Is the Pi camera enabled?  `sudo raspi-config` → Interface Options → Camera → Enable

---

### `qt.qpa.xcb: could not connect to display`

**Cause**: `cv2.imshow()` was called inside the container — it needs an X11 display which doesn't exist in a headless container.

**Fix**: v2.0 removes all `cv2.imshow()` calls. The annotated output is served via MJPEG on port 5001. If you see this error, you are running an old version of `main.py`.
