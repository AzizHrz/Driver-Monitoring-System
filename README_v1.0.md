# 🚗 Driver Monitoring System — v1.0 (Bare Metal)

> **This is the original release.** It runs directly on a Raspberry Pi 4 without Docker or containerization. All dependencies are installed on the host OS.

---

## What it does

A real-time driver safety system running on a Raspberry Pi 4 with a camera module. It continuously monitors the driver and reacts to dangerous conditions:

| Feature | Description |
|---|---|
| **Face Verification** | Recognizes the authorized driver using FaceNet embeddings |
| **Drowsiness Detection** | Detects closed eyes using a custom Keras model (`eye_state_model.h5`) |
| **Emotion Analysis** | Detects stress/anger states via DeepFace |
| **Intruder Alert** | Sends a photo to Firebase if an unknown person is detected |
| **IoT Logging** | Reports driver state to ThingSpeak (fields 1–3) |
| **Hardware Alerts** | Drives buzzer, LEDs, and I²C LCD display via GPIO |

---

## Hardware

- Raspberry Pi 4 (2 GB or 4 GB)
- Raspberry Pi Camera Module v2
- I²C LCD display (16x2, address `0x27`)
- LEDs: red (GPIO 14), green/light (GPIO 23), blue (GPIO 24)
- Buzzer (GPIO 18)
- IR proximity sensor (GPIO 25)

---

## Repository structure

```
Driver-Monitoring-System/
├── main.py                              # Main entry point
├── connect.py                           # Firebase + ThingSpeak
├── LCD.py                               # I²C LCD driver
├── embeddings.txt                       # Pre-computed owner face embeddings
├── eye_state_model.h5                   # Trained Keras eye-state classifier
├── haarcascade_frontalface_default.xml  # OpenCV face detector
├── shape_predictor_68_face_landmarks.dat# dlib landmark model
└── onethrea.py                          # Utility threading helper
```

---

## Installation

### 1. Install system dependencies

```bash
sudo apt-get update
sudo apt-get install -y python3-pip python3-picamera2 python3-opencv \
    python3-gpiozero libopenblas-dev liblapack-dev
```

### 2. Install Python dependencies

```bash
pip3 install deepface==0.0.91 tensorflow==2.15.0 tf_keras==2.15.0 \
    firebase-admin==6.5.0 requests==2.31.0 dlib-bin==19.24.6
```

### 3. Configure credentials

Place your Firebase service account file at:
```
serviceAccountKey.json
```

Edit `connect.py` and set your ThingSpeak channel ID, write key, and read key.

### 4. Prepare owner photos

Place 3 photos of the authorized driver as:
```
owner1.jpeg  owner2.jpeg  owner3.jpeg
```

### 5. Run

```bash
python3 main.py
```

---

## Known limitations of this version

- Runs only on the Raspberry Pi — no way to develop or test on a PC
- All dependencies installed globally on the host OS — version conflicts possible
- `cv2.imshow()` requires a connected monitor — no remote viewing
- No separation between hardware and AI logic — hard to maintain

> These limitations are addressed in [v2.0](https://github.com/AzizHrz/Driver-Monitoring-System/releases/tag/v2.0).
