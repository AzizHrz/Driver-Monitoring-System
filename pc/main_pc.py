"""
PC mock version — uses a USB webcam (index 0) instead of PiCamera2.
GPIO calls are replaced with print() stubs.
Run this inside the Docker container on your Debian PC.
"""
import cv2
import numpy as np
from deepface import DeepFace
from tensorflow.keras.models import load_model
import time

# ---- Mock GPIO classes ----
class MockLED:
    def __init__(self, pin): self.pin = pin
    def on(self): print(f"[LED {self.pin}] ON")
    def off(self): print(f"[LED {self.pin}] OFF")
    def blink(self, **kwargs): print(f"[LED {self.pin}] BLINK")

class MockBuzzer:
    stop_beeping = False
    def beep(self, *a, **kw): print("[BUZZER] BEEP")
    def cleanup(self): print("[BUZZER] cleanup")

class MockLCD:
    def message(self, msg, line): print(f"[LCD line {line}]: {msg}")
    def clear(self): print("[LCD] cleared")

class MockIR:
    is_active = True  # simulate driver seated

led = MockLED(14)
buz = MockBuzzer()
lightLED = MockLED(23)
blueLED = MockLED(24)
lcd = MockLCD()
ir = MockIR()

# ---- Models ----
eye_model = load_model('eye_state_model.h5')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
left_eye_cascade  = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_lefteye_2splits.xml')
right_eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_righteye_2splits.xml')

# ---- Owner embeddings ----
owner_embeddings = []
for i in range(1, 4):
    img = cv2.imread(f"owner{i}.jpeg")
    if img is not None:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        emb = DeepFace.represent(img_rgb, model_name='Facenet', enforce_detection=False)[0]['embedding']
        owner_embeddings.append(emb)

owner_embedding_mean = np.mean(owner_embeddings, axis=0) if owner_embeddings else None

# ---- Webcam ----
cap = cv2.VideoCapture(0)

class DriverState:
    NO_FACE = 0; VERIFYING = 1; VERIFIED = 2; UNKNOWN_PERSON = 3

current_state = DriverState.NO_FACE
verification_counter = 0
eyes_closed_frames = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        if len(faces) > 0:
            x, y, w, h = faces[0]
            face_roi = frame[y:y+h, x:x+w]
            gray_face = gray[y:y+h, x:x+w]

            if current_state == DriverState.NO_FACE:
                current_state = DriverState.VERIFYING
                verification_counter = 0

            elif current_state == DriverState.VERIFYING and owner_embedding_mean is not None:
                face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                emb = DeepFace.represent(face_rgb, model_name='Facenet', enforce_detection=False)[0]['embedding']
                sim = np.dot(emb, owner_embedding_mean) / (np.linalg.norm(emb) * np.linalg.norm(owner_embedding_mean))
                if sim > 0.5:
                    current_state = DriverState.VERIFIED
                    verification_counter = 0
                else:
                    verification_counter += 1
                    if verification_counter >= 3:
                        current_state = DriverState.UNKNOWN_PERSON

            elif current_state == DriverState.VERIFIED:
                eyes_closed = 0
                for eye_cascade in [left_eye_cascade, right_eye_cascade]:
                    eyes = eye_cascade.detectMultiScale(gray_face)
                    for (ex, ey, ew, eh) in eyes:
                        eye_img = gray_face[ey:ey+eh, ex:ex+ew]
                        eye_img = cv2.resize(eye_img, (52, 52)).astype('float32') / 255.0
                        pred = eye_model.predict(np.expand_dims(eye_img, axis=(0,-1)))[0][0]
                        if pred > 0.5:
                            eyes_closed += 1
                eyes_closed_frames = eyes_closed_frames + 1 if eyes_closed >= 2 else 0
                if eyes_closed_frames > 3:
                    print("[ALERT] DROWSINESS DETECTED")

            color = {DriverState.VERIFIED: (0,255,0),
                     DriverState.UNKNOWN_PERSON: (0,0,255)}.get(current_state, (0,165,255))
            cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
        else:
            current_state = DriverState.NO_FACE
            eyes_closed_frames = 0

        cv2.imshow('Driver Monitor (PC Mock)', frame)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
