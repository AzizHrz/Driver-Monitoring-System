import cv2
import numpy as np
from deepface import DeepFace
from tensorflow.keras.models import load_model
from picamera2 import Picamera2
import time
import os

# Constants
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
TARGET_FPS = 3
FACE_REC_INTERVAL = 1  # seconds
EMOTION_INTERVAL = 0.5    # second

# Load models
eye_model = load_model('eye_state_model.h5')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
left_eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_lefteye_2splits.xml')
right_eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_righteye_2splits.xml')



# Load owner embeddings
face_reference = cv2.imread('owner1.jpg', cv2.IMREAD_GRAYSCALE)



# Initialize camera
picam2 = Picamera2()
config = picam2.create_video_configuration(main={"size": (FRAME_WIDTH, FRAME_HEIGHT)})
picam2.configure(config)
picam2.start()

# Timing control
frame_interval = 1.0 / TARGET_FPS
last_frame_time = time.time()
last_face_rec_time = 0
last_emotion_time = 0

is_owner = False
FaceDetectionPeriod=0  
try:
    while True:
        if not is_owner:
        # Strict 1 FPS control
            # current_time = time.time()
            # elapsed = current_time - last_frame_time
            # if elapsed < frame_interval:
            #     time.sleep(frame_interval - elapsed)
            # last_frame_time = time.time()

            # Capture frame
            frame = picam2.capture_array()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Face detection
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

            if len(faces) > 0:
                x, y, w, h = faces[0]
                face_roi = frame[y:y+h, x:x+w]
                gray_face = gray[y:y+h, x:x+w]

                # Face recognition (every 10 seconds)
                # if current_time - last_face_rec_time > FACE_REC_INTERVAL:
                face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                try:
                    similarity_score = cv2.matchTemplate(gray[y:y+h, x:x+w], face_reference, cv2.TM_CCOEFF_NORMED)
                    if similarity_score.any() >= 0.8:
                        print("face recognized")
                        is_owner = True
                    else: 
                        print("face not recognized")
                    # last_face_rec_time = current_time
                except Exception as e:
                    print(f"Face recognition error: {e}")
                    is_owner = False
                # else:
                #     is_owner = False  # Not checked this frame

        elif is_owner:
            # Strict 1 FPS control
            frame = picam2.capture_array()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Face detection
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

            # current_time = time.time()
            # elapsed = current_time - last_frame_time
            # if elapsed < frame_interval:
            #     time.sleep(frame_interval - elapsed)
            # last_frame_time = time.time()

            # Capture frame

            if len(faces) == 0:
                is_owner = False 
                print ("Face Detection Failed ...") 
            else:
                x, y, w, h = faces[0]
                face_roi = frame[y:y+h, x:x+w]
                gray_face = gray[y:y+h, x:x+w]

                    # Emotion analysis (every 1 second)
                # if current_time - last_emotion_time > EMOTION_INTERVAL:
                face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                try:
                    analysis = DeepFace.analyze(face_rgb, actions=['emotion'], enforce_detection=False)
                    emotion = analysis[0]['dominant_emotion']
                    # last_emotion_time = current_time
                except Exception as e:
                    print(f"Emotion analysis error: {e}")
                    emotion = None

                # Eye state detection (every frame)
                eyes_closed = 0
                left_eyes = left_eye_cascade.detectMultiScale(gray_face)
                for (ex, ey, ew, eh) in left_eyes:
                    eye_img = gray_face[ey:ey+eh, ex:ex+ew]
                    eye_img = cv2.resize(eye_img, (52, 52)).astype('float32') / 255.0
                    pred = eye_model.predict(np.expand_dims(eye_img, axis=(0, -1)))[0][0]
                    if pred < 0.5:
                        eyes_closed += 1
                right_eyes = right_eye_cascade.detectMultiScale(gray_face)
                for (ex, ey, ew, eh) in right_eyes:
                    eye_img = gray_face[ey:ey+eh, ex:ex+ew]
                    eye_img = cv2.resize(eye_img, (52, 52)).astype('float32') / 255.0
                    pred = eye_model.predict(np.expand_dims(eye_img, axis=(0, -1)))[0][0]
                    if pred < 0.5:
                        eyes_closed += 1
                # Draw results
                color = (0, 255, 0) if is_owner else (0, 0, 255) if is_owner is False else (255, 255, 0)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

                if emotion:
                    cv2.putText(frame, f"Emotion: {emotion}", (x, y-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                if eyes_closed >= 2:
                    cv2.putText(frame, "EYES CLOSED!", (x, y+h+30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Display frame
        cv2.imshow('Driver Monitoring', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


finally:
    picam2.stop()
    cv2.destroyAllWindows()
