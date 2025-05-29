

import cv2
import base64
import numpy as np
from deepface import DeepFace
from tensorflow.keras.models import load_model
from picamera2 import Picamera2
import time
import os
import connect
import thingspeak
import requests


# --- ThingSpeak Functions (moved from connect.py) ---
THINGSPEAK_CHANNEL_ID = "YOUR_CHANNEL_ID"  # Replace with your actual channel ID
THINGSPEAK_WRITE_KEY = "YOUR_WRITE"
THINGSPEAK_READ_KEY = "YOUR_READ"  # Replace with your actual read key


def connect_to_thingspeak():
    print("Connected to ThingSpeak channel", THINGSPEAK_CHANNEL_ID)








def write_to_field(field_number, value):
    """
    Write a value to a specific ThingSpeak field.
    :param field_number: int (1-8)
    :param value: int or str
    """
    url = f"https://api.thingspeak.com/update?api_key={THINGSPEAK_WRITE_KEY}&field{field_number}={value}"
    response = requests.get(url)
    if response.text == '0':
        print(f"Failed to write to field{field_number}")
    else:
        print(f"Wrote value {value} to field{field_number}, Entry ID: {response.text}")


# --- End ThingSpeak Functions ---




from gpiozero import LED
from buzzer import Buzzer
from signal import pause
from LCD import LCD




from gpiozero import DigitalInputDevice, LED




from time import sleep








ir = DigitalInputDevice(25, pull_up=True)








detect = False
begin = False








def blink_until_detect():
    while (not ir.is_active):
        led.on()
        sleep(1)
        led.off()
        sleep(2.5)
        if ir.is_active:
            break




lcd = LCD(2, 0x27, True)




led = LED(14)
buz = Buzzer(pin=18)




lightLED = LED(23)
blueLED = LED(24)
# Initialize models
eye_model = load_model('eye_state_model.h5')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
left_eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_lefteye_2splits.xml')
right_eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_righteye_2splits.xml')




# Load owner embeddings
owner_embeddings = []
for i in range(1, 4):
    img = cv2.imread(f"owner{i}.jpg")
    if img is not None:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        embedding = DeepFace.represent(img_rgb, model_name='Facenet', enforce_detection=False)[0]['embedding']
        owner_embeddings.append(embedding)




owner_embedding_mean = np.mean(owner_embeddings, axis=0) if owner_embeddings else None




# Camera setup
picam2 = Picamera2()
config = picam2.create_video_configuration(main={"size": (640, 480)})
picam2.configure(config)
picam2.start()




# State machine
class DriverState:
    NO_FACE = 0
    VERIFYING = 1
    VERIFIED = 2
    UNKNOWN_PERSON = 3




current_state = DriverState.NO_FACE
verification_counter = 0
eyes_closed_frames = 0
unknown_alert_start_time = 0




def mainfunction():
    global current_state, verification_counter, eyes_closed_frames, unknown_alert_start_time, begin
    try:
        while True:
            if (not ir.is_active) & (not begin):
                begin = True
                print("No person detected")
                blink_until_detect()
                print("peron deteted")
            else:
                led.off()
                current_time = time.time()
                frame = picam2.capture_array()
                _, buffer = cv2.imencode('.jpg', frame)
                jpg_as_text = base64.b64encode(buffer).decode('utf-8')
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
               
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)


                if len(faces) > 0:
                    x, y, w, h = faces[0]
                    face_roi = frame[y:y+h, x:x+w]
                    gray_face = gray[y:y+h, x:x+w]


                    # State machine transitions
                    if current_state == DriverState.NO_FACE:
                        current_state = DriverState.VERIFYING
                        verification_counter = 0


                    elif current_state == DriverState.VERIFYING:
                        try:
                            face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                            embedding = DeepFace.represent(face_rgb, model_name='Facenet', enforce_detection=False)[0]['embedding']
                            similarity = np.dot(embedding, owner_embedding_mean) / (np.linalg.norm(embedding) * np.linalg.norm(owner_embedding_mean))
                            if similarity > 0.5:
                                current_state = DriverState.VERIFIED
                                verification_counter = 0
                            else:
                                verification_counter += 1
                                if verification_counter >= 3:
                                    current_state = DriverState.UNKNOWN_PERSON
                                    unknown_alert_start_time = current_time
                        except Exception as e:
                            print(f"Recognition error: {e}")


                    elif current_state == DriverState.UNKNOWN_PERSON:
                        # Show alert for 5 seconds
                        alert_elapsed = current_time - unknown_alert_start_time
                        if alert_elapsed >= 5:
                            current_state = DriverState.VERIFYING
                            verification_counter = 0
                            led.off()
                        else:
                            # Visual countdown
                            countdown = 5 - int(alert_elapsed)
                            cv2.putText(frame, f"ALERT: UNKNOWN PERSON! ({countdown}s)", (50, 50),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                            # Here you would add your alert sound/notification


                    elif current_state == DriverState.VERIFIED:
                        try:
                            # Emotion analysis
                            face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                            analysis = DeepFace.analyze(face_rgb, actions=['emotion'], enforce_detection=False)
                            emotion = analysis[0]['dominant_emotion']
                            if(emotion == "sad" or emotion == "angry"):
                                write_to_field(3,1)


                            # Eye detection
                            eyes_closed = 0
                            for eye_cascade in [left_eye_cascade, right_eye_cascade]:
                                eyes = eye_cascade.detectMultiScale(gray_face)
                                for (ex, ey, ew, eh) in eyes:
                                    eye_img = gray_face[ey:ey+eh, ex:ex+ew]
                                    eye_img = cv2.resize(eye_img, (52, 52)).astype('float32') / 255.0
                                    pred = eye_model.predict(np.expand_dims(eye_img, axis=(0, -1)))[0][0]
                                    if pred > 0.5:
                                        eyes_closed += 1


                            eyes_closed_frames = eyes_closed_frames + 1 if eyes_closed >= 2 else 0


                            # Draw monitoring UI
                            cv2.putText(frame, f"Emotion: {emotion}", (x, y-10),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                            if eyes_closed_frames > 0:
                                cv2.putText(frame, f"EYES CLOSED ({eyes_closed_frames})", (x, y+h+30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                                sleep = eyes_closed_frames > 3
                                if sleep:
                                    cv2.putText(frame, f"Danger", (x, y+h+40),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                                    led.blink(on_time=3, off_time=1, n=None, background=True)
                                    buz.stop_beeping=True
                                    buz.beep(0.1,0.1,1)
                                    lcd.message("", 1)
                                    lcd.message("Danger!!!", 2)
                                    write_to_field(3,2)
                                else:
                                    write_to_field(3,0)
                                    led.off()
                                    buz.stop_beeping=False
                                    lcd.message("", 1)
                                    lcd.message("EYES CLOSED!", 2)


                        except Exception as e:
                            print(f"Analysis error: {e}")


                    # Draw face rectangle based on state
                    if current_state == DriverState.VERIFIED:
                        color = (0, 255, 0)  # Green
                        status = "VERIFIED"
                        lcd.message("VERIFIED", 1)
                        lcd.message("", 2)
                        led.off()
                        blueLED.off()
                        lightLED.on()


                        


                    elif current_state == DriverState.UNKNOWN_PERSON:
                        color = (0, 0, 255)  # Red
                        status = "UNKNOWN PERSON!"
                        lcd.message("UNKNOWN PERSON!", 1)
                        lcd.message("", 2)
                        led.blink(on_time=3, off_time=1, n=None, background=True)
                        buz.stop_beeping=True
                        buz.beep(0.1,0.1,1)
                        blueLED.off()
                        lightLED.off()
                        write_to_field(1,1)
                        connect.send_latest_photo_to_firebase(jpg_as_text)
                    else:
                        color = (0, 165, 255)  # Orange
                        status = f"Verifying... ({verification_counter}/3)"
                        lcd.message("Verifying...", 1)
                        lcd.message("", 2)
                        led.off()
                        blueLED.on()
                        lightLED.off()


                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(frame, status, (x, y-35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


                else:
                    current_state = DriverState.NO_FACE
                    verification_counter = 0
                    eyes_closed_frames = 0
                    cv2.putText(frame, "NO FACE DETECTED", (50, 50),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    lcd.message("NO FACE DETECTED", 1)
                    lcd.message("", 2)
                    led.off()
                    blueLED.blink(on_time=1, off_time=1, n=2, background=True)
                    lightLED.off()


                cv2.imshow('Driver Monitoring', frame)
                if cv2.waitKey(5) & 0xFF == ord('q'):
                    break


    finally:
        buz.cleanup()
        picam2.stop()
        cv2.destroyAllWindows()


connect_to_thingspeak()
def subscribe_to_field2(callback=None, interval=5):
    """
    Poll ThingSpeak field2 every 'interval' seconds and print its value.
    """
    url = f"https://api.thingspeak.com/channels/{THINGSPEAK_CHANNEL_ID}/fields/2/last.json?api_key={THINGSPEAK_READ_KEY}"
    print("Subscribing to field2 updates (polling)...")
    try:
        while True:
            print("Subscribing to field2 updates (polling)...")
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                field2_value = data.get('field2')
                print(f"field2 value: {field2_value}")
                mainfunction()  # Call the main function to process the data
            else:
                print("Failed to fetch field2:", response.status_code)
            time.sleep(interval)
    except KeyboardInterrupt:
        print("Stopped subscribing to field2.")
subscribe_to_field2()
