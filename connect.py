import os
import base64
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, db
import thingspeak
import requests
import json
import time


# ===================== Firebase Configuration =====================
# Initialize Firebase only once
if not firebase_admin._apps:
    cred = credentials.Certificate("serviceAccountKey.json")  # Path to your Firebase service account key
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://smart-car-cbf18-default-rtdb.europe-west1.firebasedatabase.app'
    })


# ===================== ThingSpeak Configuration =====================
THINGSPEAK_CHANNEL_ID = "123456"  # Replace with your ThingSpeak channel ID
THINGSPEAK_WRITE_KEY = "YOUR_WRITE"
THINGSPEAK_READ_KEY = "YOUR_READ"


# ===================== Combined Functions =====================
def send_latest_photo_to_firebase(encoded_string):
    """Send a base64-encoded image to Firebase with timestamp"""
    from datetime import datetime


    # Prepare data
    data = {
        "date": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
        "image_base64": encoded_string,
        
    }


    # Push to Firebase under 'pi_photos'
    ref = db.reference('pi_photos')
    new_ref = ref.push(data)
    print("Sent to Firebase with current date/time.")


    return data  # Return the data that was sent


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


def connect_to_thingspeak():
    """
    Dummy connect function (ThingSpeak is REST, so no persistent connection needed).
    """
    print("Connected to ThingSpeak channel", THINGSPEAK_CHANNEL_ID)


def subscribe_to_field2(interval=15):
    """
    Poll ThingSpeak field2 every 'interval' seconds and print its value.
    """
    url = f"https://api.thingspeak.com/channels/{THINGSPEAK_CHANNEL_ID}/fields/2/last.json?api_key={THINGSPEAK_READ_KEY}"
    print("Subscribing to field2 updates (polling)...")
    try:
        while True:
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                field2_value = data.get('field2')
                print(f"field2 value: {field2_value}")
            else:
                print("Failed to fetch field2:", response.status_code)
            time.sleep(interval)
    except KeyboardInterrupt:
        print("Stopped subscribing to field2.")
