# buzzer.py
from gpiozero import TonalBuzzer, PWMOutputDevice
from time import sleep
import threading

class Buzzer:
    def __init__(self, pin=18):
        self.pin = pin
        self.stop_beeping = False
        try:
            self._buzzer = PWMOutputDevice(pin)
        except Exception:
            self._buzzer = None  # Mock for non-Pi environments

    def beep(self, on_time=0.1, off_time=0.1, n=1):
        if self._buzzer is None:
            print(f"[MOCK BUZZER] beep(on={on_time}, off={off_time}, n={n})")
            return
        for _ in range(n if n else 1):
            if self.stop_beeping:
                break
            self._buzzer.on()
            sleep(on_time)
            self._buzzer.off()
            sleep(off_time)

    def cleanup(self):
        if self._buzzer:
            self._buzzer.close()
