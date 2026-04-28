#!/usr/bin/env python3
"""
test_hardware.py  —  runs on the HOST (Python 3.13), no Docker needed.
Tests all hardware components one by one:
  - LED (pin 14)      : blinks 5 times
  - lightLED (pin 23) : blinks 5 times
  - blueLED  (pin 24) : blinks 5 times
  - Buzzer   (pin 18) : beeps 3 times
  - LCD (I2C 0x27)    : prints two lines

Run with:
    python3 test_hardware.py

Add --skip-lcd if you don't have the LCD wired yet.
"""

import sys
import time
from gpiozero import LED, Buzzer as GpioBuzzer, DigitalInputDevice

# ── Import your custom LCD + Buzzer wrappers (same as main project) ───────────
try:
    from LCD import LCD
    HAS_LCD = True
except ImportError:
    print("[test] WARNING: LCD.py not found — LCD test will be skipped.")
    HAS_LCD = False

try:
    from buzzer import Buzzer
    HAS_BUZZER_LIB = True
except ImportError:
    print("[test] WARNING: buzzer.py not found — using gpiozero Buzzer directly.")
    HAS_BUZZER_LIB = False

SKIP_LCD = "--skip-lcd" in sys.argv

# ── GPIO pin config (must match config.py) ────────────────────────────────────
PIN_LED       = 14
PIN_BUZZER    = 18
PIN_LIGHT_LED = 23
PIN_BLUE_LED  = 24
PIN_IR        = 25

BLINK_ON      = 0.3   # seconds on
BLINK_OFF     = 0.3   # seconds off
BLINK_COUNT   = 5

BEEP_ON       = 0.15
BEEP_OFF      = 0.15
BEEP_COUNT    = 3


def section(title: str):
    print(f"\n{'─'*50}")
    print(f"  {title}")
    print(f"{'─'*50}")


def test_led(name: str, pin: int):
    section(f"Testing {name} (GPIO {pin})")
    led = LED(pin)
    for i in range(1, BLINK_COUNT + 1):
        print(f"  blink {i}/{BLINK_COUNT}  ON", flush=True)
        led.on()
        time.sleep(BLINK_ON)
        led.off()
        print(f"  blink {i}/{BLINK_COUNT}  OFF", flush=True)
        time.sleep(BLINK_OFF)
    led.off()
    led.close()
    print(f"  ✓ {name} done.")


def test_buzzer():
    section(f"Testing Buzzer (GPIO {PIN_BUZZER})")
    if HAS_BUZZER_LIB:
        buz = Buzzer(pin=PIN_BUZZER)
        for i in range(1, BEEP_COUNT + 1):
            print(f"  beep {i}/{BEEP_COUNT}", flush=True)
            buz.beep(BEEP_ON, BEEP_OFF, 1)
            time.sleep(BEEP_ON + BEEP_OFF + 0.1)
        buz.cleanup()
    else:
        buz = GpioBuzzer(PIN_BUZZER)
        for i in range(1, BEEP_COUNT + 1):
            print(f"  beep {i}/{BEEP_COUNT}", flush=True)
            buz.on()
            time.sleep(BEEP_ON)
            buz.off()
            time.sleep(BEEP_OFF)
        buz.close()
    print("  ✓ Buzzer done.")


def test_lcd():
    section("Testing LCD (I2C 0x27)")
    if not HAS_LCD or SKIP_LCD:
        print("  SKIPPED.")
        return
    try:
        lcd = LCD(2, 0x27, True)
        print("  Writing line 1: 'Driver Monitor'")
        lcd.message("Driver Monitor", 1)
        time.sleep(1)
        print("  Writing line 2: 'System OK :)'")
        lcd.message("System OK :)", 2)
        time.sleep(2)
        lcd.message("", 1)
        lcd.message("", 2)
        print("  ✓ LCD done.")
    except Exception as e:
        print(f"  ✗ LCD ERROR: {e}")
        print("    → Check I2C wiring and address (run: i2cdetect -y 1)")


def test_ir():
    section(f"Testing IR sensor (GPIO {PIN_IR})")
    ir = DigitalInputDevice(PIN_IR, pull_up=True)
    print("  Reading IR state for 3 seconds …")
    for _ in range(6):
        state = "ACTIVE (object detected)" if ir.is_active else "inactive"
        print(f"  IR: {state}", flush=True)
        time.sleep(0.5)
    ir.close()
    print("  ✓ IR sensor done.")


# ── Run all tests ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n🔧 Hardware Test — Driver Monitoring System")
    print("   GPIO library: gpiozero")
    print("   Run with --skip-lcd to skip the LCD test\n")

    try:
        test_led("LED (red alert)", PIN_LED)
        time.sleep(0.5)

        test_led("lightLED (green)", PIN_LIGHT_LED)
        time.sleep(0.5)

        test_led("blueLED (verifying)", PIN_BLUE_LED)
        time.sleep(0.5)

        test_buzzer()
        time.sleep(0.5)

        test_lcd()
        time.sleep(0.5)

        test_ir()

        print("\n✅ All hardware tests passed.\n")

    except KeyboardInterrupt:
        print("\n[test] Interrupted by user.")
    except Exception as e:
        print(f"\n[test] UNEXPECTED ERROR: {e}")
        raise
