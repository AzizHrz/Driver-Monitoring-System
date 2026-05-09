# bridge.py
# ─────────────────────────────────────────────────────────────────────────────
# Runs on the HOST (Python 3.13) — direct hardware access.
#
# Responsibilities (one each):
#   • Stream raw MJPEG frames from picamera2
#     → GET  http://0.0.0.0:5000/stream
#   • Expose IR sensor state
#     → GET  http://0.0.0.0:5000/ir_status
#   • Receive AI result commands and drive GPIO
#     → POST http://0.0.0.0:5000/command
#
# FIXES vs previous version:
#   1. All GPIO actions are NON-BLOCKING — blink/beep use background=True or
#      the gpiozero Buzzer.beep() background flag; no sleep() on any hot path.
#   2. LED state is managed via a simple _apply_gpio() helper that avoids
#      redundant blink() calls (calling blink() in a loop restarts the
#      animation every frame — now it only changes when state transitions).
#   3. Flask runs with threaded=True so the stream endpoint and the command
#      endpoint never block each other.
# ─────────────────────────────────────────────────────────────────────────────

import io
import threading
from flask import Flask, Response, request, jsonify
from picamera2 import Picamera2
from picamera2.encoders import JpegEncoder
from picamera2.outputs import FileOutput
from gpiozero import LED, Buzzer, DigitalInputDevice

import config
from LCD import LCD

# ── GPIO ──────────────────────────────────────────────────────────────────────
led       = LED(config.PIN_LED)
buz       = Buzzer(config.PIN_BUZZER)
lightLED  = LED(config.PIN_LIGHT_LED)
blueLED   = LED(config.PIN_BLUE_LED)
ir        = DigitalInputDevice(config.PIN_IR, pull_up=True)
lcd       = LCD(config.LCD_BUS, config.LCD_ADDR, config.LCD_BACKLIT)

# Track last GPIO state to avoid restarting blink() every command frame
_last_hw_state: str = ""


def _apply_gpio(state: str, drowsy: bool, eyes_closed: bool) -> None:
    """Drive LEDs + buzzer based on AI state. Called at most once per command."""
    global _last_hw_state
    key = f"{state}-{drowsy}-{eyes_closed}"
    if key == _last_hw_state:
        return                          # nothing changed — skip
    _last_hw_state = key

    # Stop any running blink/beep first
    led.off()
    blueLED.off()
    lightLED.off()
    buz.off()

    if state == "VERIFIED":
        lightLED.on()
        if drowsy:
            led.blink(on_time=0.3, off_time=0.3, background=True)
            buz.beep(on_time=0.1, off_time=0.1, n=3, background=True)
        elif eyes_closed:
            led.blink(on_time=0.5, off_time=0.5, background=True)

    elif state == "UNKNOWN_PERSON":
        led.blink(on_time=0.3, off_time=0.3, background=True)
        buz.beep(on_time=0.1, off_time=0.1, n=3, background=True)

    elif state == "VERIFYING":
        blueLED.blink(on_time=0.5, off_time=0.5, background=True)

    # NO_FACE → all off (already done above)


# ── Camera / MJPEG buffer ─────────────────────────────────────────────────────
class _StreamBuffer(io.BufferedIOBase):
    def __init__(self):
        self.frame     = None
        self.condition = threading.Condition()

    def write(self, buf):
        with self.condition:
            self.frame = buf
            self.condition.notify_all()


def _mjpeg_generator(buf: _StreamBuffer):
    while True:
        with buf.condition:
            buf.condition.wait()
            frame = buf.frame
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame +
            b"\r\n"
        )


# ── Flask app ─────────────────────────────────────────────────────────────────
app        = Flask(__name__)
_stream_buf = _StreamBuffer()


@app.route("/stream")
def stream():
    """Raw MJPEG — consumed by the container."""
    return Response(
        _mjpeg_generator(_stream_buf),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/ir_status")
def ir_status():
    """IR proximity sensor state — polled by the container."""
    return jsonify({"is_active": ir.is_active})


@app.route("/command", methods=["POST"])
def command():
    """
    Receive AI result from container. Expected JSON:
    {
        "state":       "VERIFIED" | "UNKNOWN_PERSON" | "VERIFYING" | "NO_FACE",
        "drowsy":      true | false,
        "eyes_closed": true | false,
        "lcd_line1":   "...",
        "lcd_line2":   "..."
    }
    """
    data        = request.get_json(force=True)
    state       = data.get("state",       "NO_FACE")
    drowsy      = bool(data.get("drowsy",      False))
    eyes_closed = bool(data.get("eyes_closed", False))
    lcd_line1   = data.get("lcd_line1", "")
    lcd_line2   = data.get("lcd_line2", "")

    # LCD (blocking ~ms — acceptable; run in thread if jitter matters)
    try:
        lcd.message(lcd_line1, 1)
        lcd.message(lcd_line2, 2)
    except Exception as e:
        print(f"[bridge] LCD error: {e}")

    _apply_gpio(state, drowsy, eyes_closed)
    return jsonify({"status": "ok"})


@app.route("/health")
def health():
    return jsonify({"status": "bridge running"})


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    picam2 = Picamera2()
    cfg    = picam2.create_video_configuration(
        main={"size": config.CAM_RESOLUTION},
        controls={"FrameRate": config.CAM_FRAMERATE},
    )
    picam2.configure(cfg)
    picam2.start_recording(JpegEncoder(), FileOutput(_stream_buf))
    print(f"[bridge] Streaming  → http://0.0.0.0:{config.BRIDGE_PORT}/stream")
    print(f"[bridge] Commands   → POST http://0.0.0.0:{config.BRIDGE_PORT}/command")

    try:
        app.run(
            host=config.BRIDGE_HOST,
            port=config.BRIDGE_PORT,
            threaded=True,
            use_reloader=False,
        )
    finally:
        picam2.stop_recording()
        led.off(); blueLED.off(); lightLED.off(); buz.off()
        print("[bridge] Stopped.")
