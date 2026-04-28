#!/usr/bin/env python3

import time

# Import your LCD class
from LCD import LCD

def main():
    try:
        # Initialize LCD (same config as your project)
        lcd = LCD(2, 0x27, True)

        # Write messages
        lcd.message("hello", 1)
        lcd.message("azizosss", 2)

        # Keep it displayed forever
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        # Optional: clear LCD when stopping
        lcd.message("", 1)
        lcd.message("", 2)

if __name__ == "__main__":
    main()
