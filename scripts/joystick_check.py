#!/usr/bin/env python3
"""Gamepad sanity check: print live axes/buttons. Blocks until Ctrl-C.

The whole gamepad path is blocked until this prints nonzero values when the sticks are
pushed. Run inside the container via `make joystick-check`. Use `--fake` to exercise the
read path with no controller attached.
"""
import argparse
import time

from mile_franka.teleop.joystick import JoystickDevice


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fake", action="store_true",
                        help="use a synthetic reader (no hardware) to smoke-test the read path")
    args = parser.parse_args()

    if args.fake:
        class Snap:
            axes = [0.5, -0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            buttons = [1, 0, 0, 0, 0, 1, 0, 0]
        dev = JoystickDevice(reader=lambda: Snap())
        r = dev.read()
        print(f"[fake] action={r.action} intervene={r.intervene} done={r.done}")
        return

    dev = JoystickDevice()
    print("Joystick open. Push the sticks / press buttons; Ctrl-C to stop.")
    try:
        while True:
            r = dev.read()
            print(f"action={r.action} intervene={r.intervene} done={r.done}")
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        dev.close()


if __name__ == "__main__":
    main()
