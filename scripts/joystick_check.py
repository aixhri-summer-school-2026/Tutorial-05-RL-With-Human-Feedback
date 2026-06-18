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

    # Xbox 360: left-stick-forward=+x, left-stick-left=+y, right-stick-up=+z (axis4, -=up)
    # RB=segment toggle, A=gripper, Start=done, Back=discard
    dev = JoystickDevice(ax_x=1, ax_x_sign=-1.0, ax_y=0, ax_y_sign=-1.0,
                         ax_z=4, ax_z_sign=-1.0, clutch_button=5, gripper_button=0,
                         done_button=7, discard_button=6, gripper_toggle=True)
    print("Xbox360: RB=segment  A=gripper  Start=done  Back=discard  Ctrl-C to stop")
    prev_intervene = False
    try:
        while True:
            r = dev.read()
            if r.intervene != prev_intervene:
                print(f"  SEGMENT {'ON ' if r.intervene else 'OFF'}", flush=True)
                prev_intervene = r.intervene
            if r.intervene:
                dx, dy, dz = r.action[0], r.action[1], r.action[2]
                gripper = "close" if r.action[3] > 0 else "open"
                print(f"  dx={dx:+.3f} dy={dy:+.3f} dz={dz:+.3f} gripper={gripper}")
            if r.done:
                print("  DONE")
            if r.discard:
                print("  DISCARD")
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        dev.close()


if __name__ == "__main__":
    main()
