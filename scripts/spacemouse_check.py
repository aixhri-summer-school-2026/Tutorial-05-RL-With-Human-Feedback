#!/usr/bin/env python3
"""SpaceMouse sanity check: print live deflection. Blocks until Ctrl-C.

The whole SpaceMouse phase is blocked until this prints nonzero values when the
puck is pushed. Run inside the container via `make spacemouse-check`.
"""
import time

import pyspacemouse


def main() -> None:
    if not pyspacemouse.open():
        raise SystemExit("pyspacemouse.open() failed — check /dev/hidraw* mapping and permissions")
    print("SpaceMouse open. Push the puck; Ctrl-C to stop.")
    try:
        while True:
            s = pyspacemouse.read()
            print(f"x={s.x:+.3f} y={s.y:+.3f} z={s.z:+.3f} "
                  f"roll={s.roll:+.3f} pitch={s.pitch:+.3f} yaw={s.yaw:+.3f} "
                  f"buttons={tuple(s.buttons)}")
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        pyspacemouse.close()


if __name__ == "__main__":
    main()
