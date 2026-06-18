# Live MuJoCo Cube Digital-Twin Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A read-only MuJoCo window that mirrors the real Franka workspace during a live run — cubes driven by AprilTag tf, arm driven by `/joint_states` — runnable in parallel with `make mile-real` / `make eval-real`.

**Architecture:** Pure, testable helpers live in a new `mile_franka/viz/mujoco_twin.py` (quaternion repack, joint-name→qpos map, scene injection). A thin entrypoint `scripts/view_cubes_mujoco.py` wires those helpers to an `rclpy` node (reusing `AprilTagPoseSource` + a `/joint_states` subscription) and a `mujoco.viewer` passive window. The loop only writes `qpos` and calls `mj_forward` (kinematics, no physics), so it never moves anything and never commands hardware.

**Tech Stack:** Python 3.10, `mujoco<3` (`mujoco==2.3.7`, includes `mujoco.viewer`), `rclpy`/`sensor_msgs`/`tf2_ros`, `scipy` (already used by the pose layer), `pytest`.

**Spec:** `docs/superpowers/specs/2026-06-18-mujoco-cube-twin-design.md`

---

## File Structure

- Create `mile_franka/viz/__init__.py` — package marker.
- Create `mile_franka/viz/mujoco_twin.py` — pure/isolated helpers: `pose_to_freejoint_qpos`, `joint_writes`, `_franka_description_franka_dir`, `resolve_scene_path`.
- Create `scripts/view_cubes_mujoco.py` — ROS + viewer entrypoint (the only ROS/viewer-coupled file).
- Create `tests/test_mujoco_twin.py` — unit tests for the helpers.
- Modify `requirements.txt` — add `mujoco==2.3.7`.
- Modify `Makefile` — add `view-twin` target + `.PHONY`.
- Modify `CLAUDE.md` — one command-list line.

Reused unchanged: `mile_franka/pose/apriltag.py` (`AprilTagPoseSource`), `mile_franka/pose/base.py` (`Pose`), `mile_franka/config.py` (`StackTaskConfig`), `mile_franka/envs/fake_backend.py` (`BOTTOM_CUBE`, `TOP_CUBE`), `mile_franka/assets/mujoco/stacking_scene.xml` + `stacking_objects.xml`.

---

## Task 1: Pure helper — `pose_to_freejoint_qpos`

**Files:**
- Create: `mile_franka/viz/__init__.py`
- Create: `mile_franka/viz/mujoco_twin.py`
- Test: `tests/test_mujoco_twin.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_mujoco_twin.py`:

```python
import numpy as np
import pytest

from mile_franka.pose.base import Pose
from mile_franka.viz.mujoco_twin import pose_to_freejoint_qpos


def test_pose_to_freejoint_qpos_reorders_quat_to_wxyz():
    # Pose.orientation is (qx, qy, qz, qw); MuJoCo free joints store (x,y,z, qw,qx,qy,qz).
    pose = Pose(position=[0.1, 0.2, 0.3], orientation=[0.0, 0.0, 0.7071, 0.7071])
    q = pose_to_freejoint_qpos(pose)
    assert q.shape == (7,)
    np.testing.assert_allclose(q[:3], [0.1, 0.2, 0.3], atol=1e-6)
    np.testing.assert_allclose(q[3:], [0.7071, 0.0, 0.0, 0.7071], atol=1e-6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `make shell` then inside, or directly:
`docker compose -f docker/docker-compose.yml exec sim bash -lc 'source scripts/in_container_env.sh && pytest tests/test_mujoco_twin.py::test_pose_to_freejoint_qpos_reorders_quat_to_wxyz -v'`
Expected: FAIL with `ModuleNotFoundError: No module named 'mile_franka.viz'`.

- [ ] **Step 3: Write minimal implementation**

Create `mile_franka/viz/__init__.py` (empty):

```python
```

Create `mile_franka/viz/mujoco_twin.py`:

```python
"""Pure/isolated helpers for the live MuJoCo digital twin.

Kept free of ROS and the viewer so the non-trivial logic (quaternion order,
joint-name->qpos mapping, scene injection) is unit-testable without hardware.
"""
from __future__ import annotations

import os
import shutil
from typing import List, Sequence, Tuple

import numpy as np

from mile_franka.pose.base import Pose


def pose_to_freejoint_qpos(pose: Pose) -> np.ndarray:
    """Pack a base-frame Pose into MuJoCo free-joint qpos order.

    Pose.orientation is (qx, qy, qz, qw); a MuJoCo free joint stores
    qpos = [x, y, z, qw, qx, qy, qz].
    """
    x, y, z = (float(v) for v in pose.position)
    qx, qy, qz, qw = (float(v) for v in pose.orientation)
    return np.array([x, y, z, qw, qx, qy, qz], dtype=np.float64)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `docker compose -f docker/docker-compose.yml exec sim bash -lc 'source scripts/in_container_env.sh && pytest tests/test_mujoco_twin.py -v'`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add mile_franka/viz/__init__.py mile_franka/viz/mujoco_twin.py tests/test_mujoco_twin.py
git commit -m "feat(viz): pose_to_freejoint_qpos helper for the mujoco twin"
```

---

## Task 2: Add the `mujoco` dependency

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Add the pin**

Edit `requirements.txt`, insert after the `numpy` line (line 2):

```
mujoco==2.3.7
```

(Respects the pinned-stack `mujoco<3` rule; 2.3.7 is the last 2.x and includes `mujoco.viewer`.)

- [ ] **Step 2: Install into the running container so later tasks can run without a full rebuild**

Run: `docker compose -f docker/docker-compose.yml exec sim bash -lc 'source scripts/in_container_env.sh && pip install mujoco==2.3.7'`
Expected: `Successfully installed mujoco-2.3.7`.

- [ ] **Step 3: Verify import + numpy ABI compatibility**

Run: `docker compose -f docker/docker-compose.yml exec sim bash -lc 'source scripts/in_container_env.sh && python3 -c "import numpy, mujoco, mujoco.viewer; print(\"numpy\", numpy.__version__, \"mujoco\", mujoco.__version__); mujoco.MjModel.from_xml_string(\"<mujoco/>\")"'`
Expected: prints versions and no traceback.
If it raises a numpy ABI error (`numpy.core.multiarray failed to import`): the container has numpy 2.x and the chosen mujoco wheel clashes — report back before proceeding (do **not** downgrade numpy; that breaks the gymnasium/sb3 stack). In practice mujoco's pybind bindings load fine under numpy 2.x.

- [ ] **Step 4: Commit**

```bash
git add requirements.txt
git commit -m "build: add mujoco==2.3.7 for the digital-twin viewer"
```

Note: `make build` bakes this into the image permanently; the in-container `pip install` above is only so the remaining tasks run now.

---

## Task 3: Helper — `joint_writes` (joint name → qpos address map)

**Files:**
- Modify: `mile_franka/viz/mujoco_twin.py`
- Test: `tests/test_mujoco_twin.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_mujoco_twin.py`:

```python
def test_joint_writes_maps_known_and_skips_unknown():
    mujoco = pytest.importorskip("mujoco")
    from mile_franka.viz.mujoco_twin import joint_writes

    xml = """
    <mujoco>
      <worldbody>
        <body name="b1">
          <joint name="j1" type="hinge" axis="0 0 1"/>
          <geom type="box" size="0.1 0.1 0.1"/>
          <body name="b2" pos="0 0 0.3">
            <joint name="j2" type="hinge" axis="0 1 0"/>
            <geom type="box" size="0.1 0.1 0.1"/>
          </body>
        </body>
      </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    writes = joint_writes(model, ["j1", "ghost", "j2"], [0.5, 9.9, -0.3])

    by_adr = {adr: val for adr, val in writes}
    a1 = int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "j1")])
    a2 = int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "j2")])
    assert by_adr[a1] == 0.5
    assert by_adr[a2] == -0.3
    assert len(writes) == 2  # unknown "ghost" skipped
```

- [ ] **Step 2: Run test to verify it fails**

Run: `docker compose -f docker/docker-compose.yml exec sim bash -lc 'source scripts/in_container_env.sh && pytest tests/test_mujoco_twin.py::test_joint_writes_maps_known_and_skips_unknown -v'`
Expected: FAIL with `ImportError: cannot import name 'joint_writes'`.

- [ ] **Step 3: Write minimal implementation**

Append to `mile_franka/viz/mujoco_twin.py`:

```python
def joint_writes(model, names: Sequence[str],
                 positions: Sequence[float]) -> List[Tuple[int, float]]:
    """Map (joint name, position) pairs to (qpos address, value).

    `model` is a mujoco.MjModel. Joint names absent from the model are skipped,
    so a mismatched arm_id or namespaced names degrade gracefully (arm stays at
    rest) instead of erroring. mujoco is imported lazily so this module loads
    without it for the pure tests.
    """
    import mujoco

    writes: List[Tuple[int, float]] = []
    for name, value in zip(names, positions):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            continue
        writes.append((int(model.jnt_qposadr[jid]), float(value)))
    return writes
```

- [ ] **Step 4: Run test to verify it passes**

Run: `docker compose -f docker/docker-compose.yml exec sim bash -lc 'source scripts/in_container_env.sh && pytest tests/test_mujoco_twin.py -v'`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add mile_franka/viz/mujoco_twin.py tests/test_mujoco_twin.py
git commit -m "feat(viz): joint_writes name->qpos mapper for the mujoco twin"
```

---

## Task 4: Helper — `resolve_scene_path` (inject scene beside panda.xml)

**Files:**
- Modify: `mile_franka/viz/mujoco_twin.py`
- Test: `tests/test_mujoco_twin.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_mujoco_twin.py`:

```python
def test_resolve_scene_path_copies_assets(tmp_path, monkeypatch):
    import mile_franka.viz.mujoco_twin as mt

    dest = tmp_path / "mujoco" / "franka"
    dest.mkdir(parents=True)
    # Stand in for the real franka_description location (avoids needing ament/ROS).
    monkeypatch.setattr(mt, "_franka_description_franka_dir", lambda: str(dest))

    scene = mt.resolve_scene_path()

    assert scene == str(dest / "stacking_scene.xml")
    assert (dest / "stacking_scene.xml").exists()
    assert (dest / "stacking_objects.xml").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `docker compose -f docker/docker-compose.yml exec sim bash -lc 'source scripts/in_container_env.sh && pytest tests/test_mujoco_twin.py::test_resolve_scene_path_copies_assets -v'`
Expected: FAIL with `AttributeError: ... has no attribute '_franka_description_franka_dir'`.

- [ ] **Step 3: Write minimal implementation**

Append to `mile_franka/viz/mujoco_twin.py`:

```python
def _franka_description_franka_dir() -> str:
    """Path to franka_description's mujoco/franka dir (where panda.xml + assets live).

    Lazy ament import so the module loads without ROS for the pure tests.
    """
    from ament_index_python.packages import get_package_share_directory
    return os.path.join(get_package_share_directory("franka_description"),
                        "mujoco", "franka")


def resolve_scene_path() -> str:
    """Copy the stacking scene + objects beside franka_description's panda.xml and
    return the scene path.

    The scene uses relative includes (panda.xml, meshdir="assets"), so it must sit
    in that directory at load time. scripts/sim_up.sh does this for the sim; the
    real-robot path never runs sim_up.sh, so the twin repeats the injection here.
    """
    dest = _franka_description_franka_dir()
    src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "..", "assets", "mujoco")
    for fn in ("stacking_scene.xml", "stacking_objects.xml"):
        shutil.copy(os.path.join(src_dir, fn), os.path.join(dest, fn))
    return os.path.join(dest, "stacking_scene.xml")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `docker compose -f docker/docker-compose.yml exec sim bash -lc 'source scripts/in_container_env.sh && pytest tests/test_mujoco_twin.py -v'`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add mile_franka/viz/mujoco_twin.py tests/test_mujoco_twin.py
git commit -m "feat(viz): resolve_scene_path injects stacking scene beside panda.xml"
```

---

## Task 5: The viewer entrypoint — `scripts/view_cubes_mujoco.py`

**Files:**
- Create: `scripts/view_cubes_mujoco.py`

- [ ] **Step 1: Write the script**

Create `scripts/view_cubes_mujoco.py`:

```python
#!/usr/bin/env python3
"""Live MuJoCo digital twin of the real Franka workspace.

Cubes are driven by AprilTag poses (tf2), the arm by /joint_states. The window is
READ-ONLY: it subscribes to /tf, /tf_static and /joint_states, publishes nothing to
control topics, and calls no state-mutating service -- so it is safe to run in
parallel with `make mile-real` / `make eval-real`. The loop only writes qpos and
calls mj_forward (kinematics, no physics), so nothing ever falls or is commanded.

Prereqs (real run): controller up (publishes /joint_states), `make apriltag-up`
(+ calibration for base->tag), host DISPLAY (`xhost +local:root`).

Usage (via `make view-twin`, or inside `make shell`):
    python3 scripts/view_cubes_mujoco.py
    python3 scripts/view_cubes_mujoco.py --joint-states-topic /panda/joint_states
    python3 scripts/view_cubes_mujoco.py --self-check    # no ROS/DISPLAY; scene+packing smoke
"""
import argparse
import time

import numpy as np

from mile_franka.config import StackTaskConfig
from mile_franka.envs.fake_backend import BOTTOM_CUBE, TOP_CUBE
from mile_franka.pose.base import Pose
from mile_franka.viz.mujoco_twin import (
    joint_writes, pose_to_freejoint_qpos, resolve_scene_path)

CUBES = (BOTTOM_CUBE, TOP_CUBE)


def _print_status(now, js_stamp, pose_src):
    parts = []
    parts.append("joints: " + ("n/a" if not js_stamp else f"{now - js_stamp:.1f}s"))
    for c in CUBES:
        seen = pose_src.last_seen(c)
        parts.append(f"{c}: " + ("never" if seen is None else f"{now - seen:.1f}s"))
    print("[twin] " + "  ".join(parts), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--joint-states-topic", default="/joint_states",
                    help="JointState topic for the arm/gripper (default: /joint_states)")
    ap.add_argument("--rate", type=float, default=30.0, help="update rate Hz (default: 30)")
    ap.add_argument("--self-check", action="store_true",
                    help="load scene, place a cube at a known pose, verify, exit "
                         "(no ROS, no DISPLAY)")
    args = ap.parse_args()

    import mujoco

    scene = resolve_scene_path()
    model = mujoco.MjModel.from_xml_path(scene)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    cfg = StackTaskConfig()
    half_edge = cfg.cube_size / 2.0

    # Free-joint qpos start address per cube.
    cube_qadr = {}
    for c in CUBES:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{c}_joint")
        if jid < 0:
            raise RuntimeError(f"cube joint {c}_joint not found in {scene}")
        cube_qadr[c] = int(model.jnt_qposadr[jid])

    # ---- self-check: deterministic, no ROS/DISPLAY ----
    if args.self_check:
        target = Pose(position=[0.5, 0.1, 0.3], orientation=[0.0, 0.0, 0.0, 1.0])
        a = cube_qadr[BOTTOM_CUBE]
        data.qpos[a:a + 7] = pose_to_freejoint_qpos(target)
        mujoco.mj_forward(model, data)
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, BOTTOM_CUBE)
        np.testing.assert_allclose(data.xpos[bid], [0.5, 0.1, 0.3], atol=1e-6)
        print("[twin] self-check OK (scene loads; base-frame pose == world pose)")
        return

    # ---- ROS ----
    import rclpy
    from sensor_msgs.msg import JointState
    from mile_franka.pose.apriltag import AprilTagPoseSource

    rclpy.init()
    node = rclpy.create_node("mile_franka_twin")

    js = {"names": [], "positions": [], "stamp": 0.0}

    def js_cb(msg):
        js["names"] = list(msg.name)
        js["positions"] = list(msg.position)
        js["stamp"] = time.time()

    node.create_subscription(JointState, args.joint_states_topic, js_cb, 10)
    pose_src = AprilTagPoseSource(half_edge=half_edge, node=node)

    import mujoco.viewer
    print(f"Scene        : {scene}")
    print(f"Joint topic  : {args.joint_states_topic}")
    print("Opening MuJoCo viewer (read-only twin). Close the window or Ctrl-C to stop.")
    viewer = mujoco.viewer.launch_passive(model, data)

    period = 1.0 / max(args.rate, 1.0)
    last_status = 0.0
    try:
        while rclpy.ok() and viewer.is_running():
            rclpy.spin_once(node, timeout_sec=period)

            for adr, val in joint_writes(model, js["names"], js["positions"]):
                data.qpos[adr] = val

            for c in CUBES:
                try:
                    pose = pose_src.get_pose(c)
                except RuntimeError:
                    continue  # never seen yet -> leave at rest
                a = cube_qadr[c]
                data.qpos[a:a + 7] = pose_to_freejoint_qpos(pose)

            mujoco.mj_forward(model, data)
            viewer.sync()

            now = time.time()
            if now - last_status >= 1.0:
                last_status = now
                _print_status(now, js["stamp"], pose_src)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            viewer.close()
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the `--self-check` smoke (no ROS, no DISPLAY)**

Run: `docker compose -f docker/docker-compose.yml exec sim bash -lc 'source scripts/in_container_env.sh && python3 scripts/view_cubes_mujoco.py --self-check'`
Expected: prints `[twin] self-check OK (scene loads; base-frame pose == world pose)` and exits 0. This confirms the scene injection, model load, free-joint packing, and that a base-frame pose lands at the same world coordinates (panda_link0 == world).

- [ ] **Step 3: Verify `--help` works without ROS/mujoco side effects**

Run: `docker compose -f docker/docker-compose.yml exec sim bash -lc 'source scripts/in_container_env.sh && python3 scripts/view_cubes_mujoco.py --help'`
Expected: argparse help text including `--self-check` and `--joint-states-topic`.

- [ ] **Step 4: Commit**

```bash
git add scripts/view_cubes_mujoco.py
git commit -m "feat: live MuJoCo cube digital-twin viewer (read-only, parallel-safe)"
```

---

## Task 6: `make view-twin` target

**Files:**
- Modify: `Makefile`

- [ ] **Step 1: Add the target**

In `Makefile`, add `view-twin` to the `.PHONY` line (append the word `view-twin`), and add this target after the `eval-real` target:

```makefile
view-twin:                   ## live MuJoCo digital twin of the real workspace on the host display (read-only; needs controller + apriltag-up; safe in parallel with mile-real/eval-real)
	@echo "Host prereq (once per login): xhost +local:root"
	$(DC) exec -e DISPLAY=$$DISPLAY sim bash -lc '$(ENVSH) && python3 scripts/view_cubes_mujoco.py'
```

- [ ] **Step 2: Verify the recipe resolves**

Run: `make -n view-twin`
Expected: prints the `docker compose ... exec -e DISPLAY=... sim bash -lc '... python3 scripts/view_cubes_mujoco.py'` command with no make errors.

- [ ] **Step 3: Commit**

```bash
git add Makefile
git commit -m "feat: add make view-twin target for the digital-twin viewer"
```

---

## Task 7: Document the command

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Add a command-list line**

In `CLAUDE.md`, in the `## Commands` block under the Franka section, add after the `make eval-real` line:

```
make view-twin                # (in-container, host display) read-only MuJoCo twin: cubes from AprilTag, arm from /joint_states; safe to run alongside mile-real/eval-real
```

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: note make view-twin in CLAUDE.md commands"
```

---

## Self-Review

**Spec coverage:**
- Read-only / parallel-safe (unique node `mile_franka_twin`, only subscriptions) → Task 5 script + Task 6/7 docs. ✓
- Cubes from AprilTag via existing pose layer → Task 5 reuses `AprilTagPoseSource`. ✓
- Arm from `/joint_states`, names match model, skip-unknown → Task 3 `joint_writes` + Task 5 loop. ✓
- Scene injection beside panda.xml (real path lacks sim_up.sh) → Task 4 `resolve_scene_path`. ✓
- World == base frame mapping → verified in Task 5 `--self-check`. ✓
- `mj_forward` only, no physics → Task 5 loop. ✓
- Quat xyzw→wxyz → Task 1 `pose_to_freejoint_qpos`. ✓
- mujoco<3 dependency → Task 2. ✓
- Interactive host-display window → Task 5 `launch_passive` + Task 6 `-e DISPLAY` exec. ✓
- Status line / stale handling → Task 5 `_print_status` + cube-rest-on-RuntimeError. ✓
- `--joint-states-topic` override for namespaced graphs → Task 5 arg. ✓
- Unit tests for pure bits → Tasks 1, 3, 4. ✓
- Error handling (no joint_states → rest; no apriltag → rest+warn; no mujoco → Task 2 verify) ✓

**Placeholder scan:** none — every step has concrete code/commands.

**Type consistency:** `Pose.position`/`.orientation` (xyzw) used consistently; `pose_to_freejoint_qpos` returns `(7,)` np array; `joint_writes` returns `List[Tuple[int, float]]`; cube joints `f"{c}_joint"` match `stacking_objects.xml` (`bottom_cube_joint`/`top_cube_joint`); body names `BOTTOM_CUBE`/`TOP_CUBE` match `stacking_objects.xml`. ✓

**Manual verification (not automatable here — needs hardware + DISPLAY):** the full live window against a running controller + `apriltag-up` in parallel with `mile-real`. The `--self-check` smoke covers everything that doesn't need ROS/DISPLAY.
