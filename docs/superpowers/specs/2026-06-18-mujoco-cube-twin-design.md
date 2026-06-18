# Live MuJoCo digital twin for the real Franka run

**Date:** 2026-06-18
**Status:** Design approved, pre-implementation
**Related:** `docs/superpowers/specs/2026-06-15-mile-franka-stacking-design.md` (Franka adaptation),
`mile_franka/pose/apriltag.py` (tag→cube conversion), `scripts/sim_up.sh` (scene injection),
`scripts/view_camera_tags.py` (the tf-lookup + viewer template).

## Goal

Mirror the real workspace inside the familiar MuJoCo stacking scene during a **real-robot**
run: the two cubes are driven live by AprilTag poses, and the robot arm mirrors the real FR3
via its joint states. The window is **read-only** — it never commands hardware. It doubles as
a visual check of the camera calibration and the `cube_center_pose()` `half_edge` offset sign
(you can *see* whether the cube lands at the true center).

### Why this is needed

On the real path (`MultipandaRosBackend` with `sim=False`) there is **no MuJoCo process** — the
multipanda controller drives hardware directly, and the mujoco_ros services (`get_body_state`,
`set_body_state`, pause) exist only in the sim graph. So nothing visualizes the workspace today.
This twin adds a passive MuJoCo window that renders what the robot believes.

## Hard requirement: runs in parallel with `make mile-real` / `make eval-real`

The twin must run at the same time as a live MILE/eval run without perturbing it. This is
satisfied by construction:

- **Separate process, same container/graph.** `make view-twin` is its own `docker exec` into the
  persistent `sim` container, joining the same DDS graph as the `mile-real`/`eval-real` exec.
  Multiple execs into one container are independent.
- **Unique node name** `mile_franka_twin` (distinct from the backend's `mile_franka_backend`).
- **Strictly read-only.** Subscribes only to `/tf`, `/tf_static`, and `/joint_states`. Publishes
  nothing to control/equilibrium topics; calls no state-mutating service. It physically cannot
  affect the policy, data collection, or hardware.
- **Same data the policy sees.** The twin's `AprilTagPoseSource` does its own tf lookups (a second
  tf listener on the graph is fine — tf is many-listener by design) and yields the *same*
  base→tag poses the collector's pose source uses.
- **Independent display.** The host GLFW window (own `-e DISPLAY` exec) does not touch the run's
  headless video recording.

## Verified codebase facts (2026-06-18)

- **World frame == `panda_link0`.** In `panda.xml` the base body is `pos="0 0 0" quat="0 0 0 1"`,
  so AprilTag base-frame poses map directly to MuJoCo world coordinates (no extra transform).
- **Cube bodies/joints:** `bottom_cube`/`top_cube`, each with a free joint
  (`bottom_cube_joint`/`top_cube_joint`). Body names match `BOTTOM_CUBE`/`TOP_CUBE` pose keys.
- **Arm joints:** `panda_joint1..7`, fingers `panda_finger_joint1`/`panda_finger_joint2`.
- **Scene loading:** `stacking_scene.xml` uses relative includes (`panda.xml`, `meshdir="assets"`)
  and must sit beside `panda.xml` in `franka_description/mujoco/franka/` at load time. `sim_up.sh`
  copies it there at sim launch; the real path never runs `sim_up.sh`, so the twin must do the
  same idempotent copy before loading.
- **MuJoCo python is not installed** in the image (the sim uses the C++ mujoco_ros plugin). The
  twin needs `mujoco<3` (which includes `mujoco.viewer`).

## Architecture

Single standalone script `scripts/view_cubes_mujoco.py`, reusing existing layers.

### Components & data flow

1. **Scene setup.** Resolve `franka_description`'s `mujoco/franka` dir (via
   `ament_index_python`), copy `stacking_scene.xml` + `stacking_objects.xml` there (idempotent;
   same approach as `sim_up.sh`), then `model = mujoco.MjModel.from_xml_path(scene)`,
   `data = mujoco.MjData(model)`.
2. **ROS node** `mile_franka_twin` (`rclpy`):
   - `AprilTagPoseSource(half_edge=config.cube_size/2)` built on this node → cube poses
     (`Pose`, base frame) for `bottom_cube`/`top_cube`. Reuses the existing pose layer; no new
     pose code.
   - Subscription to `/joint_states` (`sensor_msgs/JointState`); cache latest position keyed by
     joint name. *(CONFIRM@bringup: the real controller's joint-states topic and joint names;
     default `/joint_states` with `panda_joint1..7` + `panda_finger_joint1/2`.)*
3. **Update loop (~30 Hz):**
   - `rclpy.spin_once(node, timeout_sec=...)` to process tf + joint_states callbacks.
   - **Arm qpos:** for each cached joint name that exists in the model, write
     `data.qpos[qpos_adr]` (via `mj_name2id(JOINT)` → `jnt_qposadr`). Missing joints stay at
     their model default (rest).
   - **Cube qpos:** for each cube, `get_pose(name)` → free-joint 7-vector
     `[x, y, z, qw, qx, qy, qz]` (AprilTag quat is xyzw; MuJoCo free joints are wxyz). A cube
     never seen stays at its scene default pose, with a warning.
   - `mujoco.mj_forward(model, data)` — **kinematics only, no `mj_step`** (so gravity/contacts
     never move anything; pure visualization).
   - `viewer.sync()`.
4. **Display.** `mujoco.viewer.launch_passive(model, data)` → GLFW window on the host display,
   launched like `make sim-gui` (`docker exec -e DISPLAY=$DISPLAY`, host `xhost +local:root`).
5. **Status.** Once per second, print a status line: per-cube last-seen age (from
   `AprilTagPoseSource.last_seen`) and joint-states age. Cubes not currently visible keep their
   last pose (the pose source already caches) and are flagged stale in the status line.

### Pure, unit-testable helpers (extracted as module-level functions)

- `apriltag_to_mujoco_freejoint(pose) -> np.ndarray(7)`: pack a `Pose` (xyz + xyzw quat) into the
  MuJoCo free-joint qpos order `[x, y, z, qw, qx, qy, qz]`.
- `joint_states_to_qpos_writes(model, names, positions) -> list[(qpos_adr, value)]`: map a
  `JointState` name/position pair list to qpos addresses for joints present in the model
  (silently skip unknown joints). *(This one needs a model; test guarded with
  `importorskip("mujoco")` and the scene available.)*

Keeping these pure isolates the only non-trivial logic (quat order, name→address mapping) from
ROS and the viewer so it can be tested without hardware.

## New artifacts

- `scripts/view_cubes_mujoco.py` — the viewer.
- `make view-twin` target — host-DISPLAY `docker exec` like `sim-gui`; documents prereqs
  (controller running + `make apriltag-up`).
- `requirements.txt` — add `mujoco==2.3.7` (respects the pinned-stack `mujoco<3` rule). Requires
  `make build`. *(Rebuild chosen over runtime `pip install` for a reproducible tutorial image.)*
- `tests/test_mujoco_twin.py` — unit tests for `apriltag_to_mujoco_freejoint` (no deps) and, when
  mujoco + scene are available, `joint_states_to_qpos_writes`.

## Error handling

- **mujoco import fails** → clear message: rebuild the image (`make build`) after the
  `requirements.txt` bump.
- **No DISPLAY / GLFW init fails** → message pointing to host `xhost +local:root` and the
  `-e DISPLAY` exec (same failure mode as `sim-gui`).
- **No AprilTag / no calibration** → base→tag lookups fail; cubes render at their rest pose with a
  warning, and the arm still mirrors `/joint_states`.
- **No `/joint_states`** → arm/fingers stay at rest (warn, do not error) so the twin is still
  useful for cube-only checks.

## Out of scope (YAGNI)

Physics stepping, recording/video, camera-follow, GUI controls, dual-arm, depth. The twin only
mirrors pose state into a passive window.

## Open confirmations for bring-up (France lab)

- `CONFIRM@bringup`: real controller's joint-states topic name and joint naming (default
  `/joint_states`, `panda_joint1..7`, `panda_finger_joint1/2`).
- `CONFIRM@bringup`: the `cube_center_pose()` `half_edge` sign — this twin is the intended visual
  test for it.
