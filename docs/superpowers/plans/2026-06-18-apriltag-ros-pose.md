# AprilTag-over-ROS Object Pose Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Provide real cube poses in the robot base frame from AprilTags, sourced from ROS nodes (`realsense2_camera` + `apriltag_ros`), behind the existing `ObjectPoseSource` abstraction.

**Architecture:** `realsense2_camera` and `apriltag_ros` (ROS 2 Humble) publish the RGB stream and per-tag tf transforms; a static transform from `panda_link0` to the camera frame (emitted from a calibration YAML) lets tf2 chain `base → camera → tag`. `AprilTagPoseSource` does a tf2 lookup per cube tag and applies a half-edge offset to report the *cube center* (not the tag face). A generic `RosPoseStampedSource` is built alongside as the seam for hucebot's FoundationPose node later (it publishes the same `PoseStamped`). All ROS access is injectable so the pose/offset/calibration logic is unit-tested with no ROS, no camera, and no hardware.

**Tech Stack:** Python 3.10, ROS 2 Humble (`rclpy`, `tf2_ros`, `geometry_msgs`), `realsense2_camera`, `apriltag_ros`, `scipy` (rotation math), `PyYAML`, `pytest`.

**Scope note:** This plan covers ONLY the pose-sensing subsystem (spec §5.1–5.2). Separate follow-on plans cover: the eye-to-hand calibration *capture* script (`calibrate_camera.py`), the `Franka-Stack-Real-v0` env builder, live controller bring-up, and MILE-on-hardware. This plan supersedes Workstream 1 of `2026-06-18-phase-b-real-fr3-apriltag.md` and adopts the ROS-node decision (spec §8.1). Parent spec: `docs/superpowers/specs/2026-06-18-phase-b-real-fr3-apriltag-design.md`.

---

## File Structure

- `mile_franka/pose/calibration.py` (create) — `CameraCalibration` dataclass + YAML load + static-transform-args emission. One responsibility: the camera↔base extrinsics file format and its conversions.
- `mile_franka/pose/apriltag.py` (create) — `AprilTagPoseSource` (tf2-backed) + cube-tag constants + the pure cube-center offset helper. One responsibility: turn a tag transform into a cube-center `Pose`.
- `mile_franka/pose/ros_posestamped.py` (create) — `RosPoseStampedSource` generic seam (for FoundationPose later). One responsibility: latest `PoseStamped` on a topic → base-frame `Pose`.
- `config/camera_calib.example.yaml` (create) — committed template; the real `config/camera_calib.yaml` is gitignored.
- `config/apriltag.yaml` (create) — `apriltag_ros` node params (family, tag size, per-id frame names).
- `launch/apriltag_realsense.launch.py` (create) — brings up `realsense2_camera`, `apriltag_ros`, and the calibration static transform.
- `docker/Dockerfile` (modify) — add the two apt ROS packages.
- `Makefile` (modify) — add `make apriltag-up` (launch) and `make pose-test` (unit tests) verbs.
- `tests/test_calibration.py`, `tests/test_apriltag_pose.py`, `tests/test_ros_posestamped.py` (create) — unit tests for the pure logic with injected ROS.

---

### Task 1: Camera calibration file format

**Files:**
- Create: `mile_franka/pose/calibration.py`
- Test: `tests/test_calibration.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_calibration.py
import numpy as np
import pytest

from mile_franka.pose.calibration import CameraCalibration, load_camera_calibration


def _write_yaml(tmp_path, body):
    p = tmp_path / "calib.yaml"
    p.write_text(body)
    return str(p)


def test_load_camera_calibration_reads_intrinsics_and_extrinsics(tmp_path):
    path = _write_yaml(tmp_path, """
intrinsics: {fx: 615.0, fy: 616.0, cx: 320.0, cy: 240.0}
camera_to_base:
  - [1.0, 0.0, 0.0, 0.2]
  - [0.0, 1.0, 0.0, -0.1]
  - [0.0, 0.0, 1.0, 0.5]
  - [0.0, 0.0, 0.0, 1.0]
""")
    calib = load_camera_calibration(path)
    assert calib.camera_params == (615.0, 616.0, 320.0, 240.0)
    assert calib.camera_to_base.shape == (4, 4)
    np.testing.assert_allclose(calib.camera_to_base[:3, 3], [0.2, -0.1, 0.5])


def test_load_rejects_non_4x4_extrinsics(tmp_path):
    path = _write_yaml(tmp_path, """
intrinsics: {fx: 1.0, fy: 1.0, cx: 0.0, cy: 0.0}
camera_to_base: [[1.0, 0.0], [0.0, 1.0]]
""")
    with pytest.raises(ValueError):
        load_camera_calibration(path)


def test_static_transform_args_emits_xyz_quat_and_frames(tmp_path):
    path = _write_yaml(tmp_path, """
intrinsics: {fx: 1.0, fy: 1.0, cx: 0.0, cy: 0.0}
camera_to_base:
  - [1.0, 0.0, 0.0, 0.2]
  - [0.0, 1.0, 0.0, -0.1]
  - [0.0, 0.0, 1.0, 0.5]
  - [0.0, 0.0, 0.0, 1.0]
""")
    calib = load_camera_calibration(path)
    args = calib.static_transform_args("panda_link0", "camera_color_optical_frame")
    # identity rotation -> quat (0,0,0,1); translation passes through
    assert args[:3] == ["0.2", "-0.1", "0.5"]
    np.testing.assert_allclose([float(a) for a in args[3:7]], [0.0, 0.0, 0.0, 1.0], atol=1e-6)
    assert args[7:] == ["panda_link0", "camera_color_optical_frame"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_calibration.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mile_franka.pose.calibration'`

- [ ] **Step 3: Write minimal implementation**

```python
# mile_franka/pose/calibration.py
"""Camera↔base calibration file format for the eye-to-hand RealSense.

`camera_to_base` is a 4x4 homogeneous transform with p_base = T @ p_cam, i.e. the camera
frame expressed in the robot base frame. That is exactly the parent→child transform a
static_transform_publisher needs with parent=panda_link0, child=camera optical frame, so
tf2 can chain base → camera → tag. Real values are machine-specific (regenerate with the
calibration capture script after any camera move); only the template is committed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import yaml


@dataclass
class CameraCalibration:
    fx: float
    fy: float
    cx: float
    cy: float
    camera_to_base: np.ndarray  # (4, 4)

    @property
    def camera_params(self) -> Tuple[float, float, float, float]:
        return (self.fx, self.fy, self.cx, self.cy)

    def static_transform_args(self, parent_frame: str, child_frame: str) -> List[str]:
        """Args for ros2 static_transform_publisher (x y z qx qy qz qw parent child)."""
        from scipy.spatial.transform import Rotation

        t = self.camera_to_base[:3, 3]
        quat = Rotation.from_matrix(self.camera_to_base[:3, :3]).as_quat()  # xyzw
        vals = [f"{v:g}" for v in (*t, *quat)]
        return [*vals, parent_frame, child_frame]


def load_camera_calibration(path: str) -> CameraCalibration:
    with open(path) as f:
        data = yaml.safe_load(f)
    intr = data["intrinsics"]
    mat = np.asarray(data["camera_to_base"], dtype=np.float64)
    if mat.shape != (4, 4):
        raise ValueError(f"camera_to_base must be 4x4, got {mat.shape}")
    return CameraCalibration(
        fx=float(intr["fx"]), fy=float(intr["fy"]),
        cx=float(intr["cx"]), cy=float(intr["cy"]),
        camera_to_base=mat,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_calibration.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add mile_franka/pose/calibration.py tests/test_calibration.py
git commit -m "pose: camera<->base calibration file format + static-transform emission"
```

---

### Task 2: Cube-center offset helper (pure geometry)

**Files:**
- Create: `mile_franka/pose/apriltag.py`
- Test: `tests/test_apriltag_pose.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_apriltag_pose.py
import numpy as np
from scipy.spatial.transform import Rotation

from mile_franka.pose.apriltag import cube_center_pose


def test_offset_along_z_with_identity_rotation():
    # tag at (0,0,0.5), facing with +z = world +z; cube center is half-edge further along +z
    pose = cube_center_pose(translation=[0.0, 0.0, 0.5],
                            quat_xyzw=[0.0, 0.0, 0.0, 1.0],
                            half_edge=0.025)
    np.testing.assert_allclose(pose.position, [0.0, 0.0, 0.525], atol=1e-6)
    np.testing.assert_allclose(pose.orientation, [0.0, 0.0, 0.0, 1.0], atol=1e-6)


def test_offset_follows_orientation():
    # rotate the tag 180deg about x -> its +z now points along world -z, so the half-edge
    # offset must subtract from z.
    q = Rotation.from_euler("x", 180, degrees=True).as_quat()
    pose = cube_center_pose(translation=[0.0, 0.0, 0.5], quat_xyzw=q, half_edge=0.025)
    np.testing.assert_allclose(pose.position, [0.0, 0.0, 0.475], atol=1e-6)


def test_zero_offset_is_passthrough():
    pose = cube_center_pose(translation=[0.1, 0.2, 0.3],
                            quat_xyzw=[0.0, 0.0, 0.0, 1.0], half_edge=0.0)
    np.testing.assert_allclose(pose.position, [0.1, 0.2, 0.3], atol=1e-6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_apriltag_pose.py -v`
Expected: FAIL with `ImportError: cannot import name 'cube_center_pose'`

- [ ] **Step 3: Write minimal implementation**

```python
# mile_franka/pose/apriltag.py
"""AprilTag object poses from apriltag_ros via tf2.

apriltag_ros publishes a tf frame per detected tag; with a static transform from the robot
base to the camera optical frame (from the calibration YAML), tf2 yields base→tag directly.
The tag sits on a cube FACE, so the cube center is half the cube edge further along the tag's
outward normal (its +z axis). cube_center_pose() is pure so the geometry is unit-tested with
no ROS. CONFIRM@bringup: the offset sign (whether the apriltag +z points into or out of the
cube on the real mount) — flip half_edge sign here if the reported center lands in front of
the face instead of at the cube center.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from mile_franka.pose.base import Pose


def cube_center_pose(translation: Sequence[float], quat_xyzw: Sequence[float],
                     half_edge: float) -> Pose:
    """Cube-center Pose from a base-frame tag transform + half the cube edge.

    The offset is applied along the tag's +z axis expressed in the base frame.
    """
    t = np.asarray(translation, dtype=np.float64).reshape(3)
    R = Rotation.from_quat(np.asarray(quat_xyzw, dtype=np.float64)).as_matrix()
    center = t + half_edge * R[:, 2]
    return Pose(tuple(float(v) for v in center),
                tuple(float(v) for v in quat_xyzw))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_apriltag_pose.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add mile_franka/pose/apriltag.py tests/test_apriltag_pose.py
git commit -m "pose: pure cube-center offset helper (tag face -> cube center)"
```

---

### Task 3: `AprilTagPoseSource` with injectable tf lookup

**Files:**
- Modify: `mile_franka/pose/apriltag.py`
- Test: `tests/test_apriltag_pose.py`

- [ ] **Step 1: Write the failing test (append to the existing test file)**

```python
# append to tests/test_apriltag_pose.py
import pytest

from mile_franka.pose.apriltag import AprilTagPoseSource, CUBE_TAG_FRAMES
from mile_franka.envs.fake_backend import BOTTOM_CUBE, TOP_CUBE


class FakeTf:
    """Stand-in for a tf2 lookup: name(frame)->(translation, quat) or raise if unknown."""
    def __init__(self, table):
        self.table = table
        self.calls = []

    def __call__(self, base_frame, tag_frame):
        self.calls.append((base_frame, tag_frame))
        if tag_frame not in self.table:
            raise LookupError(tag_frame)
        return self.table[tag_frame]


def test_get_pose_returns_cube_center_for_known_tag():
    tf = FakeTf({CUBE_TAG_FRAMES[BOTTOM_CUBE]: ([0.0, 0.0, 0.5], [0.0, 0.0, 0.0, 1.0])})
    src = AprilTagPoseSource(half_edge=0.025, base_frame="panda_link0", tf_lookup=tf)
    pose = src.get_pose(BOTTOM_CUBE)
    np.testing.assert_allclose(pose.position, [0.0, 0.0, 0.525], atol=1e-6)
    assert tf.calls == [("panda_link0", CUBE_TAG_FRAMES[BOTTOM_CUBE])]


def test_get_pose_returns_last_known_when_tag_missing_this_frame():
    table = {CUBE_TAG_FRAMES[TOP_CUBE]: ([0.0, 0.0, 0.5], [0.0, 0.0, 0.0, 1.0])}
    tf = FakeTf(table)
    src = AprilTagPoseSource(half_edge=0.025, base_frame="panda_link0", tf_lookup=tf)
    first = src.get_pose(TOP_CUBE)
    del table[CUBE_TAG_FRAMES[TOP_CUBE]]  # tag now not visible
    second = src.get_pose(TOP_CUBE)       # must fall back to the cached pose
    np.testing.assert_allclose(first.position, second.position, atol=1e-9)


def test_get_pose_raises_if_tag_never_seen():
    tf = FakeTf({})
    src = AprilTagPoseSource(half_edge=0.025, base_frame="panda_link0", tf_lookup=tf)
    with pytest.raises(RuntimeError):
        src.get_pose(BOTTOM_CUBE)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_apriltag_pose.py -v`
Expected: FAIL with `ImportError: cannot import name 'AprilTagPoseSource'`

- [ ] **Step 3: Write minimal implementation (append to `mile_franka/pose/apriltag.py`)**

```python
# append to mile_franka/pose/apriltag.py
import time
from typing import Callable, Dict, Optional, Tuple

from mile_franka.envs.fake_backend import BOTTOM_CUBE, TOP_CUBE
from mile_franka.pose.base import ObjectPoseSource

# tag36h11 ids from scripts/generate_cube_tags.py: 0 = bottom, 1 = top. apriltag_ros names
# each tag's tf frame "<family>:<id>" (configurable in config/apriltag.yaml).
TAG_FAMILY = "tag36h11"
TAG_SIZE_M = 0.036  # printed tag size reported by generate_cube_tags.py
CUBE_TAG_IDS = {BOTTOM_CUBE: 0, TOP_CUBE: 1}
CUBE_TAG_FRAMES = {name: f"{TAG_FAMILY}:{i}" for name, i in CUBE_TAG_IDS.items()}

# A tf lookup: (base_frame, tag_frame) -> (translation[3], quat_xyzw[4]); raises if no tf yet.
TfLookup = Callable[[str, str], Tuple[Tuple[float, ...], Tuple[float, ...]]]


class AprilTagPoseSource(ObjectPoseSource):
    """Cube poses (base frame) from apriltag_ros tf, via an injectable tf lookup."""

    def __init__(self, half_edge: float, base_frame: str = "panda_link0",
                 tf_lookup: Optional[TfLookup] = None, node=None):
        """half_edge: half the cube edge (m). tf_lookup injectable for tests; if None a tf2
        buffer/listener is built on `node` (a live rclpy node) lazily."""
        self.half_edge = float(half_edge)
        self.base_frame = base_frame
        self._tf_lookup = tf_lookup
        self._node = node
        self._last: Dict[str, Tuple[Pose, float]] = {}

    def _lookup(self) -> TfLookup:
        if self._tf_lookup is not None:
            return self._tf_lookup
        self._tf_lookup = _build_tf2_lookup(self._node)
        return self._tf_lookup

    def get_pose(self, name: str) -> Pose:
        frame = CUBE_TAG_FRAMES[name]
        try:
            translation, quat = self._lookup()(self.base_frame, frame)
            pose = cube_center_pose(translation, quat, self.half_edge)
            self._last[name] = (pose, time.time())
        except LookupError:
            pass  # tag not visible this frame; fall through to cached
        if name not in self._last:
            raise RuntimeError(f"AprilTag for {name!r} ({frame}) never seen; "
                               "check the camera/apriltag node and tag visibility")
        return self._last[name][0]

    def last_seen(self, name: str) -> Optional[float]:
        entry = self._last.get(name)
        return None if entry is None else entry[1]


def _build_tf2_lookup(node) -> TfLookup:
    """Default tf2-backed lookup. Imported lazily so the module loads without ROS."""
    import rclpy
    from tf2_ros import Buffer, TransformListener

    buffer = Buffer()
    TransformListener(buffer, node)

    def lookup(base_frame: str, tag_frame: str):
        rclpy.spin_once(node, timeout_sec=0.02)
        from rclpy.time import Time
        try:
            tf = buffer.lookup_transform(base_frame, tag_frame, Time())
        except Exception as exc:  # tf2 raises LookupException/ExtrapolationException
            raise LookupError(tag_frame) from exc
        tr = tf.transform.translation
        rot = tf.transform.rotation
        return ((tr.x, tr.y, tr.z), (rot.x, rot.y, rot.z, rot.w))

    return lookup
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_apriltag_pose.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add mile_franka/pose/apriltag.py tests/test_apriltag_pose.py
git commit -m "pose: AprilTagPoseSource (tf2 lookup, staleness fallback, injectable)"
```

---

### Task 4: `RosPoseStampedSource` generic seam (FoundationPose-ready)

**Files:**
- Create: `mile_franka/pose/ros_posestamped.py`
- Test: `tests/test_ros_posestamped.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ros_posestamped.py
import numpy as np
import pytest

from mile_franka.pose.ros_posestamped import RosPoseStampedSource


def test_get_pose_returns_latest_for_name():
    poses = {"object_pose": ([1.0, 2.0, 3.0], [0.0, 0.0, 0.0, 1.0])}
    src = RosPoseStampedSource(topics={"widget": "object_pose"},
                               reader=lambda topic: poses[topic])
    pose = src.get_pose("widget")
    np.testing.assert_allclose(pose.position, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(pose.orientation, [0.0, 0.0, 0.0, 1.0])


def test_get_pose_raises_until_message_arrives():
    src = RosPoseStampedSource(topics={"widget": "object_pose"},
                               reader=lambda topic: None)
    with pytest.raises(RuntimeError):
        src.get_pose("widget")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ros_posestamped.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mile_franka.pose.ros_posestamped'`

- [ ] **Step 3: Write minimal implementation**

```python
# mile_franka/pose/ros_posestamped.py
"""Generic ObjectPoseSource over geometry_msgs/PoseStamped topics.

The seam for any ROS pose publisher: AprilTag-as-PoseStamped, or hucebot's FoundationPose
node (same message type), drops in by pointing `topics` at it. The reader is injectable so
the mapping logic tests without ROS. Messages are assumed already in the robot base frame
(publish them in panda_link0, or run the publisher with the calibration static transform).
"""
from __future__ import annotations

from typing import Callable, Dict, Optional

from mile_franka.pose.base import ObjectPoseSource, Pose

# reader(topic) -> (translation[3], quat_xyzw[4]) for the latest msg, or None if none yet.
PoseReader = Callable[[str], Optional[tuple]]


class RosPoseStampedSource(ObjectPoseSource):
    def __init__(self, topics: Dict[str, str], reader: Optional[PoseReader] = None,
                 node=None, base_frame: str = "panda_link0"):
        self.topics = topics
        self.base_frame = base_frame
        self._reader = reader
        self._node = node

    def _read(self) -> PoseReader:
        if self._reader is not None:
            return self._reader
        self._reader = _build_subscriber_reader(self._node, self.topics, self.base_frame)
        return self._reader

    def get_pose(self, name: str) -> Pose:
        topic = self.topics[name]
        latest = self._read()(topic)
        if latest is None:
            raise RuntimeError(f"no PoseStamped received yet on {topic!r} for {name!r}")
        translation, quat = latest
        return Pose(tuple(float(v) for v in translation),
                    tuple(float(v) for v in quat))


def _build_subscriber_reader(node, topics, base_frame) -> PoseReader:
    """Default rclpy reader caching the latest PoseStamped per topic. Lazy ROS import."""
    import rclpy
    from geometry_msgs.msg import PoseStamped

    latest: Dict[str, tuple] = {}

    def _make_cb(topic):
        def _cb(msg: "PoseStamped"):
            p, o = msg.pose.position, msg.pose.orientation
            latest[topic] = ((p.x, p.y, p.z), (o.x, o.y, o.z, o.w))
        return _cb

    for topic in set(topics.values()):
        node.create_subscription(PoseStamped, topic, _make_cb(topic), 1)

    def reader(topic):
        rclpy.spin_once(node, timeout_sec=0.02)
        return latest.get(topic)

    return reader
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_ros_posestamped.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add mile_franka/pose/ros_posestamped.py tests/test_ros_posestamped.py
git commit -m "pose: generic RosPoseStampedSource seam (FoundationPose-ready)"
```

---

### Task 5: Calibration template + apriltag node config + gitignore

**Files:**
- Create: `config/camera_calib.example.yaml`
- Create: `config/apriltag.yaml`
- Modify: `.gitignore`

- [ ] **Step 1: Write the example calibration template**

```yaml
# config/camera_calib.example.yaml
# Copy to config/camera_calib.yaml and fill with calibrate_camera.py output (eye-to-hand).
# camera_to_base: 4x4 homogeneous transform, p_base = T @ p_cam (camera frame in base frame).
# The identity below is a PLACEHOLDER and will not give correct poses.
intrinsics: {fx: 615.0, fy: 615.0, cx: 320.0, cy: 240.0}
camera_to_base:
  - [1.0, 0.0, 0.0, 0.0]
  - [0.0, 1.0, 0.0, 0.0]
  - [0.0, 0.0, 1.0, 0.0]
  - [0.0, 0.0, 0.0, 1.0]
```

- [ ] **Step 2: Write the apriltag_ros node config**

```yaml
# config/apriltag.yaml — params for the apriltag_ros (ROS 2) detector node.
# Frame names "<family>:<id>" must match CUBE_TAG_FRAMES in mile_franka/pose/apriltag.py.
apriltag:
  ros__parameters:
    family: 36h11
    size: 0.036            # printed tag size (m) from generate_cube_tags.py
    detector:
      threads: 2
      decimate: 2.0
    tag:
      ids: [0, 1]
      frames: ["tag36h11:0", "tag36h11:1"]
```

- [ ] **Step 3: Gitignore the real calibration file**

Add these lines to `.gitignore`:

```
config/camera_calib.yaml
```

- [ ] **Step 4: Verify the template loads with the Task 1 loader**

Run: `python -c "from mile_franka.pose.calibration import load_camera_calibration; print(load_camera_calibration('config/camera_calib.example.yaml').camera_params)"`
Expected: `(615.0, 615.0, 320.0, 240.0)`

- [ ] **Step 5: Commit**

```bash
git add config/camera_calib.example.yaml config/apriltag.yaml .gitignore
git commit -m "pose: calibration template + apriltag_ros config; gitignore real calib"
```

---

### Task 6: Launch file — RealSense + apriltag + calibration static transform

**Files:**
- Create: `launch/apriltag_realsense.launch.py`

- [ ] **Step 1: Write the launch file**

```python
# launch/apriltag_realsense.launch.py
"""Bring up the third-person RealSense, apriltag_ros, and the camera->base static transform.

Static transform args come from config/camera_calib.yaml (see CameraCalibration). Run on the
same DDS graph as the controller so AprilTagPoseSource (tf2) can chain base -> camera -> tag.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

from mile_franka.pose.calibration import load_camera_calibration

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CALIB = os.path.join(REPO, "config", "camera_calib.yaml")
APRILTAG_CFG = os.path.join(REPO, "config", "apriltag.yaml")
BASE_FRAME = "panda_link0"
CAMERA_FRAME = "camera_color_optical_frame"


def generate_launch_description():
    calib = load_camera_calibration(CALIB)
    static_tf_args = calib.static_transform_args(BASE_FRAME, CAMERA_FRAME)

    return LaunchDescription([
        Node(package="realsense2_camera", executable="realsense2_camera_node",
             name="camera", parameters=[{"enable_color": True, "enable_depth": False}]),
        Node(package="apriltag_ros", executable="apriltag_node", name="apriltag",
             remappings=[("image_rect", "/camera/color/image_raw"),
                         ("camera_info", "/camera/color/camera_info")],
             parameters=[APRILTAG_CFG]),
        Node(package="tf2_ros", executable="static_transform_publisher",
             name="camera_to_base_static_tf",
             arguments=["--x", static_tf_args[0], "--y", static_tf_args[1],
                        "--z", static_tf_args[2], "--qx", static_tf_args[3],
                        "--qy", static_tf_args[4], "--qz", static_tf_args[5],
                        "--qw", static_tf_args[6], "--frame-id", static_tf_args[7],
                        "--child-frame-id", static_tf_args[8]]),
    ])
```

- [ ] **Step 2: Byte-compile to verify syntax (no ROS import at module top beyond launch)**

Run: `python -m py_compile launch/apriltag_realsense.launch.py && echo OK`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add launch/apriltag_realsense.launch.py
git commit -m "pose: launch realsense2_camera + apriltag_ros + calibration static tf"
```

---

### Task 7: Image dependencies (ROS camera + apriltag packages)

**Files:**
- Modify: `docker/Dockerfile:10` (the existing apt-get install layer)

- [ ] **Step 1: Add the two ROS packages to the apt install line**

In `docker/Dockerfile`, extend the existing `apt-get install` (currently installs `xvfb ffmpeg ...`) to also install:

```dockerfile
        ros-humble-realsense2-camera \
        ros-humble-apriltag-ros \
```

(Append these two lines to the existing `RUN apt-get update -qq && apt-get install -y --no-install-recommends ...` block, before its cleanup `&&`.)

- [ ] **Step 2: Rebuild the image**

Run: `make build`
Expected: build succeeds; the two packages install without dependency conflicts.

- [ ] **Step 3: Verify the packages are importable/available in the container**

Run: `make shell` then inside: `ros2 pkg executables apriltag_ros && ros2 pkg executables realsense2_camera`
Expected: lists `apriltag_ros apriltag_node` and `realsense2_camera realsense2_camera_node`.

- [ ] **Step 4: Commit**

```bash
git add docker/Dockerfile
git commit -m "docker: add realsense2_camera + apriltag_ros for the real pose pipeline"
```

---

### Task 8: Make verbs for tests and launch

**Files:**
- Modify: `Makefile`

- [ ] **Step 1: Add the verbs**

Add to `Makefile`:

```makefile
pose-test:   ## run the pose-layer unit tests (no ROS/hardware needed)
	pytest tests/test_calibration.py tests/test_apriltag_pose.py tests/test_ros_posestamped.py -v

apriltag-up: ## launch realsense2_camera + apriltag_ros + calibration static tf (in container)
	ros2 launch $(PWD)/launch/apriltag_realsense.launch.py
```

- [ ] **Step 2: Run the test verb**

Run: `make pose-test`
Expected: all pose-layer tests pass (11 passed).

- [ ] **Step 3: Commit**

```bash
git add Makefile
git commit -m "make: add pose-test and apriltag-up verbs"
```

---

### Task 9: Live bring-up verification (hardware — checklist, not unit-testable)

**Files:** none (verification against the real camera + controller graph).

- [ ] **Step 1: Calibrate** — run the calibration capture script (follow-on plan) to produce `config/camera_calib.yaml`; confirm it loads: `python -c "from mile_franka.pose.calibration import load_camera_calibration; load_camera_calibration('config/camera_calib.yaml')"`.

- [ ] **Step 2: Launch** — `make apriltag-up`; confirm with `ros2 topic echo /tf` that `tag36h11:0` / `tag36h11:1` frames appear when the cubes are in view, and `ros2 run tf2_ros tf2_echo panda_link0 tag36h11:0` returns a transform.

- [ ] **Step 3: Validate accuracy** — place a tag at a tape-measured point in the workspace; in a `make shell` python session build `AprilTagPoseSource(half_edge=0.025, node=<rclpy node>)` and assert `get_pose(BOTTOM_CUBE).position` is within ~5–10 mm of the measured point. If the center lands in *front* of the cube face, flip the `half_edge` sign in `cube_center_pose` (the `CONFIRM@bringup` note).

- [ ] **Step 4: Record** the measured accuracy and any offset-sign correction in `docs/superpowers/notes/2026-06-15-phase2-bringup-checklist.md`.

---

## Self-Review

**Spec coverage (§5.1–5.2):** §5.1 `RosPoseStampedSource` → Task 4. §5.2 `AprilTagPoseSource` (RealSense via ROS, tag36h11 ids 0/1, tag_size 0.036, tag→center offset, staleness) → Tasks 2–3, 5–6. Deps/passthrough (realsense + apriltag packages) → Task 7. Calibration file consumed by the source → Tasks 1, 5. Grasp-transform carry (post-grasp occlusion) is intentionally deferred to the env-builder plan (it lives in `FrankaEnv`/scripted policy, not the pose source) — noted here so it is not lost.

**Placeholder scan:** No TBD/TODO/"handle edge cases"; every code step has complete code; every command has expected output.

**Type consistency:** `Pose` (from `mile_franka.pose.base`) used uniformly. `cube_center_pose(translation, quat_xyzw, half_edge)` signature identical in Tasks 2 and 3. `CUBE_TAG_FRAMES` defined in Task 3 and referenced in Task 3 tests + Task 6 config + Task 9. `CameraCalibration.static_transform_args(parent, child)` defined Task 1, used Task 6. `TfLookup`/`PoseReader` reader signatures match their injected test doubles.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-18-apriltag-ros-pose.md`. Two execution options:

1. **Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
