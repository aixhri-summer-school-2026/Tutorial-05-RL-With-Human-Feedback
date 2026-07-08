# Part 2 — Franka Environment Smoke Test (fake backend)

**Goal:** confirm the Franka block-stacking environment loads and runs end-to-end on the
simplest possible backend — a kinematic "fake world" with **no MuJoCo physics and no ROS**.
This is a 5-second sanity check before you spend time booting the real simulator in
[05-franka-sim.md](05-franka-sim.md).

---

## Why a fake backend exists

`FrankaEnv` (`mile_franka/envs/franka_env.py`) is **backend-agnostic**: the exact same gym
environment runs against three different worlds, and only the backend + pose source swap.

| Backend | Physics | ROS | Used for |
|---|---|---|---|
| **Fake** (`FakeRobotBackend`) | none — pure kinematics | no | dev, CI, this smoke test |
| **Sim** (`MultipandaRosBackend` + MuJoCo) | MuJoCo | yes | [05-franka-sim.md](05-franka-sim.md) |
| **Real** (`MultipandaRosBackend` + FR3) | real hardware | yes | [06-franka-real.md](06-franka-real.md) |

The fake backend (`mile_franka/envs/fake_backend.py`) is a `FakeWorld` that moves the
end-effector straight toward the commanded Cartesian target and picks/places cubes when the
gripper closes over them. It imports no `rclpy`, so it runs anywhere the Python package
imports.

---

## The task and the interface

Two 5 cm cubes sit on a table; the goal is to **stack the top cube on the bottom cube** and
hold it stable. This is the same task in all three backends.

| | |
|---|---|
| **Action** | 4-D: `[Δx, Δy, Δz, gripper]` — Cartesian end-effector delta + gripper command |
| **Reduced observation** | 9-D: `[ee_xyz(3), gripper_width(1), top_cube_xyz(3), bottom_cube_xy(2)]` |
| **Wrapped observation** | `FrameStack(10)` → **90-D** (10 frames of history) |
| Privileged frame | `env.unwrapped.privileged_frame()` → full 18-D ground-truth (for scripted planners) |
| Success | cubes stacked and stable for `success_stable_steps` → `info['success'] == 1` |

The reduced 9-D obs deliberately drops the cube quaternions and the bottom cube's z (it's on
the table) — the policy only needs what matters for stacking. Scripted policies read the
richer `privileged_frame()`, but the learned policy sees only the 90-D stacked vector.

---

## Running it

```bash
make tutorial-fake
```

This runs `scripts/smoke_franka_env.py`, which builds `FrankaEnv` on the fake backend and
drives a hard-coded pick-and-place (descend → grasp → lift → carry → stack → release),
then asserts the episode reports success.

**Expected output:**
```
smoke_franka_env ok
```

If you see `smoke_franka_env ok`, the environment, observation wrapping, action scaling,
and success detection all work — you're clear to boot the real simulator. If it errors or
the assertion fails, call an instructor **before** moving on; a broken env here will only
be harder to debug once ROS and MuJoCo are in the picture.

---

## What this does *not* test

The fake backend is intentionally dumb: it has no dynamics, no contact, no controller. So
this test does **not** exercise the Cartesian-impedance controller, MuJoCo physics, cube
tracking, or teleop — those all come online in Part 3. What it *does* guarantee is that the
Python side (env, obs/action shapes, reward/success logic, the `mile_franka` package) is
intact.

## Next step

→ **[05-franka-sim.md](05-franka-sim.md)** — boot the MuJoCo sim, learn the teleop, and
become the human in the MILE loop.
