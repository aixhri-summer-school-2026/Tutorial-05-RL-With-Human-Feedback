# MILE Summer-School Tutorial — 2-Hour Hands-On Flow (Sim Ladder → Real FR3)

**Date:** 2026-06-24
**Author:** rayray2002
**Status:** Design (for review)

## 1. Context & goal

Deliver MILE (Model-based Intervention Learning, https://liralab.usc.edu/mile/) as a
**2-hour hands-on tutorial** at the AI–Human–Robot summer school
(https://aihumanrobot.sciencesconf.org). The tutorial is **delivered in France** (their
real FR3 + HTC Vive + AprilTag); it is **built and rehearsed in the home lab** against the
multipanda MuJoCo sim.

**The one thing every participant should leave having done:** personally run the full MILE
loop in sim — collect their own interventions with teleop, train, and watch their policy
improve. The real FR3 is a parallel, staffed station everyone rotates through, not a gate.

This spec defines the participant-facing **flow** and the **build items** (home-lab prep)
needed to produce it. It does not change the upstream MILE method; the only code touched in
`mile/` is a tutorial scaffold around the loss (Section 6), with production code intact.

## 2. Audience & logistics (fixed constraints)

- **BYOL**: participants bring Linux laptops (Ubuntu 22+), NVIDIA GPU, ethernet, 50–100 GB
  free. Mac/Windows unsupported; those users **pair up**. Two tower machines with RTX 5090
  are available for heavier training.
- **Setup is homework**: participants **build/import the single docker image themselves
  beforehand** (Section 4). The 2 hours assume a working image. A prebuilt image is also
  distributed at the venue (USB/LAN) as a fallback; weak laptops pair up or use the towers.
- **One real FR3**, shared, in France. It runs as a **parallel queue** during the session.
- **Teleop default is the keyboard** on the participant's own laptop, for both sim and the
  real robot. The **HTC Vive is an optional upgrade** wired on arrival; if it comes up, the
  real-robot station flips `intervener: keyboard → vive` with no other change.

## 3. The 4-tier ladder (one image, two backends)

The tutorial is a ladder from "method on autopilot" to "you are the human" to "real
hardware". All four tiers run inside **one docker image** (Section 4).

| Tier | Backend | Human? | What the participant does | Point |
|---|---|---|---|---|
| **1. MetaWorld peg-insert** | MetaWorld sim | **No** — downloaded expert | Run reduced iterative loop; watch success climb across rounds | "MILE works (the paper), fully automated" |
| **2. Franka fake backend** | kinematic fake | No | Run the pre-baked mediocre base policy (~0.5) | Same MILE code, new task; see obs/action + the one engineered failure mode; headless fallback |
| **3. Franka MuJoCo sim** | multipanda MuJoCo | **Yes — you** | Keyboard teleop: implement the loss, collect interventions, train, eval before/after | The visceral "I am the expert now" loop |
| **4. Real FR3** | hucebot controller + hardware | shared | Keyboard (or Vive) teleop a real stack; same controller | Sim→real payoff, live |

Tier 1 uses MetaWorld's synthetic human (the **downloaded expert** models +
`computational_intervention_model`, `auto_eval: true`) — no teleop, no human. Tiers 2–4 are
the `mile_franka` stack; Tier 4 is the same Cartesian-impedance controller as Tier 3 with
only the backend, pose source, and (optionally) teleop device swapped.

## 4. Single unified docker image (decision: option A)

Today MetaWorld lives in a host conda env and the Franka stack lives in docker, because of a
hard pin conflict on the Python `mujoco` package: MetaWorld-v2 needs `mujoco==2.3.7`
(`mujoco<3`), while `docker/requirements-mile.txt` pins `mujoco>=3.1.0,<3.2.0`.

**Key finding:** the Franka *sim loop* does **not** import the Python `mujoco` package — it
talks to the multipanda sim over `mujoco_ros_msgs` (ROS/C++). Python `mujoco` is imported in
only two places, `mile_franka/viz/mujoco_twin.py` and `scripts/view_cubes_mujoco.py` — the
**read-only twin viewer** (`make view-twin`), a real-robot debug aid.

**Decision (A):** downgrade the image's Python `mujoco` to **2.3.7** and add the pinned
MetaWorld v2 commit, producing **one image that runs all four tiers**. The **twin viewer is
kept** — its two modules are adapted to run on `mujoco==2.3.7` (minor API differences from
3.1 only; `mujoco.viewer` exists in both). The Franka sim loop is unaffected (it uses
`mujoco_ros`, not Python `mujoco`).

Concretely:
- `docker/requirements-mile.txt`: change `mujoco>=3.1.0,<3.2.0` → `mujoco==2.3.7`; add
  `metaworld @ git+https://github.com/Farama-Foundation/Metaworld.git@c822f28f582ba1ad49eb5dcf61016566f28003ba`.
- `docker/Dockerfile`: extend the build-time import assert to include `import metaworld`;
  remove the "MetaWorld-free" comment.
- `mile_franka/viz/mujoco_twin.py` + `scripts/view_cubes_mujoco.py`: adapt to the
  `mujoco==2.3.7` Python API so `make view-twin` still works in this image.
- **Do not** `gdown` at image-build time (every homework build would hammer Drive's
  per-file quota). Artifacts are decoupled from the image and mounted as a volume
  (Section 4.1).
- Validate at home: both the MetaWorld synthetic loop and the Franka sim loop run in the
  rebuilt image. (MetaWorld-v2 on `mujoco==2.3.7` is its native version — low risk.)

### 4.1 Artifact distribution (avoid the Drive rate limit)

The Drive download (`trained_models.zip`, **5.64 MB**, sha256 `48020ef5…`, measured
2026-06-24) is **4 SB3 models and no dataset**:

| File | Size | Type | obs | act |
|---|---|---|---|---|
| `initial_policy` | 2.1M | SAC | Box(156,) f64 | Box(4,) |
| `expert_policy` | 2.1M | SAC | Box(156,) | Box(4,) |
| `gt_mental_model` | 849K | BC ActorCritic | Box(156,) | Box(4,) |
| `warm_started_mental_model` | 849K | BC ActorCritic | Box(156,) | Box(4,) |

`net_arch=[256,256]` throughout. **MetaWorld obs is 156-dim** (peg-insert 39 × `FrameStack(4)`,
`float64`) and the policies are **SAC** — distinct from the Franka pipeline (90-dim,
`FrameStack(10)`, `bc`). Tier 1 therefore keeps MetaWorld's *own* obs/policy pipeline; the
only genuinely shared code is `mile_cont_loss_fn` (the TODO — dim-agnostic, operates on
action/ν tensors), the 4-DoF action, and `[256,256]`.

The peg-insert **intervention dataset is synthesized at runtime** by `collect_synthetic_data`
from these four models, so no peg-insert dataset is shipped. The only datasets to bundle are
the *Franka* base policy (≈0.7 MB) + seed `npz` (≈0.5 MB). Full bundle ≈ **7 MB**. The
rate-limit risk is Drive throttling request *count*, not size.

- **You hit Drive once** (home lab): `gdown 1bzKGyOmX1ZCmAWnZiq_sAFRxi3AXvm4t`, add the
  engineered-mediocre base policy + seed dataset, bundle into `tutorial-artifacts.tar` with a
  published **SHA256**.
- **Re-host as a GitHub Release asset** on the tutorial repo (robust CDN, no quota,
  versioned). Participants already clone the repo for the homework build. (Small enough to
  git-LFS instead, but a Release keeps the clone lean.)
- **Participants download once, as homework**, via a `make` verb that pulls the Release asset
  (never Drive), unpacks to the artifacts volume, and verifies the checksum.
- **Venue fallback:** the same tarball on USB / LAN, so a missed/failed homework download is
  a few-second local copy — zero internet on the day.
- `make tutorial-check` asserts the artifacts volume is present and complete (checksum), so a
  missing/partial bundle is caught at home, not at 0:05 in the session.

## 5. Teleop: keyboard default, Vive optional

- **`KeyboardDevice`** (new) implements the `TeleopDevice` ABC
  (`read() -> (action[dx,dy,dz,gripper], intervene, done, discard)`): move keys → xyz delta,
  a **clutch key** sets ν (segment toggle, mirroring `JoystickDevice` semantics), a key
  toggles the gripper, keys for done/discard. Uses an injectable raw key reader so it
  unit-tests headless like the joystick. Wired as `intervener: keyboard` in `train_mile.py`
  and the tutorial configs. Drives both sim and the real FR3 (deltas only).
- **Gamepad** (`JoystickDevice`) stays as the existing "nicer" option.
- **Vive** is wired on arrival; if it comes up, the real-robot station uses `intervener:
  vive`. Keyboard works regardless, so the Vive is pure upside.

## 6. The one guarded TODO — the MILE loss

A single fill-in-the-blank, the conceptual heart of the method, done **up front (before
Tier 1)**: **implement `mile_cont_loss_fn`** — BCE on the intervention flag ν **+** Gaussian
NLL of the human action **only on ν=1 steps**.

The loss lives in shared `mile/algorithm.py`, used by **every training tier**. So the loss a
participant writes is what trains the MetaWorld peg-insert policy in **Tier 1** *and* their
own Franka policy in **Tier 3**. Tier 1's guaranteed clean improvement is therefore the
**immediate validation that their loss is correct**; Tier 3 simply reuses it on their data.

Guards (so it never blocks the flow):
- Production `mile/algorithm.py` stays intact. The exercise is a **tutorial scaffold**: a
  copy of the function with the body blanked and a **docstring spec** (input shapes, the two
  loss terms, the ν-mask), which the tutorial config imports.
- **One-command test**: `make tutorial-check-loss` runs a unit test (known inputs → known
  loss) that goes green when correct.
- **Answer key**: a `solutions/` reference; `make tutorial-train` auto-applies it if the
  participant's version is absent/failing, so Tier 3 always proceeds.
- **Time-boxed** (~10 min). Framing: "before you run anything, implement the heart of MILE;
  you'll watch it train the paper's policy in Tier 1, then your own robot in Tier 3."

## 7. The 2-hour flow

Pre-session prep is done before the clock starts (Section 8). All commands run inside the
container (`make shell`); each tier is a single `make` verb.

- **0:00–0:10 · Land & verify.** `make up && make shell`; `make tutorial-check` (asserts
  `metaworld`/`mile`/`mile_franka` import, expert + base-policy artifacts present). 5-min
  MILE framing (ν, intervention model, joint policy + mental-model training).
- **0:10–0:22 · Implement the heart of MILE (the one TODO).** Fill in `mile_cont_loss_fn`
  (BCE on ν + Gaussian NLL on ν=1 steps); `make tutorial-check-loss` goes green. This loss
  trains *both* the Tier 1 and Tier 3 policies. Answer key auto-applies if unfinished.
- **0:22–0:40 · Tier 1 — MetaWorld, your loss on the paper benchmark (no human).**
  `make tutorial-metaworld`: synthetic human from the downloaded expert, `auto_eval: true`;
  watch success climb across 2 rounds — trained by *the loss you just wrote*. "This is the
  role you'll play next."
- **0:40–0:45 · Tier 2 — Franka fake, same code, your task (headless, quick).**
  `make tutorial-fake`: pre-baked mediocre base policy (~0.5); a 5-minute look at the 9-dim
  obs / 4-DoF action and the engineered failure mode. Also the laptop fallback if MuJoCo is
  heavy.
- **0:45–1:00 · Visual sim + learn to intervene.** `make sim-up` + `make sim-gui`;
  `make tutorial-teleop`: keyboard free-play (clutch = ν, gripper toggle) to get the feel.
- **1:00–1:30 · Tier 3 — be the human in sim.** `make eval-base` (before) →
  `make tutorial-collect` (clutch in when the policy is about to fail, 1–2 episodes) →
  `make tutorial-train` (1 round, *reusing the loss you wrote*) → `make eval-after`. The
  payoff: your interventions moved the number. A seed dataset behind the scenes ensures the
  delta shows.
- **~1:05–1:45 · Tier 4 — real FR3 station (parallel queue).** Staffed live robot;
  `make franka-up`; participants rotate in and teleop a real stack with the **keyboard**
  (or the **Vive** if wired). Sim work continues in parallel; the robot never gates the room.
- **1:45–2:00 · Wrap.** What was identical sim→real (same Cartesian-impedance controller,
  same MILE loop) vs. what swapped (Vive↔keyboard, AprilTag↔MuJoCo GT, re-tuned
  `COST_LOOKUP`). Q&A; queue tail; buffer.

## 8. France-day pre-session prep (outside the 2h)

- Mount + calibrate the D415: `make calibrate-camera` → `config/camera_calib.yaml`. (The
  one asset task that cannot be pre-done at home.)
- Launch hucebot's multipanda_ros2 controller on their FR3; confirm interface names
  (`/cartesian_impedance/equilibrium_pose`, `…/cartesian_pos_curr`, gripper action).
- Re-tune `COST_LOOKUP` for the real robot against the observed intervention rate.
- Time permitting: wire the Vive (`intervener: vive`). Keyboard works regardless.

## 9. Build items (home-lab prep — the work to produce this)

1. **Unified docker image** (Section 4): requirements + Dockerfile changes; adapt the twin
   viewer modules to `mujoco==2.3.7`; bake expert + base-policy artifacts; validate both
   loops (and `make view-twin`) in the rebuilt image.
2. **`KeyboardDevice`** (Section 5): teleop ABC impl + injectable reader + unit tests; wire
   `intervener: keyboard` in `train_mile.py`.
3. **Reduced tutorial configs**: `config_tutorial_metaworld.json` (few rounds/epochs,
   `auto_eval: true`, points at the downloaded expert; finishes ~10 min, visibly improves)
   and `config_franka_tutorial.json` (1–2 rounds, 1–2 episodes/round, fewer epochs,
   `intervener: keyboard`, `auto_eval: false`).
4. **Engineered-mediocre base policy + seed intervention dataset**: a base policy with a
   *single* fixable failure mode so a few interventions reliably move the eval number; seed
   dataset as insurance.
4b. **Artifact bundle + re-host** (Section 4.1): one-time `gdown`, assemble
   `tutorial-artifacts.tar` + SHA256, publish as a GitHub Release asset; a `make
   fetch-artifacts` verb that pulls the Release (not Drive), unpacks to the volume, and
   verifies the checksum; prepare the USB/LAN copy.
5. **Loss exercise scaffold** (Section 6): blanked function + docstring spec + answer key +
   `make tutorial-check-loss` test.
6. **Tutorial `make` verbs**: `tutorial-check`, `tutorial-metaworld`, `tutorial-fake`,
   `tutorial-teleop`, `tutorial-collect`, `tutorial-train`, `tutorial-check-loss`,
   `eval-base`/`eval-after` — thin wrappers so nobody types long commands.
7. **Participant setup doc + homework script**: build/import the image, run
   `make tutorial-check`, before the session.

## 10. Risks & mitigations

- **Franka improvement not visible from sparse human data (headline risk)** → engineer the
  base policy to one fixable failure mode; ship the seed dataset; rely on Tier 1 (MetaWorld)
  as the *guaranteed* "success improves across rounds" demonstration.
- **2h tight for 4 tiers** → Tier 2 is compressible/skippable; configs sized to finish fast;
  setup is homework; Tier 4 runs in parallel, not as a serial block.
- **Loss TODO stalls people** → docstring spec + green-light test + auto-applied answer key +
  time-box; Tier 3 always proceeds.
- **MuJoCo rendering slow on a laptop** → Tier 2 headless fallback; pairing; 5090 towers.
- **MetaWorld-v2 on mujoco 2.3.7** → native version, low risk; validated at image build via
  the import assert + a smoke run at home.
- **Single-robot bottleneck** → staffed parallel queue; the room never waits on the robot.

## 11. Acceptance (validated at home before France)

1. Unified image builds; `import metaworld, mile, mile_franka` all succeed; `make
   tutorial-check` passes; `make view-twin` still opens on `mujoco==2.3.7`.
2. `make tutorial-metaworld` finishes in ~10 min and shows success rate improving across
   rounds, trained by the participant-supplied `mile_cont_loss_fn` (or the auto-applied
   answer key).
2b. `make fetch-artifacts` pulls the bundle from the GitHub Release (not Drive), unpacks to
   the volume, and the checksum verifies; `make tutorial-check` fails clearly if the bundle
   is missing/partial.
3. `make tutorial-fake` reports a mediocre (~0.3–0.7) base-policy success.
4. `KeyboardDevice` unit tests pass; `make tutorial-teleop` reads keys and toggles ν/gripper
   in the sim window.
5. `make tutorial-check-loss` goes green with the answer key; blanked scaffold fails it.
6. End-to-end Tier 3 on the sim: `eval-base` → collect (keyboard) → train → `eval-after`
   shows a visible improvement with the engineered base policy + seed dataset.
7. The whole flow runs from the single image with no host conda env.

## 12. Out of scope

- Changing the upstream MILE method or the MetaWorld reproduction results.
- France-day robot specifics that can only be validated on their hardware (FR3 + Vive +
  calibration) — covered by pre-session prep, not the tutorial flow.
- Image-based observations / FoundationPose / peg-insertion as a Franka task.
