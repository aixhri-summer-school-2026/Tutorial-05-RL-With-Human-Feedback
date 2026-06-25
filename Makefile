DC      := docker compose -f docker/docker-compose.yml
ENVSH   := source scripts/in_container_env.sh
RUN      = $(DC) exec sim bash -lc '$(ENVSH) && $(1)'
RUND     = $(DC) exec -d sim bash -lc '$(ENVSH) && $(1)'
# Kill any leftover process still publishing to the arm (prior run or stale rclpy context).
KILLCLIENTS = self=$$$$; pgrep -f "train_mile.py|eval_base_policy_(sim|real)|eval_mile|franka_sim_rollout_record|build_base_policy|collect_synthetic_interventions" | grep -vx $$self | xargs -r kill 2>/dev/null; sleep 2; true
TS      := $(shell date -u +%Y%m%dT%H%M%S)
DEMOS   ?= output_dir/franka/sim_demos_mediocre.npz   ## base-policy input; override: make base-policy DEMOS=path.npz
GRIP_FORCE  ?= 40
CLOSE_WIDTH ?= 0.0
OPEN_WIDTH  ?= 0.08
# Resolve which gripper action is live and bail if none found.
GRIP_NS = ns=$$(ros2 action list 2>/dev/null | grep -E "/grasp$$" | head -1 | sed "s|/grasp||"); if [ -z "$$ns" ]; then echo "No gripper action server found -- is the controller/sim running?"; exit 1; fi; echo "gripper: $$ns"
FRANKA_CTR    ?= franka_ros2_humble
ROBOT_IP      ?= 169.254.202.10
LOAD_GRIPPER  ?= true
FRANKA_SRC    := source /opt/ros/humble/setup.bash && source /ros2_ws/install/setup.bash && export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

.PHONY: build up down shell sim-up sim-gui collect-mediocre collect-expert base-policy mile mile-real spacemouse-check joystick-check eval-base eval-mile pose-test franka-up franka-shell apriltag-up calibrate-camera close-gripper open-gripper eval-real view-tags view-twin real-home-smoke

build:                       ## build the image (classic builder: base hucebot:franka-humble is local-only, not on a registry)
	DOCKER_BUILDKIT=0 $(DC) build

up:                          ## start the persistent sim service
	$(DC) up -d

down:                        ## stop the service
	$(DC) down

shell:                       ## interactive shell, env sourced, cd'd into the repo
	$(DC) exec sim bash -lc '$(ENVSH) && exec bash'

sim-up:                      ## launch the stacking sim headless (detached)
	$(call RUND,bash scripts/sim_up.sh)
	@echo "sim launching headless; give it ~10s, then check: make collect-mediocre"

sim-gui:                     ## launch the stacking sim with a LIVE window on the host display
	@echo "Host prereq (once per login): xhost +local:root"
	$(DC) exec -e DISPLAY=$$DISPLAY sim bash -lc '$(ENVSH) && bash scripts/sim_gui.sh'

collect-mediocre:            ## MEDIOCRE demos (feeds the base policy) -> sim_demos_mediocre.npz
	$(call RUN,python3 scripts/franka_sim_rollout_record.py \
	    --episodes 100 --mediocre true --require_success true --max_attempts 150 \
	    --max_steps 500 --video_start_hold 0.5 --video_end_hold 0.5 \
	    --out_dir output_dir/franka/rollouts_mediocre_$(TS) \
	    --data output_dir/franka/sim_demos_$(TS).npz && \
	  cp output_dir/franka/sim_demos_$(TS).npz output_dir/franka/sim_demos_mediocre.npz)

collect-expert:              ## PERFECT (successful-only) demos -> sim_demos_expert.npz
	$(call RUN,python3 scripts/franka_sim_rollout_record.py \
	    --episodes 100 --mediocre false --require_success true \
	    --max_steps 500 --video_start_hold 0.5 --video_end_hold 0.5 \
	    --out_dir output_dir/franka/rollouts_expert_$(TS) \
	    --data output_dir/franka/sim_demos_$(TS).npz && \
	  cp output_dir/franka/sim_demos_$(TS).npz output_dir/franka/sim_demos_expert.npz)

base-policy:                 ## BC-train the (mediocre) base policy offline; override input with DEMOS=path.npz
	$(call RUN,python3 scripts/build_base_policy.py \
	  --demos $(DEMOS) --bc_batch_size 256 --bc_ent_weight 0.0 --eval_episodes 0 \
	  --save_path trained_models/franka/base_policy)

mile:                        ## iterative MILE run in sim (config/franka_sim.yaml)
	$(call RUN,cd scripts && python3 train_mile.py --config ../config/franka_sim.yaml)

mile-real:                   ## iterative MILE run on the real Franka (needs controller + apriltag-up running). MILE_REAL_STACK=fr3|multipanda (default fr3). MILE_APPLY_SIM_GAINS=1 to track against the lab sim.
	$(DC) exec -e MILE_REAL_STACK=$${MILE_REAL_STACK:-fr3} -e MILE_APPLY_SIM_GAINS sim bash -lc '$(ENVSH) && $(KILLCLIENTS); cd scripts && python3 train_mile.py --config ../config/franka_real.yaml'

spacemouse-check:            ## print live SpaceMouse deflection (sanity check; Ctrl-C to stop)
	$(call RUN,python3 scripts/spacemouse_check.py)

joystick-check:              ## print live gamepad axes/buttons (sanity check; Ctrl-C to stop)
	$(call RUN,python3 scripts/joystick_check.py)

eval-base:                   ## run the BC base policy in sim -> success rate + per-episode videos
	$(call RUN,python3 scripts/eval_base_policy_sim.py --episodes 10 \
	  --video_dir output_dir/franka/eval_videos_$(TS))

eval-mile:                   ## run the MILE-trained policy in sim -> success rate (compare with eval-base)
	$(call RUN,python3 scripts/eval_base_policy_sim.py --episodes 10 \
	  --policy output_dir/franka/policy \
	  --video_dir output_dir/franka/eval_mile_videos_$(TS))

tune-cost:                   ## tune MILE intervention cost/scale from DATASET; override POLICY/MENTAL_MODEL
	$(call RUN,python3 scripts/tune_intervention_cost.py \
	  --dataset $${DATASET:-output_dir/franka/accumulated_dataset_round0.pkl} \
	  --policy $${POLICY:-trained_models/franka/base_policy} \
	  --mental_model $${MENTAL_MODEL:-trained_models/franka/base_policy} \
	  --cost_grid $${COST_GRID:-110:170:5} \
	  --scale_grid $${SCALE_GRID:-125,150,175,200})

pose-test:                   ## run the pose-layer unit tests (no ROS/hardware needed)
	pytest tests/test_calibration.py tests/test_apriltag_pose.py tests/test_ros_posestamped.py -v

franka-up:                   ## launch the real FR3 controller stack (franka_ros2 container, CycloneDDS, foreground). Override ROBOT_IP=.. LOAD_GRIPPER=false
	@docker ps --format '{{.Names}}' | grep -qx $(FRANKA_CTR) || { echo "Container $(FRANKA_CTR) not running -- start it: docker compose -f ~/franka_ros2/docker-compose.yml up -d"; exit 1; }
	docker exec -it $(FRANKA_CTR) bash -lc '$(FRANKA_SRC) && ros2 launch mile_franka_controllers mile_bringup.launch.py robot_ip:=$(ROBOT_IP) load_gripper:=$(LOAD_GRIPPER)'

franka-shell:                ## open a shell in the franka_ros2 container (env + CycloneDDS sourced)
	docker exec -it $(FRANKA_CTR) bash -lc '$(FRANKA_SRC) && exec bash'

apriltag-up:                 ## launch realsense2_camera + apriltag_ros + calibration static tf (foreground). MILE_REAL_STACK=fr3|multipanda (default fr3).
	$(DC) exec -e MILE_REAL_STACK=$${MILE_REAL_STACK:-fr3} -e MILE_CAMERA_CALIB sim bash -lc '$(ENVSH) && self=$$$$; pgrep -f "apriltag_realsense.launch.py|apriltag_node|realsense2_camera_node|static_transform_publisher.*camera_to_base" | grep -vx $$self | xargs -r kill 2>/dev/null; sleep 2; ros2 launch mile_franka/launch/apriltag_realsense.launch.py'

calibrate-camera:            ## eye-to-hand camera calibration → MJPEG preview at http://localhost:8080 (needs controller + apriltag-up running). MILE_REAL_STACK=fr3|multipanda (default fr3).
	$(DC) exec -e MILE_REAL_STACK=$${MILE_REAL_STACK:-fr3} -e MILE_CAMERA_CALIB -e PYTHONUNBUFFERED=1 sim bash -lc '$(ENVSH) && python3 -u scripts/calibrate_camera.py'

close-gripper:               ## close the gripper to clamp (e.g. the calib board); override GRIP_FORCE=.. CLOSE_WIDTH=..
	$(DC) exec sim bash -lc '$(ENVSH) && $(GRIP_NS); ros2 action send_goal $$ns/grasp franka_msgs/action/Grasp "{width: $(CLOSE_WIDTH), speed: 0.05, force: $(GRIP_FORCE), epsilon: {inner: 0.08, outer: 0.08}}"'

open-gripper:                ## open the gripper (release); override OPEN_WIDTH=..
	$(DC) exec sim bash -lc '$(ENVSH) && $(GRIP_NS); ros2 action send_goal $$ns/grasp franka_msgs/action/Grasp "{width: $(OPEN_WIDTH), speed: 0.05, force: $(GRIP_FORCE), epsilon: {inner: 0.08, outer: 0.08}}"'

view-tags:                   ## MJPEG stream with AprilTag overlay → open http://localhost:8080 (needs apriltag-up). MILE_REAL_STACK=fr3|multipanda (default fr3).
	$(DC) exec -e MILE_REAL_STACK=$${MILE_REAL_STACK:-fr3} sim bash -lc '$(ENVSH) && python3 -u scripts/view_camera_tags.py'

eval-real:                   ## run policy eval on the real Franka. MILE_REAL_STACK=fr3|multipanda (default fr3). Both stacks must run on CycloneDDS (the container default) so the mile client sees their topics+services+actions. MILE_APPLY_SIM_GAINS=1 tracks against the lab sim.
	$(DC) exec -e DISPLAY=$$DISPLAY -e MILE_REAL_STACK=$${MILE_REAL_STACK:-fr3} -e MILE_CONTROLLER -e MILE_GRASP_ACTION -e MILE_APPLY_SIM_GAINS sim bash -lc '$(ENVSH) && echo "real stack=$${MILE_REAL_STACK:-fr3} RMW=$$RMW_IMPLEMENTATION"; $(KILLCLIENTS); python3 scripts/eval_base_policy_real.py'

real-home-smoke:             ## operator-gated real Franka home + small Cartesian square; STEP=0.025 by default. MILE_REAL_STACK=fr3|multipanda (default fr3).
	$(DC) exec -e MILE_REAL_STACK=$${MILE_REAL_STACK:-fr3} -e MILE_APPLY_SIM_GAINS -e MILE_CONTROLLER -e MILE_SUBSTEP_M sim bash -lc '$(ENVSH) && $(KILLCLIENTS); python3 scripts/franka_real_home_smoke.py --step $${STEP:-0.1}'

view-twin:                   ## live MuJoCo digital twin of the real workspace on the host display (read-only; needs controller + apriltag-up; safe in parallel with mile-real/eval-real). MILE_REAL_STACK=fr3|multipanda (default fr3).
	@echo "Host prereq (once per login): xhost +local:root"
	@if [ "$$DISPLAY" != "$${DISPLAY#localhost:}" ]; then \
		echo "[view-twin] using host display :1 (SSH-forwarded $$DISPLAY unreachable from container)"; \
		TWIN_DISPLAY=:1; \
	else \
		TWIN_DISPLAY=$$DISPLAY; \
	fi; \
	$(DC) exec -e DISPLAY=$$TWIN_DISPLAY -e MILE_REAL_STACK=$${MILE_REAL_STACK:-fr3} sim bash -lc '$(ENVSH) && python3 scripts/view_cubes_mujoco.py'

fetch-artifacts:             ## trained models are in the repo; no download needed
	@echo "Trained models are included in the repository (trained_models/). No download needed."
	@ls -l trained_models/initial_policy trained_models/expert_policy trained_models/gt_mental_model trained_models/warm_started_mental_model trained_models/franka/base_policy

tutorial-check:              ## assert imports + artifacts are present (run in container)
	$(call RUN,python3 scripts/tutorial_check.py)

tutorial-check-loss:         ## green-light test for the MILE-loss exercise
	python3 -m pytest tests/test_loss_exercise.py -v

tutorial-metaworld:          ## Tier 1: run the MetaWorld synthetic loop (uses your loss)
	cd scripts && python3 tutorial_train.py --config ../config/tutorial_metaworld.yaml

tutorial-collect-train:      ## Tier 3: collect interventions (keyboard) + train (uses your loss)
	cd scripts && python3 tutorial_train.py --config ../config/tutorial_franka.yaml

tutorial-fake:               ## Tier 2: run the mediocre base policy on the fake backend
	python3 scripts/smoke_franka_env.py
