# MILE-on-multipanda dev verbs. One short command per action; all run inside the `sim`
# compose service with the verified headless env sourced. See docs/superpowers/specs/
# 2026-06-16-mile-docker-image-and-sim-demos-design.md.
DC      := docker compose -f docker/docker-compose.yml
ENVSH   := source scripts/in_container_env.sh
RUN      = $(DC) exec sim bash -lc '$(ENVSH) && $(1)'
RUND     = $(DC) exec -d sim bash -lc '$(ENVSH) && $(1)'
TS      := $(shell date -u +%Y%m%dT%H%M%S)
DEMOS   ?= output_dir/franka/sim_demos_mediocre.npz   ## base-policy input; override: make base-policy DEMOS=path.npz

.PHONY: build up down shell sim-up sim-gui collect-mediocre collect-expert base-policy mile mile-real spacemouse-check joystick-check eval-base eval-mile pose-test apriltag-up calibrate-camera eval-real

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

mile:                        ## iterative MILE run in sim (config_franka.json)
	$(call RUN,cd scripts && python3 train_mile.py --config ../config_franka.json)

mile-real:                   ## iterative MILE run on the real FR3 (needs controller + apriltag-up running)
	$(DC) exec sim bash -lc '$(ENVSH) && cd scripts && python3 train_mile.py --config ../config_franka_real.json'

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

pose-test:                   ## run the pose-layer unit tests (no ROS/hardware needed)
	pytest tests/test_calibration.py tests/test_apriltag_pose.py tests/test_ros_posestamped.py -v

apriltag-up:                 ## launch realsense2_camera + apriltag_ros + calibration static tf (in container)
	ros2 launch $(PWD)/launch/apriltag_realsense.launch.py

calibrate-camera:            ## run eye-to-hand camera calibration (needs controller + apriltag-up running)
	$(DC) exec -e DISPLAY=$$DISPLAY sim bash -lc '$(ENVSH) && python3 scripts/calibrate_camera.py'

eval-real:                   ## run policy eval on the real FR3 (needs controller + apriltag-up running)
	$(DC) exec -e DISPLAY=$$DISPLAY sim bash -lc '$(ENVSH) && python3 scripts/eval_base_policy_real.py'
