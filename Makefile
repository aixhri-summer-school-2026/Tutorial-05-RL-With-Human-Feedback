# MILE-on-multipanda dev verbs. One short command per action; all run inside the `sim`
# compose service with the verified headless env sourced. See docs/superpowers/specs/
# 2026-06-16-mile-docker-image-and-sim-demos-design.md.
DC      := docker compose -f docker/docker-compose.yml
ENVSH   := source scripts/in_container_env.sh
RUN      = $(DC) exec sim bash -lc '$(ENVSH) && $(1)'
RUND     = $(DC) exec -d sim bash -lc '$(ENVSH) && $(1)'
TS      := $(shell date -u +%Y%m%dT%H%M%S)

.PHONY: build up down shell sim-up collect-mediocre collect-expert base-policy mile

build:                       ## build the image
	$(DC) build

up:                          ## start the persistent sim service
	$(DC) up -d

down:                        ## stop the service
	$(DC) down

shell:                       ## interactive shell, env sourced, cd'd into the repo
	$(DC) exec sim bash -lc '$(ENVSH) && exec bash'

sim-up:                      ## launch the stacking sim headless (detached)
	$(call RUND,bash scripts/sim_up.sh)
	@echo "sim launching headless; give it ~10s, then check: make collect-mediocre"

collect-mediocre:            ## MEDIOCRE demos (feeds the base policy) -> sim_demos_mediocre.npz
	$(call RUN,python3 scripts/franka_sim_rollout_record.py \
	    --episodes 5 --mediocre true --require_success false \
	    --video_start_hold 0.5 --video_end_hold 0.5 \
	    --out_dir output_dir/franka/rollouts_mediocre_$(TS) \
	    --data output_dir/franka/sim_demos_$(TS).npz && \
	  cp output_dir/franka/sim_demos_$(TS).npz output_dir/franka/sim_demos_mediocre.npz)

collect-expert:              ## PERFECT (successful-only) demos -> sim_demos_expert.npz
	$(call RUN,python3 scripts/franka_sim_rollout_record.py \
	    --episodes 5 --mediocre false --require_success true \
	    --video_start_hold 0.5 --video_end_hold 0.5 \
	    --out_dir output_dir/franka/rollouts_expert_$(TS) \
	    --data output_dir/franka/sim_demos_$(TS).npz && \
	  cp output_dir/franka/sim_demos_$(TS).npz output_dir/franka/sim_demos_expert.npz)

base-policy:                 ## BC-train the (mediocre) base policy from mediocre demos
	$(call RUN,python3 scripts/build_base_policy.py \
	  --demos output_dir/franka/sim_demos_mediocre.npz \
	  --save_path trained_models/franka/base_policy)

mile:                        ## iterative MILE run against the live sim
	$(call RUN,cd scripts && python3 train_mile.py --config ../config_franka.json)
