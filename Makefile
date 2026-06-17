# MILE-on-multipanda dev verbs. One short command per action; all run inside the `sim`
# compose service with the verified headless env sourced. See docs/superpowers/specs/
# 2026-06-16-mile-docker-image-and-sim-demos-design.md.
DC      := docker compose -f docker/docker-compose.yml
ENVSH   := source scripts/in_container_env.sh
RUN      = $(DC) exec sim bash -lc '$(ENVSH) && $(1)'
RUND     = $(DC) exec -d sim bash -lc '$(ENVSH) && $(1)'

.PHONY: build up down shell sim-up collect base-policy mile

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
	@echo "sim launching headless; give it ~10s, then check: make collect"

collect:                     ## scripted sim rollouts -> MP4 + demo .npz
	$(call RUN,python scripts/franka_sim_rollout_record.py --episodes 3 \
	  --out output_dir/franka/rollout.mp4 --data output_dir/franka/sim_demos.npz)

base-policy:                 ## BC-train the base policy from the collected sim demos
	$(call RUN,python scripts/build_base_policy.py --demos output_dir/franka/sim_demos.npz \
	  --save_path trained_models/franka/base_policy)

mile:                        ## iterative MILE run against the live sim
	$(call RUN,cd scripts && python train_mile.py --config ../config_franka.json)
