SHELL := /bin/bash

.PHONY: bootstrap px4-env-check px4-prepare assets-sync assets-sync-offline rebuild test-unit test-contract test-integration test-sim-smoke test-e2e validate

bootstrap:
	bash tools/bootstrap.sh

px4-env-check:
	source /opt/ros/noetic/setup.bash && source tools/load_local_env.sh && python3 tools/check_px4_env.py --px4-autopilot-dir "$$PX4_AUTOPILOT_DIR"

px4-prepare:
	bash tools/install_px4_python_deps.sh

assets-sync:
	python3 tools/sync_assets.py

assets-sync-offline:
	python3 tools/sync_assets.py --offline

rebuild:
	bash tools/rebuild_workspace.sh

test-unit:
	bash tools/run_unit_tests.sh

test-contract:
	bash tools/run_contract_tests.sh

test-integration:
	bash tools/run_integration_tests.sh

test-sim-smoke:
	bash tools/run_sim_smoke.sh

test-e2e:
	bash tools/run_e2e_tests.sh

validate: bootstrap assets-sync rebuild test-unit test-contract test-integration test-sim-smoke test-e2e
