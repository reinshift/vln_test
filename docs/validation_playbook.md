# Validation Playbook

## Fast Path

```bash
make bootstrap
make assets-sync
make rebuild
make test-unit
make test-contract
make test-integration
make test-sim-smoke
make test-e2e
```

Or run the whole stack:

```bash
make validate
```

If you already have the curated Fuel cache locally and want to avoid network access:

```bash
make assets-sync-offline
```

## What Each Stage Means

- `test-unit`
  - Pure domain logic in `vln_core`.
- `test-contract`
  - Message names, launch contracts, manifests, task schemas.
- `test-integration`
  - Instruction -> mission -> trajectory -> command -> reset flow.
- `test-sim-smoke`
  - Bringup graph and real PX4 SITL launch path.
- `test-e2e`
  - Hard-relaunch PX4 episode regression, success-rate gating, and summary generation.

## PX4 Lane

The default smoke and e2e paths require a PX4/Gazebo Classic workstation.
This repo already ships a repo-local `.env` and auto-detects `sim/cache/PX4-Autopilot`.
For an interactive shell inside the project, use:

```bash
source tools/enter_px4_env.sh
```

Or export manually:

```bash
export PX4_AUTOPILOT_DIR=/abs/path/to/PX4-Autopilot
```

You can verify the environment at any time with:

```bash
make px4-env-check
```

If you are using the repo-local `venv310`, install the PX4 Python build chain once:

```bash
make px4-prepare
```

Then rerun the default commands:

```bash
make test-sim-smoke
make test-e2e
```

The default real PX4 model now uses an RGB camera plus `rplidar`.
Depth images are no longer part of the required startup path for the Noetic regression lane.
`make assets-sync` now downloads the curated outdoor Fuel asset set into `sim/cache/assets/fuel/`
and stages the generated Gazebo Classic worlds under `sim/worlds/generated/`.

## Output Artifacts

- `tests/reports/contract/`
- `tests/reports/integration/`
- `tests/reports/system/latest/summary.json`
- `tests/reports/system/latest/summary.md`

Each episode entry in `summary.json` includes:

- `goal_error_m`
- `goal_tolerance_m`
- `path_length_m`
- `waypoint_count`
- `step_count`
- `timeout_sec`

The default local `test-e2e` gate requires a `1.0` success rate on the checked-in golden tasks.
Each golden task must also declare `world_name` so the PX4 episode runner can select the generated Gazebo Classic world.
