# Project Engineering Blueprint 2026

## Target

- Rebuild the repository around a UAV-first ROS Noetic architecture.
- Keep runtime nodes thin and move decision logic into `vln_core`.
- Treat the old implementation as Git history only; the live tree follows the new package graph.
- Make the default local validation path deterministic and runnable without GPU models.
- Make the default execution and validation lane use real PX4 SITL + Gazebo Classic instead of the mock UAV bridge.

## Package Layout

- `src/vln_msgs`
  - ROS message and service contracts only.
- `src/vln_core`
  - Pure Python domain logic: mission compilation, planning, safety, semantic memory, evaluation.
- `src/vln_runtime`
  - Runtime nodes that wire ROS topics/services onto `vln_core`.
- `src/vln_perception`
  - Pluggable instruction and landmark perception adapters with mock-safe defaults.
- `src/vln_sim`
  - UAV sim bridge, reset service, world profile publishing, PX4-facing adapters.
- `src/vln_bringup`
  - Development, simulation, and evaluation launch entrypoints.
- `src/vln_eval`
  - Task schema validation, offline regression runner, report generation.

## Repository Layout

- `sim/`
  - Asset locks, world templates, generated worlds, and license ledger.
- `tests/`
  - `unit/`, `contract/`, `integration/`, `system/`, `fixtures/`, `goldens/`.
- `tools/`
  - Bootstrap, asset sync, workspace rebuild, validation entrypoints.
- `docker/`
  - Noetic + Gazebo Classic + PX4 container recipe.
- `ci/`
  - CI notes and reusable configuration inputs.

## Principles

- No package outside `vln_msgs` defines public wire contracts.
- No module in `vln_core` imports `rospy`.
- Simulation, evaluation, and perception must all degrade to a deterministic mock path.
- Validation is not complete unless the golden regression report is emitted.
- Default smoke and e2e commands are fail-fast when PX4 prerequisites are missing.

## Runtime Maturity Baseline

- Mission lifecycle is explicit: compile, active tracking, waypoint progress, completion, and reset.
- Planner behavior is event-driven and stable: it republishes on mission/semantic changes instead of uncontrolled loops.
- Controller behavior is continuous: it tracks active waypoints over time and emits progress feedback separate from general runtime state.
- Safety behavior includes hysteresis so transient scan noise does not thrash command gating.
- Reset behavior is structured end-to-end through `/sim/uav/reset_event` JSON payloads.
- Evaluation artifacts must be reproducible and include path/goal metrics, not only pass-fail counts.
- Real PX4 episode regression uses hard relaunch per episode to avoid simulator state leakage.
