# Architecture Overview

## Runtime Flow

1. `instruction_gateway_node` writes raw language to `/vln/mission/raw_instruction`.
2. `mission_manager_node` compiles the mission, tracks lifecycle, and emits `/vln/mission/compiled`.
3. `landmark_grounding_node` and `aruco_node` publish observations into the semantic map.
4. `semantic_map_node` maintains `/vln/world/semantic_map` and republishes stable snapshots.
5. `planner_node` emits `/vln/planning/trajectory` whenever the mission or semantic world changes.
6. `controller_node` continuously tracks the active trajectory, publishes `/vln/control/command`, and reports progress on `/vln/runtime/controller_feedback`.
7. `safety_supervisor_node` gates the command with corridor hysteresis and forwards to `/sim/uav/cmd_vel`.
8. `episode_reset_service_node` publishes a structured `/sim/uav/reset_event` payload that runtime and sim nodes consume together.
9. `px4_uav_bridge_node` is the default `/sim/uav/*` provider and bridges MAVROS/PX4 state onto the VLN runtime contract.
10. `mock_uav_bridge_node` remains available only as an explicit non-default debug backend.

## Design Boundaries

- `vln_core` owns state evolution and deterministic algorithms.
- ROS nodes own translation, timers, and publishers/subscribers.
- `vln_eval` owns task files, metrics, and artifact generation.
- `vln_sim` owns sim-facing state, PX4/MAVROS bridging, reset behavior, and world contracts.

## Execution Contracts

- Mission progress is reported through `RuntimeState` on `/vln/runtime/state`.
- Controller-only progress events are mirrored on `/vln/runtime/controller_feedback`.
- Reset payloads are JSON objects with `episode_name`, `hard_reset`, and optional `spawn`.
- Default launch and validation flow target PX4 SITL + Gazebo Classic instead of the mock lane.
- The deterministic evaluation lane computes `goal_error_m`, `path_length_m`, `waypoint_count`, and `step_count` for each episode.
