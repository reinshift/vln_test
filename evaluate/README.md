# Indoor Headless Evaluation

This directory contains a repeatable headless evaluation pipeline for the indoor Gazebo VLN world.

Tracked files:

- `run_indoor_headless_eval.py`: main evaluation runner
- `tasks_indoor_vln.json`: default indoor task set

Ignored outputs:

- `runs/`: per-run logs, ROS logs, topic traces, and generated summaries

Typical usage:

```bash
source /opt/ros/noetic/setup.bash
source devel_local/setup.bash
python3 evaluate/run_indoor_headless_eval.py
```

For quicker regression loops:

```bash
python3 evaluate/run_indoor_headless_eval.py --max-episodes 2
```

The runner will:

- start `vln_sim.launch` headlessly against `gazebo_world/worlds/indoor_vln.world`
- spawn the built-in `rover`
- wait for Gazebo, odometry, point cloud, and VLN status topics
- optionally wait for `Qwen` to report ready on `/VLM_Status`
- reset the robot pose before every episode via `/gazebo/set_model_state`
- publish each natural-language instruction on `/instruction`
- record detailed topic events and `/rosout_agg`
- collect ROS node logs under the run directory via `ROS_LOG_DIR`
- generate `summary.json` and `summary.md` with automatic findings

The runner now also writes a partial summary when interrupted with `Ctrl-C`, so short exploratory runs still leave usable artifacts behind.

Useful outputs for a single run:

- `runs/<timestamp>/launch_console.log`
- `runs/<timestamp>/ros_logs/`
- `runs/<timestamp>/topic_events.jsonl`
- `runs/<timestamp>/rosout.jsonl`
- `runs/<timestamp>/summary.json`
- `runs/<timestamp>/summary.md`
