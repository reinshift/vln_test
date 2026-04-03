# VLN Local Bring-Up Guide

This guide is the shortest stable path to rebuild the workspace, deploy local models, and bring the stack up against a local simulation world.

## 1. Current Deployment Baseline

The repository is now configured around local model paths and YAML-backed ROS params instead of hardcoded `/catkin_ws/...` paths.

The default launch profile now prefers:

- Qwen VLM: `/var/tmp/vln_models/Qwen--Qwen2-VL-2B-Instruct`
- GroundingDINO: `/var/tmp/vln_models/grounding-dino-base`

You can still override them with:

```bash
export VLN_QWEN_MODEL_PATH=/abs/path/to/Qwen--Qwen2-VL-2B-Instruct
export VLN_GROUNDING_MODEL_PATH=/abs/path/to/grounding-dino-base
```

Default launch entry:

```bash
roslaunch vln_mock vln_system.launch
```

Simulation scaffold entry:

```bash
roslaunch vln_mock vln_sim.launch
```

Key config directory:

```text
src/vln_mock/config/
```

Key launch file:

```text
src/vln_mock/launch/vln_system.launch
```

## 2. Supported Local Model Layout

### VLM

Default VLM model path:

```text
src/llm_model/models/microsoft--Florence-2-large
```

This path is already valid in the current workspace and has been verified to load locally with `transformers`.

### Grounding

Preferred GroundingDINO model path:

```text
src/llm_model/models/GroundingDino/grounding-dino-base
```

If that directory is absent, the stack now falls back to Florence-2 phrase grounding automatically when `grounding_backend:=auto`.

### Optional Qwen VLM

For an 8 GB class GPU, prefer `Qwen/Qwen2-VL-2B-Instruct` first. It is a more realistic local bring-up target than 7B on this machine class while still matching the current loader code path.

Recommended local path:

```text
src/llm_model/models/Qwen--Qwen2-VL-2B-Instruct
```

If you want full instruction-plus-vision generation with Qwen instead of the current Florence/offline fallback path, place the local Qwen model in any accessible directory and override launch args:

```bash
roslaunch vln_mock vln_system.launch \
  vlm_backend:=qwen \
  vlm_model_path:=/abs/path/to/Qwen2-VL-2B-Instruct \
  grounding_backend:=auto
```

### Model download helper

The repository now includes a Hugging Face compatible download helper:

```bash
python3 tools/download_models.py \
  --endpoint https://hf-mirror.com \
  --qwen-dir /abs/path/to/Qwen--Qwen2-VL-2B-Instruct \
  --grounding-dir /abs/path/to/grounding-dino-base
```

When the Hugging Face API path is unstable, the helper can fall back to direct file downloads, and Qwen can also fall back to ModelScope:

```bash
python3 tools/download_models.py \
  --endpoint https://hf-mirror.com \
  --qwen-provider modelscope \
  --grounding-provider hf_manifest \
  --qwen-dir /abs/path/to/Qwen--Qwen2-VL-2B-Instruct \
  --grounding-dir /abs/path/to/grounding-dino-base
```

If disk space in the repository is tight, point `--qwen-dir` and `--grounding-dir` to a larger absolute path and pass those paths back into `roslaunch`. On the current machine, this is especially important because `/home` is much tighter than `/`.

Example:

```bash
python3 tools/download_models.py \
  --endpoint https://hf-mirror.com \
  --qwen-dir /var/tmp/vln_models/Qwen--Qwen2-VL-2B-Instruct \
  --grounding-dir /var/tmp/vln_models/grounding-dino-base
```

## 3. Fresh Workspace Build

Use the provided script instead of reusing an old cached build directory:

```bash
bash tools/rebuild_workspace.sh
```

Defaults:

- build dir: `build_local`
- devel dir: `devel_local`
- install dir: `install_local`
- compiler: `/usr/bin/gcc` and `/usr/bin/g++`

Override example:

```bash
BUILD_DIR=build_debug DEVEL_DIR=devel_debug JOBS=4 bash tools/rebuild_workspace.sh
```

## 4. Smoke Test

After building:

```bash
bash tools/smoke_test.sh
```

This validates:

- Python node syntax for the edited scripts
- `roslaunch --nodes vln_mock vln_system.launch`

Optional model validation after local downloads:

```bash
python3 tools/model_smoke_test.py \
  --qwen-path /abs/path/to/Qwen--Qwen2-VL-2B-Instruct \
  --grounding-path /abs/path/to/grounding-dino-base
```

## 5. Launch Modes

### Default local bring-up

```bash
source devel_local/setup.bash
roslaunch vln_mock vln_system.launch
```

### Operator GUI client

```bash
source devel_local/setup.bash
roslaunch vln_mock vln_operator_gui.launch
```

This GUI client publishes `/instruction` and subscribes to the main runtime feedback topics so you can drive the system like a lightweight dialogue console during Gazebo or real-robot tests.
It also embeds a live preview of the compressed camera stream by default.

### Simulation clock enabled

```bash
source devel_local/setup.bash
roslaunch vln_mock vln_system.launch use_sim_time:=true
```

Model loading now uses wall time, so the VLM and grounding backends can still initialize before `/clock` starts. Once you begin sending navigation instructions, your simulator still needs to publish `/clock` normally.

### Gazebo scaffold bring-up

```bash
source devel_local/setup.bash
roslaunch vln_mock vln_sim.launch
```

Useful overrides:

```bash
roslaunch vln_mock vln_sim.launch \
  world_file:=$(pwd)/gazebo_world/worlds/indoor_vln.world \
  robot_profile:=rover \
  launch_rviz:=true \
  image_topic:=/your/camera/topic \
  pointcloud_topic:=/your/lidar/topic \
  odometry_topic:=/your/odom/topic \
  cmd_vel_topic:=/your/cmd_vel/topic
```

Indoor debug world:

```bash
roslaunch vln_mock vln_sim.launch \
  world_file:=$(pwd)/gazebo_world/worlds/indoor_vln.world \
  robot_profile:=rover \
  launch_rviz:=true \
  launch_operator_gui:=true
```

Built-in robot profiles:

- `robot_profile:=rover`: ground robot mock with RGB camera, 3D point cloud sensor on `/magv/scan/3d`, and planar laser scan on `/magv/scan/2d`
- `robot_profile:=uav`: hovering UAV mock with the same `/magv/...` interface contract so you can switch platforms without retuning the VLN topic graph

Reference choices behind these two profiles:

- Ground robot reference: `TurtleBot3 Waffle Pi`
- UAV reference: `PX4 iris_depth_camera`

These references were used to choose a sensible “small rover + small drone” direction for Gazebo/ROS bring-up. The in-repo models are intentionally lighter than the full upstream stacks so the VLN pipeline can be exercised without first integrating PX4, MAVROS, or the full TurtleBot simulation tree.

If you want to reuse only the launch scaffold and bring your own external robot model, disable the built-in spawn path:

```bash
source devel_local/setup.bash
roslaunch vln_mock vln_sim.launch spawn_robot:=false enable_odom_tf_broadcaster:=true
```

### Explicit Florence-only bring-up

```bash
source devel_local/setup.bash
roslaunch vln_mock vln_system.launch \
  vlm_backend:=florence2 \
  grounding_backend:=auto
```

### Explicit GroundingDINO bring-up

```bash
source devel_local/setup.bash
roslaunch vln_mock vln_system.launch \
  grounding_backend:=grounding_dino \
  grounding_model_path:=/abs/path/to/grounding-dino-base
```

## 6. Local World Integration Checklist

When you connect a local Gazebo or other simulator, make sure the simulator publishes at least the following topics:

- `/magv/odometry/gt` as `nav_msgs/Odometry`
- `/magv/scan/3d` as `sensor_msgs/PointCloud2`
- `/magv/camera/image_compressed/compressed` as `sensor_msgs/CompressedImage`

Recommended frame chain:

- `map`
- `base_footprint`
- sensor frames hanging below `base_footprint`

If your simulator publishes the compressed camera feed on a different topic, update these three config files together so the visual stack stays aligned:

- `src/vln_mock/config/aruco_detector.yaml`
- `src/vln_mock/config/grounding_dino.yaml`
- `src/vln_mock/config/vlm_loader.yaml`

Important notes from the fixes already applied:

- `pointcloud_to_grid_node` now refuses to publish mislabeled map data when TF is missing and `require_transform=true`
- `vehicle_status_manager` now treats TF loss as a real safety condition instead of silently evaluating the point cloud in the wrong frame
- ArUco output is published in `base_footprint` by default, and camera pitch conversion has been corrected

## 7. Simulation World Scaffold

There is now a placeholder world scaffold here:

```text
gazebo_world/worlds/debug_empty.world
```

And a simple indoor obstacle scene here:

```text
gazebo_world/worlds/indoor_vln.world
```

And a more realistic outdoor road-and-terrain scene here:

```text
gazebo_world/worlds/indoor_vln.world
```

The simulation launch now also ships with a ready-to-use RViz profile:

```text
src/vln_mock/config/vln_sim_debug.rviz
```

Use it as the seed file when wiring your local Gazebo world. The repository now includes lightweight built-in rover/UAV mock assets so you can exercise the stack even before integrating a full external robot package.

`gazebo_world/worlds/indoor_vln.world` now also contains a first batch of explicit task targets:

- bench
- tree
- cone gate
- ArUco-style target board
- start pad and finish zone

That means you can immediately do language-grounded dry runs such as:

```bash
go through the cone gate and stop near the bench
```

```bash
move to the tree and head to the finish zone
```

## 8. Config Files Worth Tuning First

For ground robot bring-up:

- `src/vln_mock/config/vehicle_status_manager.yaml`
- `src/vln_mock/config/controller.yaml`
- `src/vln_mock/config/pointcloud_to_grid.yaml`
- `src/vln_mock/config/aruco_detector.yaml`

For model behavior:

- `src/vln_mock/config/vlm_loader.yaml`
- `src/vln_mock/config/grounding_dino.yaml`
- `src/vln_mock/config/ins_processor.yaml`

For planning behavior:

- `src/vln_mock/config/core_node.yaml`

## 9. Known Remaining Gaps

The stack is now much safer to run locally, but it is still fundamentally closer to a 2D ground robot VLN pipeline than a full UAV-ready stack.

The main follow-up items for your next development round are:

- replace raw odometry with a proper simulated TF tree when the world is ready
- add a real GroundingDINO local checkpoint if phrase grounding accuracy becomes a bottleneck
- split ground-vehicle and UAV controller assumptions, especially altitude and 3D obstacle handling

## 10. Cleanup

After your final validation pass, remove local catkin byproducts so the repository stays clean:

```bash
bash tools/cleanup_artifacts.sh
```
