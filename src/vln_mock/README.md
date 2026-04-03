# VLN (Vision-Language Navigation) System

This package implements a complete VLN system for autonomous navigation based on natural language instructions.

## System Architecture

The VLN system consists of the following key components:

### Core Nodes

1. **Instruction Processor (`ins_processor`)** - instruction parser that converts natural language into structured subtasks
2. **Vehicle Status Manager (`vehicle_status_manager`)** - State machine that manages task execution and system states
3. **Core Node (`core_node`)** - Main coordination node that handles value map computation, path planning, and task execution
4. **Controller (`controller`)** - Motion controller with multiple control interfaces
5. **ArUco Detector (`aruco_detector_node`)** - Detects ArUco markers and maintains historical detection records

### Supporting Nodes

- **Pointcloud to Grid (`pointcloud_to_grid_node`)** - Converts 3D point clouds to occupancy grids
- **Instruction Latch Bridge (`instruction_latch_bridge`)** - captures the first instruction and republishes it as a latched message
- **TF Broadcaster (`odom_tf_broadcaster`)** - republishes odometry as `map -> base_*` TF

## System Workflow

1. **Instruction Processing**: VLM processes natural language instructions into structured subtasks
2. **Task Management**: State machine receives subtasks and manages execution flow
3. **Value Map Computation**: Core node computes navigation value maps based on current subtask
4. **Path Planning**: Generate waypoints and navigation paths
5. **Motion Control**: Execute navigation using various control interfaces
6. **ArUco Detection**: Monitor for target markers during navigation
7. **Task Completion**: Handle task completion and transition to next subtask

## Key Features

- **Multi-modal Input**: Supports both language instructions and visual input
- **Flexible Control**: Multiple control interfaces (world coordinates, body coordinates, velocity control)
- **Robust Detection**: ArUco marker detection with historical tracking and deduplication
- **State Management**: Comprehensive state machine with error handling and emergency stop
- **Modular Design**: Loosely coupled nodes for easy maintenance and extension

## Usage

### Launch the Complete System

```bash
roslaunch vln_mock vln_system.launch
```

The default local VLM path now targets Qwen:

- `VLN_QWEN_MODEL_PATH` or `/var/tmp/vln_models/Qwen--Qwen2-VL-2B-Instruct`
- `VLN_GROUNDING_MODEL_PATH` or `/var/tmp/vln_models/grounding-dino-base`

### Launch the Simulation Scaffold

```bash
roslaunch vln_mock vln_sim.launch
```

### Launch the Operator GUI

```bash
roslaunch vln_mock vln_operator_gui.launch
```

The operator GUI is a ROS client node with:

- a chat-style instruction input area
- live task/state feedback from `/vln_status`
- parsed subtask display from `/subtasks`
- grounding and ArUco feedback panels
- embedded live camera preview from `/magv/camera/image_compressed/compressed`
- GroundingDINO bounding-box overlay drawn on top of the live preview
- topic freshness and low-noise system log views

This launch file can start Gazebo, spawn a lightweight `rover` or `uav` asset, publish static sensor TFs, bridge raw camera images into the existing compressed topic layout, and pass simulator topic overrides into the VLN stack from one entry point.

Choose the built-in robot profile explicitly:

```bash
roslaunch vln_mock vln_sim.launch robot_profile:=rover
roslaunch vln_mock vln_sim.launch robot_profile:=uav
```

Launch the indoor evaluation scene together with the provided RViz profile:

```bash
roslaunch vln_mock vln_sim.launch \
  world_file:=$(pwd)/gazebo_world/worlds/indoor_vln.world \
  robot_profile:=rover \
  launch_rviz:=true
```

### Rebuild the Workspace Safely

```bash
bash tools/rebuild_workspace.sh
```

### Run the Smoke Test

```bash
bash tools/smoke_test.sh
```

To validate the simulation launch wiring without requiring Gazebo itself:

```bash
LAUNCH_FILE=vln_sim.launch LAUNCH_ARGS="launch_world:=false spawn_robot:=false" bash tools/smoke_test.sh
```

### Run the Headless Indoor Evaluation

```bash
source /opt/ros/noetic/setup.bash
source devel_local/setup.bash
python3 evaluate/run_indoor_headless_eval.py
```

### Download Local Models

```bash
python3 tools/download_models.py --endpoint https://hf-mirror.com
```

For an 8 GB class GPU, `Qwen/Qwen2-VL-2B-Instruct` is the recommended first local Qwen target.
If repository disk space is tight, download into an absolute path outside the repo and pass that path through launch args.

### Send Navigation Instructions

```bash
# Example instruction
rostopic pub /instruction std_msgs/String "data: 'move forward to the tree, turn right, go straight and stop at the traffic cone'"
```

### Validate Python Scripts

```bash
python3 -m py_compile src/vln_mock/scripts/*.py src/llm_model/scripts/*.py src/aruco_detector/scripts/*.py
```

### Monitor System Status

```bash
# Monitor VLN status
rostopic echo /vln_status

# Monitor final completion status
rostopic echo /status
```

## Configuration Parameters

### Core Node Parameters
- `grid_resolution`: Grid cell size in meters (default: 0.1)
- `path_planning_distance`: Path planning distance in meters (default: 2.0)
- `goal_tolerance`: Goal reaching tolerance in meters (default: 0.5)

### Controller Parameters
- `position_kp`: Position control proportional gain (default: 1.0)
- `max_linear_vel`: Maximum linear velocity (default: 1.0 m/s)
- `max_angular_vel`: Maximum angular velocity (default: 1.0 rad/s)

### ArUco Detector Parameters
- `marker_size`: Physical size of ArUco markers in meters (default: 0.1)
- `detection_threshold`: Distance threshold for duplicate detection (default: 0.3)

## Topics

### Input Topics
- `/instruction` (std_msgs/String): Natural language navigation instructions
- `/magv/camera/image_compressed/compressed` (sensor_msgs/CompressedImage): Camera images
- `/magv/scan/3d` (sensor_msgs/PointCloud2): 3D point cloud data
- `/magv/odometry/gt` (nav_msgs/Odometry): Vehicle odometry

### Output Topics
- `/vln_status` (magv_vln_msgs/VehicleStatus): System status and current task
- `/status` (std_msgs/Int32): Final task completion status
- `/aruco_info` (aruco_detector/ArucoInfo): Detected ArUco markers
- `/value_map` (magv_vln_msgs/ValueMap): Navigation value map
- `/magv/omni_drive_controller/cmd_vel` (geometry_msgs/Twist): Velocity commands

## Troubleshooting

### Common Issues

1. **VLM Model Loading**: Ensure the configured local model directory exists in `llm_model/models/`
2. **Topic Connections**: Verify all nodes are publishing/subscribing to correct topics
3. **Coordinate Frames**: Check that coordinate transformations are properly configured
4. **Parameter Tuning**: Adjust control parameters based on your specific robot platform
5. **Disk Space**: Large local model downloads can exceed the free space of the repository partition, so use absolute model paths on a larger disk when needed
6. **Simulator Topic Layout**: If your world publishes different topic names, override `image_topic`, `pointcloud_topic`, `odometry_topic`, and `cmd_vel_topic` in `vln_system.launch` or `vln_sim.launch` instead of editing node code
7. **Gazebo Robot Profile**: The built-in `uav` profile is intentionally simplified to planar hover motion so the existing VLN stack can be exercised before a full PX4 or multirotor dynamics integration
8. **Photorealism Expectations**: `indoor_vln.world` is still a Gazebo Classic engineering scene, not a film-grade renderer. If you ultimately want Unreal-like photoreal rendering, AirSim/Unreal or Isaac Sim/Omniverse is the more suitable long-term direction

For a full local deployment and model bring-up flow, see `LOCAL_BRINGUP.md` in the repository root.

### Debug Commands

```bash
# Check node status
rosnode list
rosnode info /core_node

# Monitor topic data
rostopic list
rostopic echo /vln_status

# View system logs
rosrun rqt_console rqt_console
```

## Dependencies

- ROS Noetic
- OpenCV (for ArUco detection)
- PCL (for point cloud processing)
- Transformers (for VLM model)
- PyTorch (for VLM inference)

## Future Enhancements

- Integration with semantic segmentation for better object detection
- Advanced path planning algorithms (A*, RRT)
- Multi-robot coordination
- Real-time SLAM integration
- Enhanced VLM capabilities with larger models
