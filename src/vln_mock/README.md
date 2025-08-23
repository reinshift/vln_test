# VLN (Vision-Language Navigation) System

This package implements a complete VLN system for autonomous navigation based on natural language instructions.

## System Architecture

The VLN system consists of the following key components:

### Core Nodes

1. **Instruction Processor (`instruction_processor`)** - VLM node that processes natural language instructions into structured subtasks
2. **Vehicle Status Manager (`vehicle_status_manager`)** - State machine that manages task execution and system states
3. **Core Node (`core_node`)** - Main coordination node that handles value map computation, path planning, and task execution
4. **Controller (`controller`)** - Motion controller with multiple control interfaces
5. **ArUco Detector (`aruco_detector_node`)** - Detects ArUco markers and maintains historical detection records

### Supporting Nodes

- **Pointcloud to Grid (`pointcloud_to_grid_node`)** - Converts 3D point clouds to occupancy grids
- **Test Node (`test_vln_system`)** - System testing and validation

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

### Send Navigation Instructions

```bash
# Example instruction
rostopic pub /instruction std_msgs/String "data: 'move forward to the tree, turn right, go straight and stop at the traffic cone'"
```

### Run System Tests

```bash
rosrun vln_mock test_vln_system.py
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
- `/magv/camera/image_compressed` (sensor_msgs/CompressedImage): Camera images
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

1. **VLM Model Loading**: Ensure the Qwen-VL model is properly installed in `llm_model/models/`
2. **Topic Connections**: Verify all nodes are publishing/subscribing to correct topics
3. **Coordinate Frames**: Check that coordinate transformations are properly configured
4. **Parameter Tuning**: Adjust control parameters based on your specific robot platform

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
