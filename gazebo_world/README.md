# Gazebo World Scaffold

这个目录用于放置本地 Gazebo/world 联调资源。

当前补充内容:
- `worlds/debug_empty.world`: 最小可启动 world 骨架
- `worlds/indoor_vln.world`: 带基础障碍物、长椅、树、锥桶门、真实 ArUco 纹理板、起点与终点区域的简易室内任务场景
- `models/vln_rover/`: 轻量地面小车资产
- `models/vln_uav/`: 轻量悬停无人机资产
- `models/vln_aruco_board/`: 使用仓库内 `Aruco.png` 作为纹理的目标板
- `launch/`: 预留仿真启动脚本目录

推荐统一入口:

- `roslaunch vln_mock vln_sim.launch`

这个 launch 会做三件事:

- 可选启动 Gazebo world
- 生成 `vln_rover` 或 `vln_uav`
- 发布 base -> camera / lidar 的静态 TF 骨架
- 将 image / pointcloud / odometry / cmd_vel 这些仿真 topic 透传到 VLN 主系统

当前 `indoor_vln.world` 里已经补上的任务目标物:

- `entry_pad`: 推荐起点区域
- `bench_target`: 长椅类目标
- `tree_target`: 树类目标
- `cone_gate_left/right`: 锥桶门
- `vln_aruco_board`: 使用 `Aruco.png` 的纹理目标板
- `finish_zone`: 推荐终点区域

当前场景渲染提升内容:

- 地板改为 Gazebo 自带 `WoodFloor` 贴图
- 墙面改为 `PaintedWall`
- 障碍物改为 `Bricks` / `WoodPallet`
- 长椅改为木质 + 金属腿材质
- 树干/树冠改为 `Trunk` / `Grass`
- 场景已开启天空云层、阴影和轻雾
- 新增 `.rviz` 配置：`src/vln_mock/config/vln_sim_debug.rviz`

建议后续继续补齐:
- 机器人模型与 `robot_state_publisher`
- 相机/雷达静态 TF
- 更丰富的 Fuel / mesh 资产替换当前 box primitive 建筑和障碍物
- 更复杂的走廊/房间连通结构
- 统一 topic 映射到:
  - `/magv/odometry/gt`
  - `/magv/scan/3d`
  - `/magv/scan/2d`
  - `/magv/camera/image_compressed/compressed`
  - `/magv/omni_drive_controller/cmd_vel`

当前内置机器人说明:

- `vln_rover`: 参考小型差速/全向地面平台布局，内置 RGB 相机、前向 3D 点云传感器、平面激光雷达话题。
- `vln_uav`: 参考四旋翼任务载荷布局，使用简化的平面运动控制保持话题兼容，适合先把 VLN 感知与任务链条跑通。

参考平台选择:

- 地面机器人参考 `TurtleBot3 Waffle Pi`
- 无人机参考 `PX4 iris_depth_camera`

注意:

- 这里内置的是轻量 mock 资产，不是把 TurtleBot3 或 PX4 大仓库整包拉进来。
- `vln_uav` 当前是“悬停式平面任务机”近似模型，优先保证感知、话题和任务流可跑，不等价于完整飞控动力学。
- 内置模型的 ROS 传感器话题默认固定到 `/magv/...`。如果你后面要改成别的话题名，更推荐 `spawn_robot:=false` 后接入你自己的外部机器人包。

常用启动方式:

```bash
roslaunch vln_mock vln_sim.launch \
  world_file:=$(pwd)/gazebo_world/worlds/indoor_vln.world \
  robot_profile:=rover
```

```bash
roslaunch vln_mock vln_sim.launch \
  world_file:=$(pwd)/gazebo_world/worlds/indoor_vln.world \
  robot_profile:=uav
```

```bash
roslaunch vln_mock vln_sim.launch \
  world_file:=$(pwd)/gazebo_world/worlds/indoor_vln.world \
  robot_profile:=rover \
  launch_rviz:=true
```

真实感边界说明:

- 当前这套是 `Gazebo Classic + SDF + 内置材质` 的增强版室内 world，适合做 VLN 算法联调、传感器话题联调和演示。
- 如果目标是接近 Unreal Engine 的写实感，建议后续考虑迁移到 `AirSim/Unreal` 或 `Isaac Sim/Omniverse`，Gazebo 更适合中等真实感和工程调试。

建议的第一批回归指令:

```bash
go through the two cones and stop near the bench
```

```bash
move to the tree and then head to the yellow finish area
```

```bash
find the marker board on the wall and stop in front of it
```
