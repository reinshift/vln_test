# VLN 项目研究报告

生成时间: 2026-04-03

分析范围: `/home/cq/vln_test-fix_bug`

分析方式:
- 静态代码审阅
- ROS 包/launch/topic/消息关系梳理
- Python 脚本语法检查
- catkin 轻量重建验证

结论先行:
- 这套代码目前更准确的定位是: 一个面向二维地面平台的 VLN 原型系统，而不是已经完整打通的小车/无人机通用方案。
- 仓库具备“自然语言 -> 子任务 -> 感知 -> 简单 value map -> 单点导航 -> ArUco 收尾”的基本闭环雏形，但距离稳定本地仿真、世界模型接入、长期迭代还有明显工程缺口。
- 当前最关键的问题不在“模型不够大”，而在“坐标系链路不完整、占据图语义不正确、运行环境硬编码、模型资产漂移、路径规划仍停留在 argmax 选点级别”。
- 如果后续要在本地 world 仿真环境里做稳定联调，建议先完成本报告中列出的 P0 修复项，再进入 Gazebo/world 级测试。

---

## 1. 项目整体判断

### 1.1 项目目标

从仓库组织和代码职责看，这个项目想实现的是:

1. 接收自然语言导航指令。
2. 使用 VLM/LLM 将指令拆成结构化子任务。
3. 使用 GroundingDINO 和 ArUco 做目标感知。
4. 使用点云生成局部网格。
5. 用一个 value map 选择导航目标。
6. 通过控制器把目标转成 `cmd_vel`。

### 1.2 当前真实能力边界

当前实现具备以下能力:

- 以 ROS Noetic + catkin 多包方式组织代码。
- 有完整消息定义，便于后续扩展。
- 有启动总入口 `src/vln_mock/launch/vln_system.launch`。
- 有状态机节点、核心决策节点、控制节点、点云节点、VLM 节点、ArUco 节点。
- 有 360 度扫描 + DINO 检测 + 简单 value map 融合思路。

当前实现不具备或尚未完整具备:

- 真正可用的栅格占据建图/局部规划。
- 严格一致的 TF/坐标系链路。
- 无人机三维导航能力。
- 自带的仿真 world、模型、测试脚本、CI 验证链。
- 可移植的本地运行环境。

### 1.3 一句话定性

这不是一个“已经可以稳定跑 world 仿真”的项目，而是一个“节点骨架基本齐全、核心实验逻辑已经写出，但还需要做坐标体系收口和工程化收口”的研究型原型。

---

## 2. 仓库架构树

### 2.1 工作区级目录树

说明:
- 下列树保留了源码和关键工作区目录。
- `build/`、`devel/`、`venv310/` 为构建或环境产物，不在树中展开。

```text
/home/cq/vln_test-fix_bug
├── .catkin_workspace
├── .vscode/
│   └── settings.json
├── Aruco.png
├── gazebo_world/
├── params.txt
├── requirements.txt
├── VLN_Project_Research_Report.md
└── src/
    ├── CMakeLists.txt
    ├── aruco_detector/
    │   ├── CMakeLists.txt
    │   ├── package.xml
    │   └── scripts/
    │       └── aruco_node.py
    ├── llm_model/
    │   ├── CMakeLists.txt
    │   ├── package.xml
    │   ├── models/
    │   │   └── microsoft--Florence-2-large/
    │   ├── scripts/
    │   │   ├── grounding_dino_node.py
    │   │   ├── ins_processor.py
    │   │   └── vlm_loader.py
    │   └── src/
    ├── magv_vln_msgs/
    │   ├── CMakeLists.txt
    │   ├── package.xml
    │   └── msg/
    │       ├── ArucoInfo.msg
    │       ├── ArucoMarker.msg
    │       ├── BoundingBox2D.msg
    │       ├── DetectedObject.msg
    │       ├── DetectedObjectArray.msg
    │       ├── Detection2D.msg
    │       ├── Detection2DArray.msg
    │       ├── PathPoint.msg
    │       ├── PositionCommand.msg
    │       ├── SubTask.msg
    │       ├── ValueMap.msg
    │       └── VehicleStatus.msg
    ├── pointcloud_to_grid/
    │   ├── .gitignore
    │   ├── CMakeLists.txt
    │   ├── cfg/
    │   │   └── MyParams.cfg
    │   ├── include/
    │   │   └── pointcloud_to_grid/
    │   │       └── pointcloud_to_grid_core.hpp
    │   ├── package.xml
    │   └── src/
    │       └── pointcloud_to_grid_node.cpp
    └── vln_mock/
        ├── CMakeLists.txt
        ├── README.md
        ├── launch/
        │   └── vln_system.launch
        ├── package.xml
        ├── scripts/
        │   ├── controller.py
        │   ├── core_node.py
        │   ├── instruction_latch_bridge.py
        │   ├── odom_tf_broadcaster.py
        │   └── vehicle_status_manager.py
        └── src/
```

### 2.2 目录树解读

- `magv_vln_msgs/` 是协议层，定义了整个系统内部的数据契约。
- `pointcloud_to_grid/` 是感知底座，但目前更像“点云投影器”，还不是严格意义上的占据栅格构建器。
- `aruco_detector/` 负责 ArUco 目标识别与相机坐标到机体坐标的转换。
- `llm_model/` 负责指令解析与多模态推理。
- `vln_mock/` 是系统调度层，状态机、控制、核心价值图规划都在这里。
- `gazebo_world/` 当前是空目录，说明“world 仿真”还没有真正接进工程。

---

## 3. 包级职责与分层

### 3.1 包职责表

| 包名 | 角色 | 核心文件 | 作用 |
| --- | --- | --- | --- |
| `magv_vln_msgs` | 消息协议层 | `msg/*.msg` | 定义状态、路径点、检测框、ArUco、value map 等消息 |
| `pointcloud_to_grid` | 点云栅格化层 | `src/pointcloud_to_grid_node.cpp` | 将点云投影成网格并发布 `/occupancy_grid`、`/height_grid` |
| `aruco_detector` | 视觉目标检测层 | `scripts/aruco_node.py` | 识别 ArUco 并估计其在机体坐标系中的位置 |
| `llm_model` | 语言/多模态感知层 | `scripts/ins_processor.py`、`vlm_loader.py`、`grounding_dino_node.py` | 解析自然语言、加载 VLM、发布 GroundingDINO 检测 |
| `vln_mock` | 业务编排层 | `scripts/core_node.py`、`controller.py`、`vehicle_status_manager.py` | 状态管理、任务调度、value map 计算、控制执行 |

### 3.2 逻辑分层

可以把这套系统分成五层:

1. 输入层
   - `/instruction`
   - `/magv/camera/image_compressed/compressed`
   - `/magv/scan/3d`
   - `/magv/odometry/gt`

2. 协议层
   - `magv_vln_msgs/*`

3. 感知层
   - `pointcloud_to_grid_node`
   - `grounding_dino_node`
   - `aruco_detector_node`

4. 决策层
   - `ins_processor`
   - `vehicle_status_manager`
   - `core_node`

5. 执行层
   - `controller`
   - `/magv/omni_drive_controller/cmd_vel`

---

## 4. 运行时节点架构树

### 4.1 启动链路

总入口: `src/vln_mock/launch/vln_system.launch`

launch 中节点启动顺序大致是:

1. `instruction_latch_bridge`
2. `pointcloud_to_grid_node`
3. `aruco_detector_node`
4. `vlm_loader`
5. `ins_processor`
6. `grounding_dino_node`
7. `vehicle_status_manager`
8. `odom_tf_broadcaster`
9. `controller`
10. `core_node`

### 4.2 运行时拓扑树

```text
/instruction
└── instruction_latch_bridge
    └── /instruction (latched)
        └── ins_processor
            ├── /vlm_query
            │   └── vlm_loader
            │       └── /vlm_response
            │           ├── ins_processor
            │           └── core_node
            └── /subtasks
                └── vehicle_status_manager
                    └── /vln_status
                        ├── core_node
                        ├── controller
                        └── grounding_dino_node

/magv/camera/image_compressed/compressed
├── aruco_detector_node
│   └── /aruco_info
│       └── core_node
├── vlm_loader
└── grounding_dino_node
    └── /grounding_dino/detections
        └── core_node

/magv/scan/3d
├── pointcloud_to_grid_node
│   ├── /occupancy_grid
│   │   └── core_node
│   └── /height_grid
└── vehicle_status_manager

/magv/odometry/gt
├── controller
├── core_node
├── grounding_dino_node
├── vehicle_status_manager
└── odom_tf_broadcaster
    └── TF: map -> base_footprint (+ alias magv/base_link)

core_node
├── /grounding_dino/prompt
├── /world_goal
│   └── controller
├── /velocity_goal
│   └── controller
├── /core_feedback
│   └── vehicle_status_manager
└── /status
```

### 4.3 重要观察

- `core_node` 与 `ins_processor` 共用 `/vlm_response`，这是一个很脆弱的协议设计。
- `core_node` 实际上既是“局部规划器”，又兼任“任务编排器”和“ArUco 任务收尾器”，职责偏重。
- `controller` 当前是二维平面控制器，三维无人机控制没有真正实现。

---

## 5. 完整工作流

### 5.1 指令处理流

1. 用户向 `/instruction` 发送文本指令。
2. `instruction_latch_bridge` 抢先订阅首条指令并在同一 topic 上做 latched 重发布:
   - 证据: `src/vln_mock/scripts/instruction_latch_bridge.py:13-15`, `:21`, `:32`, `:46`
3. `ins_processor` 收到指令后等待 `/VLM_Status` 变为 `True`。
4. 如果 VLM 未及时就绪，则退化为 `instruction_offline` 离线解析:
   - 证据: `src/llm_model/scripts/ins_processor.py:122-141`
5. `vlm_loader` 接收 `/vlm_query`，返回结构化 JSON 子任务。
6. `ins_processor` 将子任务发布到 `/subtasks`:
   - 证据: `src/llm_model/scripts/ins_processor.py:182-227`

### 5.2 状态机流

1. `vehicle_status_manager` 收到 `/subtasks` 后:
   - 记录任务列表
   - 将状态置为 `STATE_INITIALIZING`
   - 证据: `src/vln_mock/scripts/vehicle_status_manager.py:89-104`
2. `core_node` 通过 `/vln_status` 进入初始化流程。
3. 初始化结束后，`core_node` 通过 `/core_feedback` 通知状态机。
4. `vehicle_status_manager` 收到 `"initialization_complete"` 后切到 `STATE_NAVIGATION`:
   - 证据: `src/vln_mock/scripts/vehicle_status_manager.py:107-118`
5. 当前子任务执行完后，再次切换到下一个子任务或最终 `IDLE`。

### 5.3 初始化扫描流

1. `core_node` 在 `STATE_INITIALIZING` 中读取当前子任务。
2. 如果子任务含有 `goal`，向 `/grounding_dino/prompt` 发送目标名词。
3. `core_node` 向 `controller` 发 `/velocity_goal`，执行 360 度转圈扫描。
4. `grounding_dino_node` 在扫描期间不断输出 2D 检测框。
5. `core_node` 将检测结果与当前 yaw 绑定，存入 `directional_detections`。
6. 旋转结束后读取 `/occupancy_grid`，计算 value map 和路径点。

### 5.4 导航执行流

1. `core_node` 进入 `navigation_active=True`。
2. `control_timer_callback()` 每 0.1s 将当前 waypoint 转成 `/world_goal`。
3. `controller` 根据 `/world_goal` 与当前 odom 做 PID，输出 `/magv/omni_drive_controller/cmd_vel`。
4. `vehicle_status_manager` 持续从点云中做前向障碍检测，必要时触发急停。

### 5.5 ArUco 收尾流

1. 导航过程中如果 `aruco_detector_node` 输出唯一目标，则 `core_node` 直接锁定该目标。
2. 如果检测到多个 ArUco，则 `core_node` 再次通过 `/vlm_query` 请求视觉问答判断哪一个是目标。
3. `core_node` 将 ArUco 的机体系位置转换成世界系目标点，发布 `/world_goal`。
4. 到达后发布 `/status=0`，任务完成。

---

## 6. 坐标系与变换链梳理

### 6.1 代码里出现的关键坐标系

| 坐标系 | 角色 | 在代码中的用法 |
| --- | --- | --- |
| `map` | 全局世界坐标 | 控制目标、value map、路径点、odometry parent frame |
| `base_footprint` | 机体平面坐标 | `aruco_info` header frame，launch 中默认 TF child |
| `magv/base_link` | 机体系别名 | `vehicle_status_manager` 默认 base frame |
| camera optical | OpenCV 相机系 | ArUco `tvec` 原始输出 |
| pointcloud frame | 雷达原始坐标 | 点云输入 frame，需变换到 `map` 或 `base` |

### 6.2 当前仓库里真正提供的 TF

仓库内只明确提供了一条动态 TF:

- `odom_tf_broadcaster` 从 `/magv/odometry/gt` 广播:
  - `parent = msg.header.frame_id`
  - `child = msg.child_frame_id or base_footprint`
  - 可选 alias: `magv/base_link`
  - 证据: `src/vln_mock/scripts/odom_tf_broadcaster.py:18-37`

也就是说，仓库内没有明确提供:

- `base_link -> camera`
- `base_link -> lidar`
- `map -> odom`
- `camera optical -> camera link`
- `lidar -> base_link`

### 6.3 `params.txt` 与真实代码的关系

`params.txt` 记录了:

- 相机内参/外参
- 激光雷达位置和姿态
- 小车几何参数

但实际代码只显式消费了相机相关参数:

- `camera_translation_*`
- `camera_pitch`
- `image_width`
- `horizontal_fov`

雷达外参在仓库内没有被真正接入 TF 链，也没有由任何节点读取 `params.txt` 自动配置。

这意味着:

- `params.txt` 更像人工备忘录，而不是当前系统的单一事实源。
- 对本地仿真或世界环境接入来说，真正决定行为的是 launch 参数和 TF 发布链，而不是 `params.txt`。

### 6.4 当前系统的几何假设

当前系统隐含地假设:

1. 导航是二维平面导航。
2. 机器人主要沿 `x-y` 平面运动。
3. yaw 是唯一需要控制的姿态自由度。
4. `base_footprint` 和 `magv/base_link` 可以视为同一个 frame。
5. 相机前向与机器人前向近似对齐。
6. 点云只要能“看起来”转到 map 或 base，就可以直接用于规划与急停。

这些假设对于地面小车勉强成立，但对无人机并不成立。

---

## 7. 可能存在的坐标系转换 bug

这一章是本报告的重点。下面按照“影响等级 + 置信度”列出。

### 7.1 P0: TF 失败时仍发布标成 `map` 的网格

证据:

- `src/pointcloud_to_grid/src/pointcloud_to_grid_node.cpp:116-137`
- `src/pointcloud_to_grid/include/pointcloud_to_grid/pointcloud_to_grid_core.hpp:40-51`

现象:

- 节点尝试把点云从 `src_frame` 转到 `target_frame`。
- 如果 TF 失败，它会保留原始点云 `cloud_ptr = msg`。
- 但 grid 的 `header.frame_id` 依然是 `frame_out`，默认值是 `map`。

后果:

- 如果雷达点云原本不是 `map`，但 TF 恰好缺失，系统会发布“内容仍在雷达系/机体系、标签却写着 map”的假地图。
- `core_node` 会把它当世界系占据图使用，value map、路径点和导航目标全部发生系统性偏差。

结论:

- 这是高置信度、高影响的坐标系 bug。

### 7.2 P0: `/occupancy_grid` 实际上是 intensity grid，不是真正占据图

证据:

- launch 将 intensity map topic 直接绑定到了 `/occupancy_grid`:
  - `src/vln_mock/launch/vln_system.launch:35-36`
- 点云节点写入的是 intensity/height，而不是 occupancy:
  - `src/pointcloud_to_grid/src/pointcloud_to_grid_node.cpp:178-182`
- `core_node` 用 `> 50` 作为障碍物判据:
  - `src/vln_mock/scripts/core_node.py:515-517`

问题本质:

- 这是“地图语义错位”问题，但最终会直接表现为坐标空间中的障碍位置错误。
- 如果点云没有 intensity 字段，代码会把 `inten=0.0` 写入 grid。
- 这样 `/occupancy_grid` 里很多真正有点云的格子值会是 `0`，而不是“occupied”。

后果:

- 规划层会把大量真实障碍当成自由空间。
- “坐标变换正确”也救不了“占据含义错误”的地图。

结论:

- 这是当前工程最核心的 P0 问题之一。

### 7.3 P0: ArUco 相机 pitch 旋转方向大概率反了

证据:

- `src/aruco_detector/scripts/aruco_node.py:173-188`
- 相机俯仰参数来自:
  - `src/vln_mock/launch/vln_system.launch:55-58`
  - `params.txt`

代码逻辑:

1. 先把 OpenCV 光学坐标系 `(x right, y down, z forward)` 映射到基座 nominal 坐标。
2. 再绕 base Y 轴乘以:

```text
[[ c, 0, -s],
 [ 0, 1,  0],
 [ s, 0,  c]]
```

风险判断:

- 若 `camera_pitch=0.314` 表示“相机向下俯仰”，那么相机前方目标在 base 系下通常应表现为“前方且 z 更低”。
- 当前矩阵更像把前向向量旋到了“z 更高”的方向。
- 这会让 ArUco 目标的 z 值、前向距离乃至最终世界坐标偏差。

结论:

- 这是高影响、中高置信度 bug。
- 建议在仿真里放一个已知位置 ArUco，直接对比 `_cam_to_base()` 输出。

### 7.4 P0: 紧急避障在 TF 缺失时继续按 base frame 解释点云

证据:

- `src/vln_mock/scripts/vehicle_status_manager.py:167-186`
- `src/vln_mock/scripts/vehicle_status_manager.py:188-230`

现象:

- 急停逻辑希望把点云转到 `base_frame=magv/base_link`。
- 若 TF 查找失败，它只报警告，但继续用原始 `(x, y, z)` 做“前向走廊”判断。

后果:

- 如果点云 frame 不是 base frame，急停会出现误触发或漏检。
- 这在 world 仿真里尤其危险，因为传感器插件通常有自己独立 frame。

结论:

- 这是另一个高置信度的坐标解释 bug。

### 7.5 P1: DINO 检测的距离惩罚相对的是世界原点，不是机器人当前位置

证据:

- `src/vln_mock/scripts/core_node.py:621-625`

具体问题:

```python
distance = math.sqrt(x**2 + y**2)
distance_penalty = math.exp(-0.1 * distance)
```

这里 `x, y` 是世界坐标格点，而不是相对机器人位置的 `dx, dy`。

后果:

- 机器人一旦不在世界原点附近，value map 对“近/远”的判断就会发生系统性失真。
- 同样的局部场景，换个地图起点，value 分布就变了。

结论:

- 这是高置信度的空间评价 bug。

### 7.6 P1: DINO 扇区投影只用了 yaw 和像素中心，没有用相机外参

证据:

- `src/vln_mock/scripts/core_node.py:611-617`
- 相机外参只在 ArUco 节点里使用，没有进入 DINO value map 投影链。

现象:

- `core_node` 用 bbox 中心像素换算成角度，再与扫描时 yaw 叠加，得到绝对检测方向。
- 但没有显式考虑:
  - 相机相对 base 的平移 `(+0.5, -0.04, +0.57)`
  - 相机俯仰角 `0.314`

后果:

- 在远距离上，这种简化可能还能工作。
- 在近场、侧前方目标、机体尺寸不可忽略的条件下，扇区中心会偏。

结论:

- 这是中高置信度的几何近似误差。

### 7.7 P1: `base_footprint` 与 `magv/base_link` 被当成完全同义

证据:

- launch:
  - `src/vln_mock/launch/vln_system.launch:111-115`
- broadcaster:
  - `src/vln_mock/scripts/odom_tf_broadcaster.py:18-37`
- ArUco 输出 frame:
  - `src/aruco_detector/scripts/aruco_node.py:165-168`
- ES 默认 base frame:
  - `src/vln_mock/scripts/vehicle_status_manager.py:80`

问题本质:

- `base_footprint` 通常是“将机体投影到地平面的 2D frame”。
- `base_link` 通常保留真实 6D 位姿。
- 当前实现把同一份四元数和位置同时广播给二者。

后果:

- 对地面车，平地上可能暂时看不出问题。
- 对坡道、起伏地形、无人机、带俯仰/横滚的平台，这个假设会直接破坏 frame 语义。

结论:

- 这是架构层面的 frame 设计问题。

### 7.8 P1: 网格原点固定在 `(0, 0)`，不是随机器人移动的局部图

证据:

- launch:
  - `src/vln_mock/launch/vln_system.launch:25-29`
- grid origin:
  - `src/pointcloud_to_grid/include/pointcloud_to_grid/pointcloud_to_grid_core.hpp:46-48`

现象:

- 地图中心由 `position_x/position_y` 固定给出。
- 代码注释写的是“center the map around the robot position”，但实际上没有订阅 odom 动态更新。

后果:

- 一旦机器人偏离原点，局部图就不再真正围绕机器人。
- 对长航程导航或 world 级仿真不友好。

结论:

- 这是高概率的局部地图坐标漂移问题。

### 7.9 P2: 图像横向像素到 yaw 偏移的符号约定存在风险

证据:

- `src/vln_mock/scripts/core_node.py:612-617`

具体逻辑:

```python
center_angle_offset = math.atan((self.image_width / 2 - center_pixel) / self.f_x)
absolute_detection_angle = detection_yaw + center_angle_offset
```

说明:

- 这一写法默认“图像右侧目标对应负 yaw 偏移”。
- 如果相机/机体系 yaw 正方向、光学坐标约定或图像镜像方向与此不一致，DINO 扇区会左右颠倒。

结论:

- 这是中等置信度风险项。
- 需要在仿真里做“目标在左/右侧”对照实验才能最终坐实。

---

## 8. 已确认的其他关键问题

### 8.1 当前工程运行环境是硬编码的，不可移植

证据:

- 几乎所有 Python 节点都通过 `/catkin_ws/venv310/bin/python3` 启动:
  - `src/vln_mock/launch/vln_system.launch:10`, `:44`, `:64`, `:76`, `:91`, `:103`, `:118`, `:141`
- `PYTHONPATH` 被硬编码成 Python 3.10:
  - `src/vln_mock/launch/vln_system.launch:13`, `:47`, `:67`, `:79`, `:94`, `:106`, `:121`, `:144`
- 当前工作区真实虚拟环境是:
  - `venv310/pyvenv.cfg` 显示 Python `3.8.10`

结论:

- 当前 launch 文件明显来自另一个工作目录或容器环境。
- 在当前仓库路径下直接 `roslaunch`，运行环境大概率不成立。

### 8.2 模型资产与代码配置已经漂移

证据:

- `vlm_loader` launch 配置期待:
  - `$(find llm_model)/models/Qwen2.5-VL-7B-Instruct`
  - 证据: `src/vln_mock/launch/vln_system.launch:69-73`
- `grounding_dino_node` 期待:
  - `/catkin_ws/src/llm_model/models/GroundingDino/grounding-dino-base`
  - 证据: `src/vln_mock/launch/vln_system.launch:96-99`
- 实际仓库里存在的模型目录只有:
  - `src/llm_model/models/microsoft--Florence-2-large`

结论:

- 代码配置和仓库内资产不一致，系统无法按当前 launch 正常完成模型加载。
- 对于本次本地修复环境，考虑到 RTX 4060 Laptop 8 GB 显存与磁盘余量，更现实的 Qwen 落地目标应下调为 `Qwen/Qwen2-VL-2B-Instruct` 或同级别轻量模型，而不是直接强推 7B。

### 8.3 当前 catkin 重建已经坏掉

验证结果:

- `python3 -m py_compile src/vln_mock/scripts/*.py src/llm_model/scripts/*.py src/aruco_detector/scripts/*.py`
  - 结果: 通过
- `cmake --build build -j2`
  - 结果: 失败

错误证据:

- `build/CMakeCache.txt:349`
- `build/CMakeCache.txt:632`

缓存里仍然指向:

```text
/home/cq/vln_test-fix_bug/venv/lib/python3.8/site-packages/em.py
```

但当前仓库下没有这个 `venv/` 路径。

结论:

- 这是构建缓存污染问题。
- 在进入仿真联调前必须先清理并重建 `build/` 与 `devel/`。

### 8.4 规划器现在不是“路径规划”，而是“取 value map 最大值”

证据:

- `src/vln_mock/scripts/core_node.py:741-810`

关键事实:

- `num_waypoints = 1`
- 直接 `np.argmax(temp_values)` 取一个点
- 没有 A*
- 没有 Dijkstra
- 没有 RRT
- 没有障碍膨胀
- 没有重规划

结论:

- 当前路径规划能力只到“选一个吸引点”，还不是可泛化的导航规划器。

### 8.5 未知区被当成自由区

证据:

- grid 初始化值为 `-1`:
  - `src/pointcloud_to_grid/include/pointcloud_to_grid/pointcloud_to_grid_core.hpp:51`
- `core_node` 只把 `>50` 看作障碍:
  - `src/vln_mock/scripts/core_node.py:515-517`

后果:

- 未观测区域会得到正常 value，甚至成为最优点。
- 在仿真里这会表现为机器人主动走向地图边界或未感知区域。

### 8.6 value map 带随机噪声，不利于复现

证据:

- `src/vln_mock/scripts/core_node.py:728`

现象:

- 每个格子都会叠加 `np.random.normal(0, 1.0)`。

后果:

- 同样场景下 value map 不可复现。
- 不利于调参、回归测试和科研对比。

### 8.7 `/vlm_response` 协议复用，存在消息串扰

证据:

- `core_node` 订阅 `/vlm_response`:
  - `src/vln_mock/scripts/core_node.py:90`, `:434-441`, `:941-1014`
- `ins_processor` 也订阅同一 topic:
  - `src/llm_model/scripts/ins_processor.py:53`, `:182-227`
- `vlm_loader` 同时承载指令解析与视觉问答:
  - `src/llm_model/scripts/vlm_loader.py:130-182`

后果:

- 指令解析结果和 ArUco 视觉问答结果共用一个 topic，靠 JSON 字段猜意图。
- `core_node` 很容易误收到本该给 `ins_processor` 的响应。

这是一个非常典型的“原型期快跑可用，工程化后必须拆分”的问题。

### 8.8 代码漂移与死代码已经出现

证据:

- `core_node` 中:
  - `path_point_pub` 定义了但没有真正发布使用
  - `controller_body_pub` 只在死代码分支里出现
  - `_check_aruco_arrival()` 的 `return` 后面还有旧逻辑
  - 证据: `src/vln_mock/scripts/core_node.py:73`, `:75`, `:1080-1111`
- `detected_objects` 变量定义后未形成稳定流水线:
  - `src/vln_mock/scripts/core_node.py:44`

结论:

- 代码已经开始出现“思路演进后旧接口未清理”的现象。

### 8.9 README 已经过期

证据:

- README 仍写 `instruction_processor`
  - `src/vln_mock/README.md:11`
- README 写有 `test_vln_system.py`
  - `src/vln_mock/README.md:20`, `:55-59`
- 仓库中并不存在对应文件。

结论:

- 文档与代码不同步，后续接手开发非常容易被误导。

### 8.10 `gazebo_world/` 目前为空

事实:

- 根目录下存在 `gazebo_world/`
- 当前目录没有 world 文件、模型、launch、桥接配置

结论:

- “本地获取一个 world 仿真环境并运行”这件事还没有开始接入工程主线。

---

## 9. 项目痛点总结

### 9.1 几何与感知链路痛点

- TF 链不完整，很多地方默认“上游已经把 frame 弄好了”。
- 相机外参、雷达外参没有形成统一配置入口。
- `params.txt` 只是备忘录，不是系统事实源。
- 点云地图、DINO 检测、ArUco 定位三条感知链对 frame 的理解不完全统一。

### 9.2 决策与规划痛点

- 子任务拆解使用 JSON-over-String，协议脆弱。
- value map 与 path planning 强耦合在 `core_node`。
- 当前“路径规划”缺少显式障碍搜索与重规划机制。
- 状态机、规划器、视觉问答都压在 topic 交互上，缺少 action/service 级事务约束。

### 9.3 控制与平台适配痛点

- 控制器明确是二维控制器:
  - `pos_error.z = 0.0`
  - `point.position.z = 0.0`
  - 使用 `/magv/omni_drive_controller/cmd_vel`
- 这意味着当前仓库并没有真正覆盖无人机三维导航。
- 如果你后续要兼容小车和无人机，必须把“平台接口层”单独抽出来。

### 9.4 工程化痛点

- launch 路径硬编码。
- 模型路径硬编码。
- 构建缓存污染。
- 没有测试。
- 没有 world 仿真。
- 文档漂移。

---

## 10. 可以优化的地方

下面按优先级给出建议。

### 10.1 P0 优化建议: 不做这些，后续 world 测试会非常痛苦

1. 重构占据图语义
   - 不要再把 intensity grid 直接当 `/occupancy_grid`。
   - 至少应改成:
     - 有点即占据
     - 或高度过滤 + 命中计数后转 occupancy probability
   - 最好区分:
     - `occupancy_grid`
     - `intensity_grid`
     - `height_grid`

2. 收口 TF 体系
   - 明确发布:
     - `map -> odom`
     - `odom -> base_link`
     - `base_link -> camera_link`
     - `base_link -> lidar_link`
     - `camera_link -> camera_optical`
   - `base_footprint` 和 `base_link` 不要再简单复用同一姿态。

3. 修复 launch 环境硬编码
   - 去掉 `/catkin_ws/venv310`
   - 去掉固定 `python3.10/site-packages`
   - 改为:
     - 使用 `#!/usr/bin/env python3`
     - 或使用 `$(env VIRTUAL_ENV)` 风格
     - 或直接依赖 catkin/python 环境

4. 修复模型资产路径
   - 统一选择一套真实存在且你要长期维护的 VLM/VLM-detector。
   - 现在仓库内模型与 launch 配置明显不一致。

5. 清理构建缓存
   - 当前 `build/` 和 `devel/` 不是可信状态。

### 10.2 P1 优化建议: 提升算法正确性与可调试性

1. 将 `core_node` 拆成三个模块
   - `task_coordinator`
   - `value_map_builder`
   - `goal_selector / local_planner`

2. 让 DINO 投影显式依赖相机模型
   - 统一相机内外参
   - 使用相机光线与 base/map 的几何投影
   - 不要只靠 bbox 中心角和 yaw

3. 把 value map 做成可复现
   - 删除随机噪声
   - 或引入固定随机种子

4. 增加真正的路径规划
   - 先做 A* on occupancy grid
   - 再加障碍膨胀和轨迹平滑
   - 最后再看是否需要更高级方法

5. 拆分 `/vlm_response`
   - 指令解析单独 topic 或 service
   - ArUco 问答单独 topic 或 service

### 10.3 P2 优化建议: 提升工程可维护性

1. 用 YAML 管理参数
   - 相机参数
   - 雷达参数
   - 控制参数
   - 模型参数
   - 仿真参数

2. 让 `params.txt` 退出舞台
   - 要么把它转成正式 YAML
   - 要么删掉，避免“双事实源”

3. 给每个节点补状态/诊断 topic
   - 当前只有部分节点做了状态输出

4. 文档收口
   - 修 README
   - 给仿真与实车分别写 bringup 文档

---

## 11. 面向小车/无人机的适配判断

### 11.1 小车适配性

当前版本更接近小车/地面全向底盘原型，原因如下:

- 使用 `base_footprint`
- 只控制 `x-y-yaw`
- 明确忽略 z
- 使用二维前向障碍走廊
- 路径点 z 固定为 0

### 11.2 无人机适配性

当前版本并不真正适用于无人机，原因如下:

1. 控制层没有 z 控制闭环。
2. 没有 roll/pitch 控制。
3. `base_footprint` 语义本来就是地面平台语义。
4. 点云急停逻辑是二维前向走廊，不适合空中平台。
5. 规划层是二维 occupancy/value map，不是 3D voxel 或 ESDF。

### 11.3 结论

如果你后续确实要兼容“小车 + 无人机”，建议从架构上拆成:

- `platform_ground/`
- `platform_uav/`
- `common_vln/`

其中:

- 感知与语言可复用
- 控制与局部规划必须分平台实现

---

## 12. 本地 world 仿真接入建议

### 12.1 当前仿真状态

当前仓库对 world 仿真的支持现状:

- 有 `gazebo_world/` 目录
- 但目录为空
- launch 里没有 world 节点
- `use_sim_time=false`
- 没有 `robot_state_publisher`
- 没有静态 TF 发布器
- 没有 Gazebo sensor plugin 配置

所以，当前并不是“差一步就能跑仿真”，而是“还没把仿真正式接进来”。

### 12.2 推荐的仿真目录结构

建议将 `gazebo_world/` 真正扩成如下结构:

```text
gazebo_world/
├── launch/
│   ├── world.launch
│   ├── robot_bringup.launch
│   ├── sensors.launch
│   └── vln_sim.launch
├── worlds/
│   ├── indoor_vln.world
│   └── debug_empty.world
├── models/
│   ├── tree/
│   ├── bench/
│   ├── cone/
│   └── aruco_board/
├── rviz/
│   └── vln_debug.rviz
└── config/
    ├── robot_frames.yaml
    ├── sensor_topics.yaml
    └── vln_params.yaml
```

### 12.3 最小仿真 topic 合同

为了让现有 VLN 主链跑起来，仿真至少要提供:

- `/clock`
- `/tf`
- `/tf_static`
- `/magv/odometry/gt`
- `/magv/scan/3d`
- `/magv/camera/image_compressed/compressed`
- `/magv/omni_drive_controller/cmd_vel`

同时需要有:

- `map/base_link/camera/lidar` 的完整 frame 树
- 世界中能被 DINO/ArUco 识别的目标物

### 12.4 推荐 launch 组合方式

建议把整体仿真分四段:

1. `world.launch`
   - 启 Gazebo world
   - `use_sim_time=true`

2. `robot_bringup.launch`
   - 机器人模型
   - 控制器
   - `robot_state_publisher`
   - 静态传感器外参 TF

3. `sensors.launch`
   - 相机插件
   - 雷达插件
   - odom 发布

4. `vln_sim.launch`
   - 引用前三者
   - 再拉起 `vln_system.launch`

### 12.5 仿真接入前的建议改动

1. `use_sim_time` 改成可配置参数。
2. 把 topic 名全部收口到一个 YAML。
3. 为 camera 和 lidar 增加静态 TF 发布器。
4. 在 RViz 中同时显示:
   - TF
   - 点云
   - occupancy grid
   - value map
   - world goal
   - ArUco marker pose

### 12.6 建议的 world 测试场景

建议至少做 5 组场景:

1. 空旷场景
   - 只验证指令拆解和基本控制

2. 单目标可见场景
   - 比如“go to the tree”
   - 验证 DINO -> value map -> world goal

3. 目标被障碍遮挡场景
   - 验证 occupancy 与规划是否冲突

4. 多个 ArUco 标记场景
   - 验证 `/vlm_query` 二次决策逻辑

5. TF 故障注入场景
   - 故意断掉 camera 或 lidar TF
   - 验证系统是否 fail-fast，而不是带错 frame 继续运行

---

## 13. 建议的开发路线图

### 13.1 第一阶段: 先让工程能稳定构建和启动

目标:

- 修 build/devel 缓存
- 修 launch 环境路径
- 修模型路径
- 修 README

交付标准:

- `catkin_make` 或等价构建通过
- `roslaunch vln_mock vln_system.launch` 能正常起节点
- 所有节点至少能进入“等待传感器输入”状态

### 13.2 第二阶段: 收口 TF 和地图语义

目标:

- 给 lidar/camera 建立正式 TF 树
- 把 intensity map 和 occupancy grid 分离
- 修复 TF 缺失时仍冒充 `map` 的问题

交付标准:

- RViz 中 frame 树清晰
- 点云、网格、world goal 在 RViz 中几何关系一致

### 13.3 第三阶段: 把规划从“argmax 选点”升级成真正局部规划

目标:

- 引入 A* 或 Dijkstra
- 处理 unknown/occupied/free 三态
- 做局部重规划

交付标准:

- 面对障碍物不再只靠急停
- world 仿真里能绕障

### 13.4 第四阶段: 接入本地 world 仿真

目标:

- 建立 `gazebo_world/` 完整内容
- 打通传感器插件和 topic
- 形成一键仿真 launch

交付标准:

- 可以在本地 world 里反复回归测试固定指令集

### 13.5 第五阶段: 再考虑无人机适配

目标:

- 抽象平台接口
- 补三维控制/三维规划
- 将 base_footprint 语义与 UAV frame 彻底分离

交付标准:

- 地面车与无人机共享上层任务与感知，不共享底层控制

---

## 14. 我建议你优先修的 10 件事

1. 修 launch 中所有 `/catkin_ws/venv310` 和 `python3.10` 硬编码。
2. 清理 `build/`、`devel/`，重新初始化当前工作区构建缓存。
3. 统一模型目录，决定到底使用 Qwen、Florence，还是别的 VLM。
4. 给 camera/lidar 正式建立静态 TF，不再依赖“外参写在注释里”。
5. 修 `pointcloud_to_grid`，不要再把 intensity map 当 occupancy。
6. 修 `pointcloud_to_grid` 的 TF 失败降级逻辑，失败时不要发布假 `map` 数据。
7. 修 `vehicle_status_manager` 的急停坐标解释，TF 不通时直接 fail-safe。
8. 修 `aruco_node.py` 中 `_cam_to_base()` 的 pitch 方向并做仿真对照验证。
9. 把 `core_node` 的 `/vlm_response` 串扰拆开。
10. 在 world 接入前，至少先补一个 RViz 调试工作流和 3 个基础回归场景。

---

## 15. 本次分析的验证结论

本次没有做端到端 `roslaunch + 仿真 + 模型推理` 的完整运行，原因很明确:

- launch 环境路径与当前工作区不一致
- 模型目录与配置不一致
- `gazebo_world/` 为空
- catkin 重建缓存已损坏

但已经完成的验证包括:

- Python 节点脚本语法检查通过
- 包结构、launch、topic、message、状态机、控制流、感知流已完整梳理
- 关键坐标系风险和工程风险已经定位到具体代码位置

---

## 16. 最终结论

如果你的目标是“在本地获取一个 world 仿真环境并进行本地测试运行”，那么这份仓库当前最需要的不是继续堆功能，而是先做三件基础建设:

1. 运行环境可移植化
2. TF/地图语义正确化
3. 仿真 world 接入工程化

从研发视角看，这套代码最值得保留的是:

- 节点职责已经初步分层
- 指令拆解、状态机、感知融合、控制闭环已经形成最小链路

最需要尽快重构的是:

- 坐标系与 TF
- occupancy 语义
- 规划器
- 运行环境与模型管理

一句话收束:

这个项目已经具备“原型系统”的骨架，但要进入稳定的本地 world 联调阶段，必须先把“几何正确性”和“工程正确性”补上。

---

## 17. 2026-04-03 二次整改复核

说明:

- 本节基于当前仓库最新状态补充。
- 前文 1 到 16 章保留的是“第一次系统审查时的基线判断”。
- 如果本节与前文某些历史描述不一致，以本节的整改状态为准。

### 17.1 总体复核结论

截至 2026-04-03，仓库已经从“仅有原型骨架”推进到“可本地构建、可本地模型推理、可启动轻量 Gazebo 仿真脚手架、可切换 rover/uav mock 资产”的状态。

已经明显改善的部分:

- 本地构建与缓存污染问题已处理。
- 模型路径、launch 参数、topic 参数的硬编码大幅减少。
- `pointcloud_to_grid` 的 TF 失败降级和 occupancy 语义问题已修正。
- `vehicle_status_manager` 的 TF 缺失 fail-safe 已补上。
- `aruco_node.py` 的相机 pitch 变换和输出 frame 已修正。
- `gazebo_world/` 不再为空，已经具备 world、目标物和轻量机器人资产。

仍然没有完全闭环的部分:

- 无人机仍然不是完整 3D 飞控动力学方案。
- `core_node` 仍是 value map + argmax 选点，不是真正的局部规划器。
- `base_footprint` / `base_link` 的平台语义还没有做成彻底的平台抽象。
- 仿真里的 ArUco 目标目前是 “ArUco-style board”，不是已经贴真实纹理并验证过检测精度的最终版本。

### 17.2 关键问题复核表

| 报告原问题 | 当前状态 | 结论 |
| --- | --- | --- |
| 7.1 TF 失败时仍发布标成 `map` 的假网格 | `pointcloud_to_grid` 已启用 `require_transform: true`，TF 不通时直接拒绝继续发布伪 `map` 数据 | 已解决 |
| 7.2 `/occupancy_grid` 实际是 intensity grid | 当前网格节点已按 `occupied_value` 写入占据语义，`/occupancy_grid` 不再复用 intensity 语义 | 已解决 |
| 7.3 ArUco pitch 方向可能反了 | `_cam_to_base()` 已重写并明确注释“downward pitch positive” | 代码层已解决，仍建议仿真标定复核 |
| 7.4 急停在 TF 缺失时继续误解点云 | `vehicle_status_manager` 在无有效 TF 时直接进入 fail-safe | 已解决 |
| 7.5 value map 距离/坐标解释混乱 | `core_node` 已使用 cell center 世界坐标 `(x+0.5, y+0.5)` 计算 value | 基本解决 |
| 7.7 `base_footprint` 与 `base_link` 混用 | launch 已支持 `alias_base_frame`，并允许关闭 `odom_tf_broadcaster` 避免 TF 互相覆盖 | 部分解决 |
| 8.1 运行环境硬编码 | `/catkin_ws/venv310` 与 `python3.10` 级别硬编码已收口；模型、topic、frame 可通过 launch/YAML 覆盖 | 基本解决 |
| 8.2 模型目录漂移/模型不一致 | Florence、Qwen、GroundingDINO 的本地下载、回退和 smoke test 已补齐 | 已解决 |
| 8.3 `gazebo_world/` 为空 | 已补 world、rover/uav 资产、起点终点和多类目标物 | 已解决 |
| 9.x 工程化薄弱 | 已新增 `LOCAL_BRINGUP.md`、下载脚本、模型 smoke test、构建脚本、清理脚本 | 大幅改善 |

### 17.3 当前仓库已具备的新能力

1. 可本地下载并验证的视觉模型链:
   - Florence-2
   - Qwen2-VL-2B-Instruct
   - GroundingDINO

2. 可本地切换的 Gazebo 轻量平台:
   - `robot_profile:=rover`
   - `robot_profile:=uav`

3. 可直接用于指令回归的 world 目标物:
   - bench
   - tree
   - cone gate
   - ArUco-style board
   - finish zone

4. 更稳定的启动方式:
   - `vln_system.launch`
   - `vln_sim.launch`
   - `tools/rebuild_workspace.sh`
   - `tools/smoke_test.sh`
   - `tools/cleanup_artifacts.sh`

### 17.4 仍建议优先推进的后续项

1. 把 `aruco_style_board` 升级成真实纹理 ArUco 板，并做一轮实际检测闭环验证。
2. 把 `core_node` 从 `argmax` 选点升级为 A* 或 Dijkstra on occupancy grid。
3. 为外部真实仿真机器人建立正式 `robot_state_publisher + urdf/xacro + tf tree`。
4. 将 UAV 平台改造成真正的 3D 控制接口，而不是当前的平面 hover mock。
5. 增加固定指令集回归测试，把 “bench/tree/cone/board” 场景跑成标准化测试用例。
