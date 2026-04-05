#!/usr/bin/env python3

import argparse
import json
import math
import os
import queue
import shlex
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import tkinter as tk
from PIL import Image, ImageDraw, ImageFont
from tkinter import scrolledtext, ttk


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TASKS_FILE = ROOT / "tests" / "goldens" / "mock_uav_tasks.json"
DEFAULT_WORLD_NAME = "campus_stub"
DEFAULT_DEBUG_BOXES_TOPIC = "/vln/perception/debug_boxes"


@dataclass
class DebugBox:
    label: str
    confidence: float
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    source: str = ""


@dataclass
class GridSpec:
    width: int = 0
    height: int = 0
    resolution: float = 1.0
    origin_x: float = 0.0
    origin_y: float = 0.0
    data: Optional[np.ndarray] = None


@dataclass
class PoseSnapshot:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    yaw_rad: float = 0.0


def parse_operator_command(text: str) -> Dict[str, str]:
    stripped = text.strip()
    if not stripped:
        return {"kind": "noop"}
    if stripped.startswith("/episode "):
        return {"kind": "episode", "episode_name": stripped.split(None, 1)[1].strip()}
    if stripped.startswith("/reset"):
        suffix = stripped.split(None, 1)[1].strip() if " " in stripped else ""
        return {"kind": "reset", "episode_name": suffix}
    return {"kind": "instruction", "instruction": stripped}


def load_episode_names(tasks_file: Path) -> List[str]:
    if not tasks_file.exists():
        return []
    payload = json.loads(tasks_file.read_text(encoding="utf-8"))
    names = [str(item["name"]) for item in payload if isinstance(item, dict) and item.get("name")]
    return sorted(names)


def build_launch_command(repo_root: Path, world_name: str, tasks_file: Path) -> List[str]:
    devel_setup = repo_root / "devel_local" / "setup.bash"
    load_env = repo_root / "tools" / "load_local_env.sh"
    shell_parts = [
        "source /opt/ros/noetic/setup.bash",
    ]
    if devel_setup.exists():
        shell_parts.append(f"source {shlex.quote(str(devel_setup))}")
    shell_parts.append(f"source {shlex.quote(str(load_env))}")
    shell_parts.append(
        "roslaunch vln_bringup evaluation.launch "
        f"headless:=false world_name:={shlex.quote(world_name)} tasks_file:={shlex.quote(str(tasks_file))}"
    )
    return ["bash", "-lc", " && ".join(shell_parts)]


def normalize_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def image_message_to_pil(msg) -> Optional[Image.Image]:
    if msg is None or not getattr(msg, "data", None):
        return None

    encoding = str(getattr(msg, "encoding", "")).lower()
    height = int(getattr(msg, "height", 0))
    width = int(getattr(msg, "width", 0))
    step = int(getattr(msg, "step", 0))
    if width <= 0 or height <= 0:
        return None

    if encoding in {"rgb8", "bgr8"}:
        channels = 3
        frame = np.frombuffer(msg.data, dtype=np.uint8).reshape((height, step // channels, channels))[:, :width, :]
        if encoding == "bgr8":
            frame = frame[:, :, ::-1]
        return Image.fromarray(frame, mode="RGB")

    if encoding in {"rgba8", "bgra8"}:
        channels = 4
        frame = np.frombuffer(msg.data, dtype=np.uint8).reshape((height, step // channels, channels))[:, :width, :]
        if encoding == "bgra8":
            frame = frame[:, :, [2, 1, 0, 3]]
        return Image.fromarray(frame, mode="RGBA").convert("RGB")

    if encoding == "mono8":
        frame = np.frombuffer(msg.data, dtype=np.uint8).reshape((height, step))[:, :width]
        return Image.fromarray(frame, mode="L").convert("RGB")

    return None


def draw_debug_boxes(image: Image.Image, boxes: Sequence[DebugBox], active_labels: Sequence[str]) -> Image.Image:
    rendered = image.copy()
    draw = ImageDraw.Draw(rendered)
    font = ImageFont.load_default()
    width, height = rendered.size
    active_lookup = set(active_labels)

    for box in boxes:
        x0 = int(max(0.0, min(1.0, box.xmin)) * width)
        y0 = int(max(0.0, min(1.0, box.ymin)) * height)
        x1 = int(max(0.0, min(1.0, box.xmax)) * width)
        y1 = int(max(0.0, min(1.0, box.ymax)) * height)
        color = "#5ce65c" if box.label in active_lookup else "#ffb347"
        draw.rectangle((x0, y0, x1, y1), outline=color, width=3)
        caption = f"{box.label} {box.confidence:.2f}"
        text_bbox = draw.textbbox((x0, y0), caption, font=font)
        text_height = text_bbox[3] - text_bbox[1]
        draw.rectangle((x0, max(0, y0 - text_height - 6), x0 + (text_bbox[2] - text_bbox[0]) + 6, y0), fill=color)
        draw.text((x0 + 3, max(0, y0 - text_height - 3)), caption, fill="black", font=font)
    return rendered


def scan_to_world_points(scan, pose: PoseSnapshot, sample_stride: int = 8) -> List[Tuple[float, float]]:
    if scan is None:
        return []
    points = []
    ranges = getattr(scan, "ranges", [])
    angle_min = float(getattr(scan, "angle_min", 0.0))
    angle_inc = float(getattr(scan, "angle_increment", 0.0))
    range_min = float(getattr(scan, "range_min", 0.0))
    range_max = float(getattr(scan, "range_max", 0.0))
    if angle_inc == 0.0:
        return points

    for index in range(0, len(ranges), max(1, sample_stride)):
        distance = float(ranges[index])
        if not math.isfinite(distance) or distance < range_min or distance > range_max:
            continue
        beam_yaw = pose.yaw_rad + angle_min + (angle_inc * index)
        points.append((pose.x + math.cos(beam_yaw) * distance, pose.y + math.sin(beam_yaw) * distance))
    return points


def build_value_map(
    grid: GridSpec,
    pose: PoseSnapshot,
    waypoints: Sequence[Tuple[float, float]],
    landmarks: Sequence[Tuple[str, float, float, float]],
    scan_points: Sequence[Tuple[float, float]],
    active_labels: Sequence[str],
) -> np.ndarray:
    width = max(1, grid.width)
    height = max(1, grid.height)
    resolution = max(0.2, grid.resolution)
    xs = grid.origin_x + (np.arange(width) + 0.5) * resolution
    ys = grid.origin_y + (np.arange(height) + 0.5) * resolution
    xx, yy = np.meshgrid(xs, ys)

    value = np.zeros((height, width), dtype=np.float32)
    if grid.data is not None and grid.data.size == width * height:
        occupancy = grid.data.reshape((height, width)).astype(np.float32) / 100.0
        value -= np.clip(occupancy, 0.0, 1.0) * 2.5

    def add_gaussian(center_x: float, center_y: float, sigma: float, weight: float) -> None:
        nonlocal value
        dist_sq = ((xx - center_x) ** 2 + (yy - center_y) ** 2) / max(1e-6, 2.0 * sigma * sigma)
        value += np.exp(-dist_sq).astype(np.float32) * weight

    for index, (x_coord, y_coord) in enumerate(waypoints[:8]):
        add_gaussian(x_coord, y_coord, sigma=1.2 + (index * 0.15), weight=max(0.4, 1.5 - index * 0.18))

    active_lookup = set(active_labels)
    for label, x_coord, y_coord, confidence in landmarks:
        weight = 0.6 + float(confidence)
        if label in active_lookup:
            weight += 1.2
        add_gaussian(x_coord, y_coord, sigma=1.0, weight=weight)

    for x_coord, y_coord in scan_points:
        add_gaussian(x_coord, y_coord, sigma=0.55, weight=-1.4)

    add_gaussian(pose.x, pose.y, sigma=0.85, weight=0.9)
    value -= np.hypot(xx - pose.x, yy - pose.y).astype(np.float32) * 0.02
    return value


def colorize_value_map(value_map: np.ndarray, size: Tuple[int, int]) -> Image.Image:
    if value_map.size == 0:
        return Image.new("RGB", size, "#1e2530")

    low = float(np.min(value_map))
    high = float(np.max(value_map))
    spread = max(1e-6, high - low)
    normalized = np.clip((value_map - low) / spread, 0.0, 1.0)

    red = np.clip(255.0 * np.minimum(1.0, normalized * 1.5), 0, 255).astype(np.uint8)
    green = np.clip(255.0 * (1.0 - np.abs(normalized - 0.5) * 1.7), 0, 255).astype(np.uint8)
    blue = np.clip(255.0 * (1.0 - normalized * 0.9), 0, 255).astype(np.uint8)
    rgb = np.dstack((red, green, blue))
    return Image.fromarray(rgb, mode="RGB").resize(size, resample=Image.BILINEAR)


class OperatorGui:
    def __init__(self, world_name: str, tasks_file: Path, autostart_sim: bool, geometry: str) -> None:
        self.world_name = world_name
        self.tasks_file = tasks_file
        self.autostart_sim = autostart_sim
        self.geometry = geometry

        self.root = tk.Tk()
        self.root.title("VLN Operator Client")
        self.root.geometry(self.geometry)
        self.root.minsize(1280, 760)

        self.state_lock = threading.Lock()
        self.ui_queue: "queue.Queue[Tuple[str, str]]" = queue.Queue()
        self.launch_process: Optional[subprocess.Popen] = None
        self.launch_log_handle = None
        self.ros_ready = False
        self.shutdown_requested = False

        self.runtime_state = None
        self.current_pose = PoseSnapshot(z=1.5)
        self.grid = GridSpec()
        self.compiled_mission = None
        self.trajectory_points: List[Tuple[float, float]] = []
        self.semantic_landmarks: Dict[str, Tuple[float, float, float]] = {}
        self.scan_msg = None
        self.latest_image_msg = None
        self.latest_image_seq = -1
        self.rendered_image_seq = -1
        self.debug_boxes: List[DebugBox] = []
        self.last_phase_logged = ""
        self.episode_names = load_episode_names(self.tasks_file)

        self.rospy = None
        self.reset_event_pub = None
        self.reset_proxy = None
        self.StringMsg = None

        self.camera_photo = None
        self.valuemap_photo = None

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        threading.Thread(target=self._boot_ros_stack, daemon=True).start()
        self.root.after(120, self._drain_ui_events)
        self.root.after(250, self._refresh_status_page)
        self.root.after(160, self._refresh_camera_page)
        self.root.after(500, self._refresh_valuemap_page)
        self.root.after(1000, self._poll_launch_process)

    def _build_ui(self) -> None:
        self.root.columnconfigure(0, weight=0, minsize=360)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        left = ttk.Frame(self.root, padding=12)
        left.grid(row=0, column=0, sticky="nsew")
        left.columnconfigure(0, weight=1)
        left.rowconfigure(2, weight=1)

        ttk.Label(left, text="VLN Chat", font=("TkDefaultFont", 13, "bold")).grid(row=0, column=0, sticky="w")
        self.connection_var = tk.StringVar(value=f"World: {self.world_name} | Launching simulation...")
        ttk.Label(left, textvariable=self.connection_var, wraplength=320).grid(row=1, column=0, sticky="ew", pady=(4, 10))

        controls = ttk.Frame(left)
        controls.grid(row=2, column=0, sticky="nsew")
        controls.columnconfigure(0, weight=1)
        controls.rowconfigure(1, weight=1)

        quick = ttk.Frame(controls)
        quick.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        quick.columnconfigure(0, weight=1)
        self.episode_var = tk.StringVar(value=self.episode_names[0] if self.episode_names else "")
        self.episode_combo = ttk.Combobox(quick, textvariable=self.episode_var, values=self.episode_names, state="readonly")
        self.episode_combo.grid(row=0, column=0, sticky="ew")
        ttk.Button(quick, text="Run Episode", command=self._run_selected_episode).grid(row=0, column=1, padx=(8, 0))
        ttk.Button(quick, text="Reset", command=self._publish_manual_reset).grid(row=0, column=2, padx=(8, 0))

        self.chat_log = scrolledtext.ScrolledText(controls, wrap="word", height=24, state="disabled")
        self.chat_log.grid(row=1, column=0, sticky="nsew")

        entry_row = ttk.Frame(controls)
        entry_row.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        entry_row.columnconfigure(0, weight=1)
        self.message_var = tk.StringVar()
        self.message_entry = ttk.Entry(entry_row, textvariable=self.message_var)
        self.message_entry.grid(row=0, column=0, sticky="ew")
        self.message_entry.bind("<Return>", lambda _event: self._handle_send())
        ttk.Button(entry_row, text="Send", command=self._handle_send).grid(row=0, column=1, padx=(8, 0))
        ttk.Label(
            controls,
            text="Send free-text VLN instructions, or use /episode <name> and /reset.",
            wraplength=320,
        ).grid(row=3, column=0, sticky="ew", pady=(8, 0))

        right = ttk.Frame(self.root, padding=(0, 12, 12, 12))
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)

        self.notebook = ttk.Notebook(right)
        self.notebook.grid(row=0, column=0, sticky="nsew")

        self.task_page = ttk.Frame(self.notebook, padding=12)
        self.camera_page = ttk.Frame(self.notebook, padding=12)
        self.valuemap_page = ttk.Frame(self.notebook, padding=12)
        self.notebook.add(self.task_page, text="Task")
        self.notebook.add(self.camera_page, text="FPV")
        self.notebook.add(self.valuemap_page, text="ValueMap")

        self._build_task_page()
        self._build_camera_page()
        self._build_valuemap_page()
        self._append_chat("system", "GUI ready. Simulation will auto-start and reuse an existing Gazebo session when possible.")

    def _build_task_page(self) -> None:
        self.task_page.columnconfigure(0, weight=1)
        self.task_page.rowconfigure(2, weight=1)

        summary = ttk.Frame(self.task_page)
        summary.grid(row=0, column=0, sticky="ew")
        for index in range(4):
            summary.columnconfigure(index, weight=1)

        self.phase_var = tk.StringVar(value="Phase: idle")
        self.pose_var = tk.StringVar(value="Pose: (0.0, 0.0, 0.0)")
        self.mission_var = tk.StringVar(value="Mission: none")
        self.trajectory_var = tk.StringVar(value="Trajectory: 0 waypoints")
        ttk.Label(summary, textvariable=self.phase_var).grid(row=0, column=0, sticky="w")
        ttk.Label(summary, textvariable=self.pose_var).grid(row=0, column=1, sticky="w")
        ttk.Label(summary, textvariable=self.mission_var).grid(row=0, column=2, sticky="w")
        ttk.Label(summary, textvariable=self.trajectory_var).grid(row=0, column=3, sticky="w")

        self.detail_var = tk.StringVar(value="Detail: waiting for runtime state")
        ttk.Label(self.task_page, textvariable=self.detail_var, wraplength=820).grid(row=1, column=0, sticky="ew", pady=(8, 12))

        split = ttk.Panedwindow(self.task_page, orient=tk.HORIZONTAL)
        split.grid(row=2, column=0, sticky="nsew")

        steps_frame = ttk.Labelframe(split, text="Mission Steps", padding=8)
        steps_frame.columnconfigure(0, weight=1)
        steps_frame.rowconfigure(0, weight=1)
        self.steps_tree = ttk.Treeview(steps_frame, columns=("step", "action", "target", "terminal"), show="headings", height=14)
        for column, width in (("step", 55), ("action", 80), ("target", 170), ("terminal", 70)):
            self.steps_tree.heading(column, text=column.title())
            self.steps_tree.column(column, width=width, anchor="center" if column in {"step", "terminal"} else "w")
        self.steps_tree.grid(row=0, column=0, sticky="nsew")
        split.add(steps_frame, weight=1)

        landmarks_frame = ttk.Labelframe(split, text="Semantic Landmarks", padding=8)
        landmarks_frame.columnconfigure(0, weight=1)
        landmarks_frame.rowconfigure(0, weight=1)
        self.landmarks_tree = ttk.Treeview(
            landmarks_frame,
            columns=("label", "x", "y", "confidence"),
            show="headings",
            height=14,
        )
        for column, width in (("label", 160), ("x", 70), ("y", 70), ("confidence", 90)):
            self.landmarks_tree.heading(column, text=column.title())
            self.landmarks_tree.column(column, width=width, anchor="center" if column != "label" else "w")
        self.landmarks_tree.grid(row=0, column=0, sticky="nsew")
        split.add(landmarks_frame, weight=1)

    def _build_camera_page(self) -> None:
        self.camera_page.columnconfigure(0, weight=1)
        self.camera_page.rowconfigure(1, weight=1)
        self.camera_status_var = tk.StringVar(value="Camera: waiting for frames")
        ttk.Label(self.camera_page, textvariable=self.camera_status_var).grid(row=0, column=0, sticky="w", pady=(0, 8))
        self.camera_label = ttk.Label(self.camera_page, anchor="center")
        self.camera_label.grid(row=1, column=0, sticky="nsew")
        self.camera_boxes_var = tk.StringVar(value="Boxes: none")
        ttk.Label(self.camera_page, textvariable=self.camera_boxes_var, wraplength=820).grid(row=2, column=0, sticky="ew", pady=(8, 0))

    def _build_valuemap_page(self) -> None:
        self.valuemap_page.columnconfigure(0, weight=1)
        self.valuemap_page.rowconfigure(1, weight=1)
        self.valuemap_status_var = tk.StringVar(value="ValueMap: waiting for occupancy and pose")
        ttk.Label(self.valuemap_page, textvariable=self.valuemap_status_var).grid(row=0, column=0, sticky="w", pady=(0, 8))
        self.valuemap_label = ttk.Label(self.valuemap_page, anchor="center")
        self.valuemap_label.grid(row=1, column=0, sticky="nsew")

    def _append_chat(self, role: str, message: str) -> None:
        self.chat_log.configure(state="normal")
        prefix = "You" if role == "user" else "System"
        self.chat_log.insert("end", f"[{prefix}] {message}\n")
        self.chat_log.see("end")
        self.chat_log.configure(state="disabled")

    def _drain_ui_events(self) -> None:
        while True:
            try:
                event_name, payload = self.ui_queue.get_nowait()
            except queue.Empty:
                break
            if event_name == "log":
                self._append_chat("system", payload)
            elif event_name == "status":
                self.connection_var.set(payload)
            elif event_name == "ready":
                self.connection_var.set(payload)
                self.ros_ready = True
                self.message_entry.focus_set()
            elif event_name == "error":
                self._append_chat("system", f"ERROR: {payload}")
                self.connection_var.set(payload)
        if not self.shutdown_requested:
            self.root.after(120, self._drain_ui_events)

    def _queue_log(self, message: str) -> None:
        self.ui_queue.put(("log", message))

    def _simulation_running(self) -> bool:
        try:
            import rosgraph

            master = rosgraph.Master("/vln_operator_gui_check")
            _publishers, _subscribers, services = master.getSystemState()
        except Exception:
            return False
        service_names = {name for name, _providers in services}
        return "/gazebo/get_world_properties" in service_names or "/vln/sim/reset_episode" in service_names

    def _start_simulation_process(self) -> None:
        log_path = ROOT / "logs" / "ros" / "vln_operator_gui_launch.log"
        if not log_path.parent.exists():
            log_path = Path("/tmp/vln_operator_gui_launch.log")
        self.launch_log_handle = log_path.open("a", encoding="utf-8")
        command = build_launch_command(ROOT, self.world_name, self.tasks_file)
        self.launch_process = subprocess.Popen(
            command,
            cwd=str(ROOT),
            stdout=self.launch_log_handle,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )
        self._queue_log(f"Started Gazebo/PX4 stack for world '{self.world_name}'. Log: {log_path}")

    def _wait_for_master(self, timeout_sec: float) -> bool:
        deadline = time.time() + timeout_sec
        while time.time() < deadline and not self.shutdown_requested:
            try:
                import rosgraph

                rosgraph.Master("/vln_operator_gui_wait").getPid()
                return True
            except Exception:
                time.sleep(0.5)
        return False

    def _boot_ros_stack(self) -> None:
        try:
            if self.autostart_sim:
                if self._simulation_running():
                    self._queue_log("Detected an existing Gazebo/VLN stack. Reusing it instead of starting another one.")
                else:
                    self.ui_queue.put(("status", f"World: {self.world_name} | Starting Gazebo and PX4..."))
                    self._start_simulation_process()

            if not self._wait_for_master(timeout_sec=120.0):
                self.ui_queue.put(("error", "ROS master did not come up within 120 seconds."))
                return

            import rospy
            from nav_msgs.msg import OccupancyGrid, Odometry
            from sensor_msgs.msg import Image as RosImage, LaserScan
            from std_msgs.msg import String
            from vln_msgs.msg import Mission, RuntimeState, SemanticLandmark, Trajectory
            from vln_msgs.srv import ResetEpisode

            rospy.init_node("vln_operator_gui", anonymous=True, disable_signals=True)
            self.rospy = rospy
            self.StringMsg = String
            self.reset_event_pub = rospy.Publisher("/sim/uav/reset_event", String, queue_size=10)
            self.reset_proxy = rospy.ServiceProxy("/vln/sim/reset_episode", ResetEpisode)

            rospy.Subscriber("/vln/mission/compiled", Mission, self._on_mission, queue_size=10)
            rospy.Subscriber("/vln/runtime/state", RuntimeState, self._on_runtime_state, queue_size=30)
            rospy.Subscriber("/vln/planning/trajectory", Trajectory, self._on_trajectory, queue_size=10)
            rospy.Subscriber("/vln/world/semantic_map", SemanticLandmark, self._on_landmark, queue_size=30)
            rospy.Subscriber("/vln/world/occupancy", OccupancyGrid, self._on_grid, queue_size=2)
            rospy.Subscriber("/sim/uav/odom", Odometry, self._on_odom, queue_size=30)
            rospy.Subscriber("/sim/uav/scan", LaserScan, self._on_scan, queue_size=5)
            rospy.Subscriber("/sim/uav/rgb/image_raw", RosImage, self._on_image, queue_size=1)
            rospy.Subscriber(DEFAULT_DEBUG_BOXES_TOPIC, String, self._on_debug_boxes, queue_size=5)

            self.ui_queue.put(("ready", f"World: {self.world_name} | ROS connected and GUI controls are live."))
            self._queue_log("Operator client connected. Use chat to publish an instruction, or run a named episode from the dropdown.")
        except Exception as exc:
            self.ui_queue.put(("error", f"Failed to initialize GUI ROS client: {exc}"))

    def _on_mission(self, msg) -> None:
        with self.state_lock:
            self.compiled_mission = msg

    def _on_runtime_state(self, msg) -> None:
        with self.state_lock:
            self.runtime_state = msg
        phase_key = f"{msg.mission_id}:{msg.phase}:{msg.active_step_index}:{msg.detail}"
        if phase_key != self.last_phase_logged:
            self.last_phase_logged = phase_key
            self._queue_log(f"Runtime {msg.phase} | step={msg.active_step_index} | {msg.detail}")

    def _on_trajectory(self, msg) -> None:
        with self.state_lock:
            self.trajectory_points = [(pose.pose.position.x, pose.pose.position.y) for pose in msg.waypoints]

    def _on_landmark(self, msg) -> None:
        with self.state_lock:
            self.semantic_landmarks[msg.label] = (msg.position.x, msg.position.y, msg.confidence)

    def _on_grid(self, msg) -> None:
        with self.state_lock:
            self.grid = GridSpec(
                width=int(msg.info.width),
                height=int(msg.info.height),
                resolution=float(msg.info.resolution),
                origin_x=float(msg.info.origin.position.x),
                origin_y=float(msg.info.origin.position.y),
                data=np.array(msg.data, dtype=np.int16),
            )

    def _on_odom(self, msg) -> None:
        q = msg.pose.pose.orientation
        yaw = math.atan2(2.0 * ((q.w * q.z) + (q.x * q.y)), 1.0 - 2.0 * ((q.y * q.y) + (q.z * q.z)))
        with self.state_lock:
            self.current_pose = PoseSnapshot(
                x=float(msg.pose.pose.position.x),
                y=float(msg.pose.pose.position.y),
                z=float(msg.pose.pose.position.z),
                yaw_rad=yaw,
            )

    def _on_scan(self, msg) -> None:
        with self.state_lock:
            self.scan_msg = msg

    def _on_image(self, msg) -> None:
        with self.state_lock:
            self.latest_image_msg = msg
            self.latest_image_seq += 1

    def _on_debug_boxes(self, msg) -> None:
        try:
            payload = json.loads(msg.data) if msg.data else {}
        except json.JSONDecodeError:
            payload = {}
        boxes = []
        for item in payload.get("boxes", []):
            try:
                boxes.append(
                    DebugBox(
                        label=str(item["label"]),
                        confidence=float(item.get("confidence", 0.0)),
                        xmin=float(item["xmin"]),
                        ymin=float(item["ymin"]),
                        xmax=float(item["xmax"]),
                        ymax=float(item["ymax"]),
                        source=str(item.get("source", "")),
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue
        with self.state_lock:
            self.debug_boxes = boxes

    def _active_labels(self) -> List[str]:
        mission = self.compiled_mission
        runtime_state = self.runtime_state
        if mission is None or not getattr(mission, "steps", None):
            return []
        step_index = int(getattr(runtime_state, "active_step_index", 0))
        if 0 <= step_index < len(mission.steps):
            label = str(mission.steps[step_index].target_label).strip()
            return [label] if label else []
        if step_index > 0 and step_index - 1 < len(mission.steps):
            label = str(mission.steps[step_index - 1].target_label).strip()
            return [label] if label else []
        return []

    def _handle_send(self) -> None:
        raw = self.message_var.get().strip()
        if not raw:
            return
        self.message_var.set("")
        self._append_chat("user", raw)
        command = parse_operator_command(raw)
        if command["kind"] == "noop":
            return
        if not self.ros_ready:
            self._append_chat("system", "ROS is still connecting. Please wait a moment and try again.")
            return
        if command["kind"] == "instruction":
            self._publish_instruction(command["instruction"])
            return
        if command["kind"] == "episode":
            self._run_episode(command["episode_name"])
            return
        if command["kind"] == "reset":
            self._publish_manual_reset(command.get("episode_name", "manual_reset"))

    def _publish_instruction(self, instruction: str) -> None:
        if self.reset_event_pub is None or self.StringMsg is None:
            self._append_chat("system", "Reset publisher is not ready yet.")
            return
        payload = {
            "episode_name": "manual_chat",
            "hard_reset": False,
            "instruction": instruction,
        }
        self.reset_event_pub.publish(self.StringMsg(data=json.dumps(payload, sort_keys=True)))
        self._append_chat("system", "Published manual instruction and soft-reset the runtime stack.")

    def _publish_manual_reset(self, episode_name: str = "manual_reset") -> None:
        if self.reset_event_pub is None or self.StringMsg is None:
            self._append_chat("system", "Reset publisher is not ready yet.")
            return
        payload = {
            "episode_name": episode_name or "manual_reset",
            "hard_reset": False,
        }
        self.reset_event_pub.publish(self.StringMsg(data=json.dumps(payload, sort_keys=True)))
        self._append_chat("system", f"Published reset event for '{payload['episode_name']}'.")

    def _run_selected_episode(self) -> None:
        episode_name = self.episode_var.get().strip()
        if not episode_name:
            self._append_chat("system", "No episode selected.")
            return
        self._run_episode(episode_name)

    def _run_episode(self, episode_name: str) -> None:
        if not self.ros_ready or self.reset_proxy is None:
            self._append_chat("system", "Episode reset service is not ready yet.")
            return

        def _worker() -> None:
            try:
                self.rospy.wait_for_service("/vln/sim/reset_episode", timeout=10.0)
                response = self.reset_proxy(episode_name, True)
                if response.accepted:
                    self.ui_queue.put(("log", f"Started episode '{episode_name}'."))
                else:
                    self.ui_queue.put(("error", f"Episode '{episode_name}' was rejected: {response.message}"))
            except Exception as exc:
                self.ui_queue.put(("error", f"Failed to start episode '{episode_name}': {exc}"))

        threading.Thread(target=_worker, daemon=True).start()

    def _refresh_status_page(self) -> None:
        with self.state_lock:
            runtime_state = self.runtime_state
            pose = self.current_pose
            mission = self.compiled_mission
            landmarks = dict(self.semantic_landmarks)
            waypoint_count = len(self.trajectory_points)

        mission_id = getattr(mission, "mission_id", "") or getattr(runtime_state, "mission_id", "") or "none"
        phase = getattr(runtime_state, "phase", "idle")
        detail = getattr(runtime_state, "detail", "waiting for runtime state")
        active_step = int(getattr(runtime_state, "active_step_index", 0))

        self.phase_var.set(f"Phase: {phase} | Step: {active_step}")
        self.pose_var.set(f"Pose: ({pose.x:.1f}, {pose.y:.1f}, {pose.z:.1f})")
        self.mission_var.set(f"Mission: {mission_id}")
        self.trajectory_var.set(f"Trajectory: {waypoint_count} waypoints")
        self.detail_var.set(f"Detail: {detail}")

        self.steps_tree.delete(*self.steps_tree.get_children())
        if mission is not None:
            for step in mission.steps:
                values = (step.step_index, step.action, step.target_label or "-", "yes" if step.terminal else "no")
                item_id = self.steps_tree.insert("", "end", values=values)
                if int(step.step_index) == active_step:
                    self.steps_tree.selection_set(item_id)

        self.landmarks_tree.delete(*self.landmarks_tree.get_children())
        for label, (x_coord, y_coord, confidence) in sorted(landmarks.items()):
            self.landmarks_tree.insert("", "end", values=(label, f"{x_coord:.1f}", f"{y_coord:.1f}", f"{confidence:.2f}"))

        if not self.shutdown_requested:
            self.root.after(250, self._refresh_status_page)

    def _refresh_camera_page(self) -> None:
        with self.state_lock:
            image_msg = self.latest_image_msg
            image_seq = self.latest_image_seq
            boxes = list(self.debug_boxes)
            active_labels = self._active_labels()

        if image_msg is None:
            self.camera_status_var.set("Camera: waiting for /sim/uav/rgb/image_raw")
        elif image_seq != self.rendered_image_seq:
            pil_image = image_message_to_pil(image_msg)
            if pil_image is not None:
                pil_image = pil_image.resize((720, 405), resample=Image.BILINEAR)
                overlay = draw_debug_boxes(pil_image, boxes, active_labels)
                from PIL import ImageTk

                self.camera_photo = ImageTk.PhotoImage(overlay)
                self.camera_label.configure(image=self.camera_photo)
                self.camera_status_var.set(
                    f"Camera: {image_msg.width}x{image_msg.height} | detections={len(boxes)} | overlay_fps≈6"
                )
                self.camera_boxes_var.set(
                    "Boxes: " + (", ".join(f"{box.label} ({box.confidence:.2f})" for box in boxes) if boxes else "none")
                )
                self.rendered_image_seq = image_seq

        if not self.shutdown_requested:
            self.root.after(160, self._refresh_camera_page)

    def _refresh_valuemap_page(self) -> None:
        with self.state_lock:
            grid = self.grid
            pose = self.current_pose
            waypoints = list(self.trajectory_points)
            landmarks = [(label, values[0], values[1], values[2]) for label, values in self.semantic_landmarks.items()]
            scan_msg = self.scan_msg
            active_labels = self._active_labels()

        scan_points = scan_to_world_points(scan_msg, pose)
        value_map = build_value_map(grid, pose, waypoints, landmarks, scan_points, active_labels)
        rendered = colorize_value_map(value_map, size=(720, 520))
        draw = ImageDraw.Draw(rendered)

        def world_to_pixel(x_coord: float, y_coord: float) -> Tuple[int, int]:
            grid_width = max(1.0, grid.width * max(0.2, grid.resolution))
            grid_height = max(1.0, grid.height * max(0.2, grid.resolution))
            px = int(((x_coord - grid.origin_x) / grid_width) * rendered.width)
            py = int(rendered.height - (((y_coord - grid.origin_y) / grid_height) * rendered.height))
            return px, py

        if len(waypoints) >= 2:
            draw.line([world_to_pixel(x, y) for x, y in waypoints], fill="#ffffff", width=3)
        for label, x_coord, y_coord, _confidence in landmarks:
            px, py = world_to_pixel(x_coord, y_coord)
            radius = 6 if label in set(active_labels) else 4
            color = "#9cff7a" if label in set(active_labels) else "#ffe28a"
            draw.ellipse((px - radius, py - radius, px + radius, py + radius), fill=color, outline="black")
            draw.text((px + 8, py - 8), label, fill="white", font=ImageFont.load_default())

        pose_px, pose_py = world_to_pixel(pose.x, pose.y)
        draw.ellipse((pose_px - 7, pose_py - 7, pose_px + 7, pose_py + 7), fill="#4cc9ff", outline="black")
        nose_x = pose_px + int(math.cos(pose.yaw_rad) * 14)
        nose_y = pose_py - int(math.sin(pose.yaw_rad) * 14)
        draw.line((pose_px, pose_py, nose_x, nose_y), fill="#4cc9ff", width=3)

        from PIL import ImageTk

        self.valuemap_photo = ImageTk.PhotoImage(rendered)
        self.valuemap_label.configure(image=self.valuemap_photo)
        self.valuemap_status_var.set(
            f"ValueMap: waypoints={len(waypoints)} landmarks={len(landmarks)} scan_hits={len(scan_points)} active={', '.join(active_labels) or 'none'}"
        )

        if not self.shutdown_requested:
            self.root.after(500, self._refresh_valuemap_page)

    def _poll_launch_process(self) -> None:
        if self.launch_process is not None and self.launch_process.poll() is not None:
            code = self.launch_process.returncode
            self.launch_process = None
            self.ui_queue.put(("error", f"Gazebo/PX4 launch process exited with code {code}."))
        if not self.shutdown_requested:
            self.root.after(1000, self._poll_launch_process)

    def _stop_launch_process(self) -> None:
        if self.launch_process is not None and self.launch_process.poll() is None:
            try:
                os.killpg(os.getpgid(self.launch_process.pid), signal.SIGTERM)
                self.launch_process.wait(timeout=10)
            except Exception:
                try:
                    os.killpg(os.getpgid(self.launch_process.pid), signal.SIGKILL)
                except Exception:
                    pass
        self.launch_process = None
        if self.launch_log_handle is not None:
            self.launch_log_handle.close()
            self.launch_log_handle = None

    def _on_close(self) -> None:
        self.shutdown_requested = True
        try:
            if self.rospy is not None:
                self.rospy.signal_shutdown("operator_gui_closed")
        except Exception:
            pass
        self._stop_launch_process()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    parser = argparse.ArgumentParser(description="Lightweight Tkinter operator GUI for the PX4 VLN stack.")
    parser.add_argument("--world-name", default=DEFAULT_WORLD_NAME)
    parser.add_argument("--tasks-file", default=str(DEFAULT_TASKS_FILE))
    parser.add_argument("--geometry", default="1440x860")
    parser.add_argument("--no-autostart-sim", action="store_true", help="Open the GUI without starting Gazebo/PX4.")
    args = parser.parse_args()

    gui = OperatorGui(
        world_name=str(args.world_name),
        tasks_file=Path(args.tasks_file).resolve(),
        autostart_sim=not bool(args.no_autostart_sim),
        geometry=str(args.geometry),
    )
    gui.run()


if __name__ == "__main__":
    main()
