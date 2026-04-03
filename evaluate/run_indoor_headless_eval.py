#!/usr/bin/env python3
"""Headless indoor VLN evaluation runner.

This script launches the indoor Gazebo world without GUI, loops over a task set,
resets the robot before every episode, records topic events and ROS logs, and
produces a markdown/json summary with basic failure analysis.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import rosgraph
import rospy
import tf.transformations as tfs
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
from geometry_msgs.msg import Quaternion, Twist
from magv_vln_msgs.msg import ArucoInfo, Detection2DArray, VehicleStatus
from nav_msgs.msg import Odometry
from rosgraph_msgs.msg import Log
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Bool, Int32, String


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORLD = ROOT / "gazebo_world" / "worlds" / "indoor_vln.world"
DEFAULT_TASKS = ROOT / "evaluate" / "tasks_indoor_vln.json"


def now_wall() -> float:
    return time.time()


def iso_ts(ts: Optional[float] = None) -> str:
    return datetime.fromtimestamp(ts or now_wall()).strftime("%Y-%m-%d %H:%M:%S")


def safe_json_dumps(data: Dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True)


def yaw_to_quat(yaw: float) -> Quaternion:
    quat = tfs.quaternion_from_euler(0.0, 0.0, yaw)
    return Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3])


def parse_diag_fields(text: str) -> Dict[str, str]:
    fields: Dict[str, str] = {}
    for key, value in re.findall(r"([A-Za-z0-9_]+)=([^\s|]+)", text or ""):
        fields[key] = value
    return fields


class JsonlWriter:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fp = self.path.open("a", encoding="utf-8")
        self._lock = threading.Lock()

    def write(self, data: Dict[str, Any]) -> None:
        with self._lock:
            self._fp.write(safe_json_dumps(data) + "\n")
            self._fp.flush()

    def close(self) -> None:
        with self._lock:
            self._fp.close()


@dataclass
class EpisodeResult:
    index: int
    name: str
    instruction: str
    spawn: Dict[str, Any]
    timeout_sec: float
    start_wall: float = 0.0
    end_wall: float = 0.0
    success: bool = False
    failure_reason: str = ""
    final_status_code: Optional[int] = None
    state_transitions: List[Dict[str, Any]] = field(default_factory=list)
    emergency_stop_events: List[Dict[str, Any]] = field(default_factory=list)
    emergency_stop_release_events: List[Dict[str, Any]] = field(default_factory=list)
    diagnostics: List[Dict[str, Any]] = field(default_factory=list)
    subtasks_messages: List[Dict[str, Any]] = field(default_factory=list)
    core_feedback: List[Dict[str, Any]] = field(default_factory=list)
    grounding_status: List[Dict[str, Any]] = field(default_factory=list)
    detections_frames: int = 0
    detections_by_label: Counter = field(default_factory=Counter)
    detections_total: int = 0
    aruco_frames: int = 0
    aruco_ids: Counter = field(default_factory=Counter)
    rosout_by_level: Counter = field(default_factory=Counter)
    rosout_by_node: Counter = field(default_factory=Counter)
    warnings: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    first_subtask_wall: Optional[float] = None
    first_detection_wall: Optional[float] = None
    first_navigation_wall: Optional[float] = None
    final_vehicle_status: Optional[Dict[str, Any]] = None
    notes: List[str] = field(default_factory=list)

    @property
    def duration_sec(self) -> float:
        if self.end_wall <= self.start_wall:
            return 0.0
        return self.end_wall - self.start_wall


class EvalCollector:
    def __init__(self, topic_writer: JsonlWriter, rosout_writer: JsonlWriter):
        self.topic_writer = topic_writer
        self.rosout_writer = rosout_writer
        self.lock = threading.Lock()

        self.last_seen: Dict[str, float] = defaultdict(float)
        self.vlm_ready = False
        self.episodes: List[EpisodeResult] = []
        self.active_episode: Optional[EpisodeResult] = None
        self._last_vehicle_state: Optional[int] = None
        self._last_episode_status_wall: float = 0.0
        self._last_high_rate_log_wall: Dict[str, float] = defaultdict(float)

    def start_episode(self, episode: EpisodeResult) -> None:
        with self.lock:
            self.active_episode = episode
            self.episodes.append(episode)
            self._last_vehicle_state = None
            self._last_episode_status_wall = 0.0

    def finish_episode(self) -> None:
        with self.lock:
            self.active_episode = None
            self._last_vehicle_state = None
            self._last_episode_status_wall = 0.0

    def _write_topic_event(self, topic: str, payload: Dict[str, Any]) -> None:
        event = {
            "wall_time": now_wall(),
            "wall_time_iso": iso_ts(),
            "topic": topic,
            "episode_index": self.active_episode.index if self.active_episode else None,
            "payload": payload,
        }
        self.topic_writer.write(event)

    def _touch(self, topic: str) -> None:
        self.last_seen[topic] = now_wall()

    def record_internal_event(self, label: str, payload: Dict[str, Any]) -> None:
        with self.lock:
            self._write_topic_event(label, payload)

    def on_vlm_status(self, msg: Bool) -> None:
        with self.lock:
            self.vlm_ready = bool(msg.data)
            self._touch("/VLM_Status")
            self._write_topic_event("/VLM_Status", {"data": bool(msg.data)})

    def on_odometry(self, msg: Odometry) -> None:
        with self.lock:
            self._touch("/magv/odometry/gt")
            wall = now_wall()
            if wall - self._last_high_rate_log_wall["/magv/odometry/gt"] < 0.5:
                return
            self._last_high_rate_log_wall["/magv/odometry/gt"] = wall
            pose = msg.pose.pose.position
            orient = msg.pose.pose.orientation
            yaw = tfs.euler_from_quaternion([orient.x, orient.y, orient.z, orient.w])[2]
            self._write_topic_event(
                "/magv/odometry/gt",
                {
                    "frame_id": msg.header.frame_id,
                    "x": round(pose.x, 3),
                    "y": round(pose.y, 3),
                    "z": round(pose.z, 3),
                    "yaw": round(yaw, 3),
                },
            )

    def on_pointcloud(self, msg: PointCloud2) -> None:
        with self.lock:
            self._touch("/magv/scan/3d")
            wall = now_wall()
            if wall - self._last_high_rate_log_wall["/magv/scan/3d"] < 0.5:
                return
            self._last_high_rate_log_wall["/magv/scan/3d"] = wall
            self._write_topic_event(
                "/magv/scan/3d",
                {"frame_id": msg.header.frame_id, "width": int(msg.width), "height": int(msg.height)},
            )

    def on_subtasks(self, msg: String) -> None:
        with self.lock:
            self._touch("/subtasks")
            payload: Dict[str, Any] = {"raw": msg.data}
            try:
                parsed = json.loads(msg.data)
                payload["count"] = len(parsed) if isinstance(parsed, list) else None
                payload["parsed"] = parsed
            except Exception:
                pass
            self._write_topic_event("/subtasks", payload)
            if self.active_episode:
                if self.active_episode.first_subtask_wall is None:
                    self.active_episode.first_subtask_wall = now_wall()
                self.active_episode.subtasks_messages.append(payload)

    def on_core_feedback(self, msg: String) -> None:
        with self.lock:
            self._touch("/core_feedback")
            payload: Dict[str, Any] = {"raw": msg.data}
            try:
                payload["parsed"] = json.loads(msg.data)
            except Exception:
                pass
            self._write_topic_event("/core_feedback", payload)
            if self.active_episode:
                self.active_episode.core_feedback.append(payload)

    def on_vlm_error_log(self, msg: String) -> None:
        with self.lock:
            self._touch("/vlm_error_log")
            payload = {"raw": msg.data}
            self._write_topic_event("/vlm_error_log", payload)

    def on_grounding_status(self, msg: String) -> None:
        with self.lock:
            self._touch("/grounding_dino/status")
            payload: Dict[str, Any] = {"raw": msg.data}
            try:
                payload["parsed"] = json.loads(msg.data)
            except Exception:
                pass
            self._write_topic_event("/grounding_dino/status", payload)
            if self.active_episode:
                self.active_episode.grounding_status.append(payload)

    def on_detections(self, msg: Detection2DArray) -> None:
        with self.lock:
            self._touch("/grounding_dino/detections")
            labels = [det.label for det in msg.detections]
            payload = {"count": len(msg.detections), "labels": labels}
            self._write_topic_event("/grounding_dino/detections", payload)
            if self.active_episode:
                if self.active_episode.first_detection_wall is None and msg.detections:
                    self.active_episode.first_detection_wall = now_wall()
                self.active_episode.detections_frames += 1
                self.active_episode.detections_total += len(msg.detections)
                self.active_episode.detections_by_label.update(labels)

    def on_aruco(self, msg: ArucoInfo) -> None:
        with self.lock:
            self._touch("/aruco_info")
            ids = [int(marker.id) for marker in msg.markers]
            payload = {"count": len(msg.markers), "ids": ids}
            self._write_topic_event("/aruco_info", payload)
            if self.active_episode:
                self.active_episode.aruco_frames += 1
                self.active_episode.aruco_ids.update(ids)

    def on_status_code(self, msg: Int32) -> None:
        with self.lock:
            self._touch("/status")
            wall = now_wall()
            payload = {"data": int(msg.data)}
            self._write_topic_event("/status", payload)
            if self.active_episode and wall >= self.active_episode.start_wall:
                self.active_episode.final_status_code = int(msg.data)
                self._last_episode_status_wall = wall
                if int(msg.data) == 0 and not self.active_episode.success:
                    self.active_episode.success = True
                    self.active_episode.end_wall = wall

    def on_vehicle_status(self, msg: VehicleStatus) -> None:
        with self.lock:
            self._touch("/vln_status")
            payload = {
                "state": int(msg.state),
                "state_description": msg.state_description,
                "task_completed": bool(msg.task_completed),
                "current_subtask_index": int(msg.current_subtask_index),
                "total_subtasks": int(msg.total_subtasks),
                "diagnostic_info": msg.diagnostic_info,
                "target_position": {
                    "x": round(msg.target_position.x, 3),
                    "y": round(msg.target_position.y, 3),
                    "z": round(msg.target_position.z, 3),
                },
            }
            self._write_topic_event("/vln_status", payload)

            if not self.active_episode:
                self._last_vehicle_state = int(msg.state)
                return

            wall = now_wall()
            diag_fields = parse_diag_fields(msg.diagnostic_info)
            self.active_episode.final_vehicle_status = payload
            self.active_episode.diagnostics.append(
                {"wall_time": wall, "diagnostic_info": msg.diagnostic_info, "fields": diag_fields}
            )
            if msg.state == VehicleStatus.STATE_NAVIGATION and self.active_episode.first_navigation_wall is None:
                self.active_episode.first_navigation_wall = wall
            if self._last_vehicle_state != int(msg.state):
                transition = {
                    "wall_time": wall,
                    "from": self._last_vehicle_state,
                    "to": int(msg.state),
                    "to_name": msg.state_description,
                    "diagnostic_info": msg.diagnostic_info,
                }
                self.active_episode.state_transitions.append(transition)
                if int(msg.state) == VehicleStatus.STATE_EMERGENCY_STOP:
                    self.active_episode.emergency_stop_events.append(
                        {"wall_time": wall, "diagnostic_info": msg.diagnostic_info, "fields": diag_fields}
                    )
                if self._last_vehicle_state == VehicleStatus.STATE_EMERGENCY_STOP and int(msg.state) != VehicleStatus.STATE_EMERGENCY_STOP:
                    self.active_episode.emergency_stop_release_events.append(
                        {"wall_time": wall, "to": int(msg.state), "diagnostic_info": msg.diagnostic_info, "fields": diag_fields}
                    )
            self._last_vehicle_state = int(msg.state)

    def on_rosout(self, msg: Log) -> None:
        with self.lock:
            level_map = {
                Log.DEBUG: "DEBUG",
                Log.INFO: "INFO",
                Log.WARN: "WARN",
                Log.ERROR: "ERROR",
                Log.FATAL: "FATAL",
            }
            payload = {
                "name": msg.name,
                "level": level_map.get(msg.level, str(msg.level)),
                "msg": msg.msg,
                "file": msg.file,
                "function": msg.function,
                "line": int(msg.line),
            }
            self.rosout_writer.write(
                {
                    "wall_time": now_wall(),
                    "wall_time_iso": iso_ts(),
                    "episode_index": self.active_episode.index if self.active_episode else None,
                    "payload": payload,
                }
            )
            if self.active_episode:
                self.active_episode.rosout_by_level[payload["level"]] += 1
                self.active_episode.rosout_by_node[msg.name] += 1
                if payload["level"] in ("WARN", "ERROR", "FATAL"):
                    target = self.active_episode.errors if payload["level"] in ("ERROR", "FATAL") else self.active_episode.warnings
                    target.append(payload)


class HeadlessEval:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.run_dir = self._make_run_dir()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "episodes").mkdir(exist_ok=True)
        self.topic_writer = JsonlWriter(self.run_dir / "topic_events.jsonl")
        self.rosout_writer = JsonlWriter(self.run_dir / "rosout.jsonl")
        self.collector = EvalCollector(self.topic_writer, self.rosout_writer)
        self.launch_proc: Optional[subprocess.Popen] = None
        self.cmd_pub = None
        self.ins_pub = None
        self.set_model_state = None

    def _make_run_dir(self) -> Path:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = f"_{self.args.run_name}" if self.args.run_name else ""
        return ROOT / "evaluate" / "runs" / f"{stamp}{suffix}"

    def _launch_env(self) -> Dict[str, str]:
        env = os.environ.copy()
        env["HOME"] = str(self.run_dir / "home")
        env["ROS_HOME"] = str(self.run_dir / "ros_home")
        env["ROS_LOG_DIR"] = str(self.run_dir / "ros_logs")
        env["GAZEBO_LOG_PATH"] = str(self.run_dir / "gazebo_logs")
        env["GAZEBO_MODEL_DATABASE_URI"] = ""
        extra_pythonpath = str(self.args.extra_pythonpath).strip()
        if extra_pythonpath and Path(extra_pythonpath).exists():
            current = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = extra_pythonpath if not current else f"{extra_pythonpath}:{current}"
        for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "all_proxy"):
            env.pop(key, None)
        Path(env["HOME"]).mkdir(parents=True, exist_ok=True)
        Path(env["ROS_HOME"]).mkdir(parents=True, exist_ok=True)
        Path(env["ROS_LOG_DIR"]).mkdir(parents=True, exist_ok=True)
        Path(env["GAZEBO_LOG_PATH"]).mkdir(parents=True, exist_ok=True)
        return env

    def start_launch(self) -> None:
        cmd = [
            "roslaunch",
            "vln_mock",
            "vln_sim.launch",
            f"world_file:={self.args.world_file}",
            f"robot_profile:={self.args.robot_profile}",
            f"gui:={'true' if self.args.gui else 'false'}",
            f"headless:={'true' if self.args.headless else 'false'}",
            "launch_rviz:=false",
            "launch_operator_gui:=false",
            "paused:=false",
            "verbose:=false",
            "launch_world:=true",
            "spawn_robot:=true",
            f"vlm_backend:={self.args.vlm_backend}",
            f"vlm_model_path:={self.args.vlm_model_path}",
            f"grounding_backend:={self.args.grounding_backend}",
            f"grounding_model_path:={self.args.grounding_model_path}",
        ]
        console_path = self.run_dir / "launch_console.log"
        console_fp = console_path.open("w", encoding="utf-8")
        self.launch_proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=console_fp,
            stderr=subprocess.STDOUT,
            text=True,
            env=self._launch_env(),
            preexec_fn=os.setsid,
        )

    def stop_launch(self) -> None:
        if not self.launch_proc:
            return
        if self.launch_proc.poll() is None:
            try:
                os.killpg(self.launch_proc.pid, signal.SIGTERM)
                self.launch_proc.wait(timeout=20)
            except Exception:
                try:
                    os.killpg(self.launch_proc.pid, signal.SIGKILL)
                except Exception:
                    pass
        self.launch_proc = None

    def wait_for_master(self, timeout_sec: float) -> None:
        master = rosgraph.Master("/vln_eval")
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            if self.launch_proc and self.launch_proc.poll() is not None:
                raise RuntimeError("roslaunch exited early; check launch_console.log")
            try:
                master.getPid()
                return
            except Exception:
                time.sleep(1.0)
        raise TimeoutError("ROS master did not become ready in time")

    def init_ros(self) -> None:
        rospy.init_node("vln_indoor_eval", anonymous=False, disable_signals=True)
        self.cmd_pub = rospy.Publisher("/magv/omni_drive_controller/cmd_vel", Twist, queue_size=10)
        self.ins_pub = rospy.Publisher("/instruction", String, queue_size=10)

        rospy.Subscriber("/VLM_Status", Bool, self.collector.on_vlm_status, queue_size=10)
        rospy.Subscriber("/vln_status", VehicleStatus, self.collector.on_vehicle_status, queue_size=20)
        rospy.Subscriber("/status", Int32, self.collector.on_status_code, queue_size=10)
        rospy.Subscriber("/subtasks", String, self.collector.on_subtasks, queue_size=10)
        rospy.Subscriber("/core_feedback", String, self.collector.on_core_feedback, queue_size=20)
        rospy.Subscriber("/vlm_error_log", String, self.collector.on_vlm_error_log, queue_size=50)
        rospy.Subscriber("/grounding_dino/status", String, self.collector.on_grounding_status, queue_size=20)
        rospy.Subscriber("/grounding_dino/detections", Detection2DArray, self.collector.on_detections, queue_size=20)
        rospy.Subscriber("/aruco_info", ArucoInfo, self.collector.on_aruco, queue_size=20)
        rospy.Subscriber("/magv/odometry/gt", Odometry, self.collector.on_odometry, queue_size=20)
        rospy.Subscriber("/magv/scan/3d", PointCloud2, self.collector.on_pointcloud, queue_size=5)
        rospy.Subscriber("/rosout_agg", Log, self.collector.on_rosout, queue_size=200)

        rospy.wait_for_service("/gazebo/set_model_state", timeout=self.args.startup_timeout_sec)
        self.set_model_state = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)

    def wait_for_topic(self, topic: str, timeout_sec: float) -> None:
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline and not rospy.is_shutdown():
            if self.collector.last_seen.get(topic, 0.0) > 0.0:
                return
            if self.launch_proc and self.launch_proc.poll() is not None:
                raise RuntimeError(f"roslaunch exited while waiting for {topic}")
            time.sleep(0.2)
        raise TimeoutError(f"Timed out waiting for topic {topic}")

    def wait_for_startup(self) -> Dict[str, Any]:
        started = now_wall()
        self.wait_for_topic("/magv/odometry/gt", self.args.startup_timeout_sec)
        self.wait_for_topic("/magv/scan/3d", self.args.startup_timeout_sec)
        self.wait_for_topic("/vln_status", self.args.startup_timeout_sec)

        vlm_ready_at = None
        if self.args.wait_vlm_ready:
            deadline = time.monotonic() + self.args.vlm_ready_timeout_sec
            while time.monotonic() < deadline and not rospy.is_shutdown():
                if self.collector.vlm_ready:
                    vlm_ready_at = now_wall()
                    break
                time.sleep(0.5)
        return {
            "startup_begin_wall": started,
            "startup_ready_wall": now_wall(),
            "vlm_ready_wall": vlm_ready_at,
            "vlm_ready": bool(self.collector.vlm_ready),
        }

    def publish_zero_cmd(self, repeats: int = 6, sleep_sec: float = 0.1) -> None:
        zero = Twist()
        for _ in range(repeats):
            self.cmd_pub.publish(zero)
            time.sleep(sleep_sec)

    def reset_pose(self, model_name: str, spawn: Dict[str, Any]) -> None:
        self.collector.record_internal_event(
            "/evaluate/reset_pose",
            {"model_name": model_name, "spawn": spawn},
        )
        state = ModelState()
        state.model_name = model_name
        state.pose.position.x = float(spawn["x"])
        state.pose.position.y = float(spawn["y"])
        state.pose.position.z = float(spawn.get("z", self.args.default_spawn_z))
        state.pose.orientation = yaw_to_quat(float(spawn.get("yaw", 0.0)))
        state.reference_frame = "world"
        state.twist = Twist()
        for _ in range(3):
            response = self.set_model_state(state)
            if not response.success:
                raise RuntimeError(f"set_model_state failed: {response.status_message}")
            time.sleep(0.2)

    def run_episode(self, task: Dict[str, Any], episode_index: int) -> EpisodeResult:
        timeout_sec = float(task.get("timeout_sec", self.args.default_timeout_sec))
        episode = EpisodeResult(
            index=episode_index,
            name=task["name"],
            instruction=task["instruction"],
            spawn=task["spawn"],
            timeout_sec=timeout_sec,
        )
        self.collector.start_episode(episode)
        episode.start_wall = now_wall()

        self.publish_zero_cmd()
        self.reset_pose(self.args.model_name, task["spawn"])
        time.sleep(self.args.reset_settle_sec)
        self.publish_zero_cmd(repeats=3, sleep_sec=0.05)

        self.collector.record_internal_event(
            "/evaluate/instruction_publish",
            {"instruction": task["instruction"]},
        )
        for _ in range(2):
            self.ins_pub.publish(String(data=task["instruction"]))
            time.sleep(0.15)

        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline and not rospy.is_shutdown():
            if self.launch_proc and self.launch_proc.poll() is not None:
                episode.failure_reason = "roslaunch exited"
                break
            if episode.success:
                break
            time.sleep(0.2)

        if not episode.end_wall:
            episode.end_wall = now_wall()
        self.publish_zero_cmd(repeats=4, sleep_sec=0.05)

        if not episode.success and not episode.failure_reason:
            episode.failure_reason = "timeout"

        self._postprocess_episode(episode)
        self.collector.finish_episode()
        self._write_episode_summary(episode)
        return episode

    def _postprocess_episode(self, episode: EpisodeResult) -> None:
        if episode.first_subtask_wall is None:
            episode.notes.append("No /subtasks message arrived after instruction publish.")
        if episode.first_navigation_wall is None:
            episode.notes.append("Vehicle never reached NAVIGATION state during the episode.")
        if episode.emergency_stop_events:
            reasons = Counter()
            for event in episode.emergency_stop_events:
                reason = event["fields"].get("es_reason", "unknown")
                reasons[reason] += 1
            common_reason, count = reasons.most_common(1)[0]
            episode.notes.append(
                f"Emergency stop entered {len(episode.emergency_stop_events)} times; dominant reason={common_reason} ({count}x)."
            )
        if episode.detections_total == 0:
            episode.notes.append("No GroundingDINO detections were observed during the episode.")
        if episode.success and episode.emergency_stop_events:
            episode.notes.append("Task completed, but success path still crossed emergency-stop states.")

    def _write_episode_summary(self, episode: EpisodeResult) -> None:
        path = self.run_dir / "episodes" / f"{episode.index:02d}_{episode.name}.json"
        data = {
            "index": episode.index,
            "name": episode.name,
            "instruction": episode.instruction,
            "spawn": episode.spawn,
            "timeout_sec": episode.timeout_sec,
            "success": episode.success,
            "failure_reason": episode.failure_reason,
            "duration_sec": round(episode.duration_sec, 3),
            "final_status_code": episode.final_status_code,
            "state_transitions": episode.state_transitions,
            "emergency_stop_events": episode.emergency_stop_events,
            "emergency_stop_release_events": episode.emergency_stop_release_events,
            "detections_frames": episode.detections_frames,
            "detections_total": episode.detections_total,
            "detections_by_label": dict(episode.detections_by_label),
            "aruco_frames": episode.aruco_frames,
            "aruco_ids": dict(episode.aruco_ids),
            "rosout_by_level": dict(episode.rosout_by_level),
            "rosout_by_node": dict(episode.rosout_by_node),
            "notes": episode.notes,
            "final_vehicle_status": episode.final_vehicle_status,
        }
        path.write_text(safe_json_dumps(data) + "\n", encoding="utf-8")

    def generate_summary(self, startup_meta: Dict[str, Any]) -> Dict[str, Any]:
        episodes = self.collector.episodes
        success_count = sum(1 for ep in episodes if ep.success)
        total_es = sum(len(ep.emergency_stop_events) for ep in episodes)
        dominant_warn_nodes = Counter()
        dominant_error_nodes = Counter()
        for ep in episodes:
            for warning in ep.warnings:
                dominant_warn_nodes[warning["name"]] += 1
            for error in ep.errors:
                dominant_error_nodes[error["name"]] += 1

        findings: List[Dict[str, Any]] = []

        if total_es > 0:
            tf_related = 0
            no_obs_related = 0
            for ep in episodes:
                for event in ep.emergency_stop_events:
                    reason = event["fields"].get("es_reason", "")
                    if reason == "tf_unavailable":
                        tf_related += 1
                    if event["fields"].get("min_forward_x", "") in ("infm", "inf", ""):
                        no_obs_related += 1
            if tf_related:
                findings.append(
                    {
                        "severity": "high",
                        "title": "Emergency stop still enters on TF failures",
                        "detail": f"{tf_related} emergency-stop entries carried es_reason=tf_unavailable. This points to TF readiness or timestamp alignment remaining brittle.",
                    }
                )
            findings.append(
                {
                    "severity": "high" if total_es >= len(episodes) else "medium",
                    "title": "Emergency-stop corridor remains a top tuning risk",
                    "detail": f"Observed {total_es} emergency-stop entries across {len(episodes)} episodes. Because the status manager keys off current cmd_vel rather than strictly NAVIGATION state, reset/settle phases can still contribute false-positive safety stops.",
                }
            )
            if no_obs_related:
                findings.append(
                    {
                        "severity": "medium",
                        "title": "Some emergency stops were reported without a clear forward obstacle",
                        "detail": f"{no_obs_related} emergency-stop samples had no finite min_forward_x in diagnostics. This is a strong hint that corridor filtering, pointcloud frame assumptions, or sparse speckle handling still need tightening.",
                    }
                )

        timeout_eps = [ep for ep in episodes if not ep.success]
        if timeout_eps:
            no_subtasks = sum(1 for ep in timeout_eps if not ep.subtasks_messages)
            no_detection = sum(1 for ep in timeout_eps if ep.detections_total == 0)
            findings.append(
                {
                    "severity": "high" if success_count == 0 else "medium",
                    "title": "Not all tasks completed inside the evaluation window",
                    "detail": f"{len(timeout_eps)} of {len(episodes)} episodes timed out or exited unsuccessfully. {no_subtasks} had no parsed subtasks, and {no_detection} had no grounding detections.",
                }
            )
        if dominant_warn_nodes:
            node, count = dominant_warn_nodes.most_common(1)[0]
            findings.append(
                {
                    "severity": "medium",
                    "title": "Warnings concentrated on one node",
                    "detail": f"Most warnings came from {node} ({count} messages). Inspect this node's ROS log first.",
                }
            )
        if dominant_error_nodes:
            node, count = dominant_error_nodes.most_common(1)[0]
            findings.append(
                {
                    "severity": "medium",
                    "title": "Errors concentrated on one node",
                    "detail": f"Most errors came from {node} ({count} messages).",
                }
            )

        improvements = [
            "Gate emergency-stop triggering by explicit NAVIGATION state, not only by translational cmd_vel magnitude.",
            "Add a short post-reset and post-goal arming grace window so the safety corridor ignores transient pointcloud/TF noise immediately after teleport or state handoff.",
            "Expose emergency-stop counters and the last trigger source as a dedicated diagnostic topic or structured status field instead of only packing them into diagnostic_info text.",
            "Add an evaluation mode in the launch stack that disables instruction_latch_bridge or re-latches every new instruction, so repeated benchmark episodes are less stateful.",
            "Persist per-episode planner metrics such as first goal publish time, path-point count, and controller saturation, because current logs are stronger on perception/status than on control health.",
        ]

        return {
            "generated_at": iso_ts(),
            "run_dir": str(self.run_dir),
            "startup": startup_meta,
            "episodes": [
                {
                    "index": ep.index,
                    "name": ep.name,
                    "instruction": ep.instruction,
                    "success": ep.success,
                    "failure_reason": ep.failure_reason,
                    "duration_sec": round(ep.duration_sec, 3),
                    "emergency_stop_count": len(ep.emergency_stop_events),
                    "detections_total": ep.detections_total,
                    "aruco_frames": ep.aruco_frames,
                    "notes": ep.notes,
                }
                for ep in episodes
            ],
            "success_count": success_count,
            "episode_count": len(episodes),
            "findings": findings,
            "recommended_improvements": improvements,
        }

    def write_summary_files(self, summary: Dict[str, Any]) -> None:
        (self.run_dir / "summary.json").write_text(safe_json_dumps(summary) + "\n", encoding="utf-8")

        lines = [
            "# VLN Indoor Headless Evaluation",
            "",
            f"- Generated: {summary['generated_at']}",
            f"- Run dir: `{summary['run_dir']}`",
            f"- Success: {summary['success_count']} / {summary['episode_count']}",
            f"- VLM ready at startup: `{summary['startup'].get('vlm_ready')}`",
            "",
            "## Episodes",
            "",
            "| # | Name | Success | Duration(s) | E-stop count | Detections | Notes |",
            "| --- | --- | --- | ---: | ---: | ---: | --- |",
        ]
        for ep in summary["episodes"]:
            notes = "; ".join(ep["notes"]) if ep["notes"] else "-"
            lines.append(
                f"| {ep['index']} | {ep['name']} | {ep['success']} | {ep['duration_sec']:.1f} | "
                f"{ep['emergency_stop_count']} | {ep['detections_total']} | {notes} |"
            )

        lines.extend(["", "## Findings", ""])
        if summary["findings"]:
            for finding in summary["findings"]:
                lines.append(f"- [{finding['severity']}] {finding['title']}: {finding['detail']}")
        else:
            lines.append("- No major issues were auto-detected in this run.")

        lines.extend(["", "## Recommended Improvements", ""])
        for item in summary["recommended_improvements"]:
            lines.append(f"- {item}")

        (self.run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def close(self) -> None:
        self.topic_writer.close()
        self.rosout_writer.close()


def load_tasks(path: Path, repeat: int) -> List[Dict[str, Any]]:
    tasks = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("Task file must contain a non-empty JSON list")
    expanded: List[Dict[str, Any]] = []
    for cycle in range(repeat):
        for task in tasks:
            item = dict(task)
            if repeat > 1:
                item["name"] = f"{task['name']}_cycle{cycle + 1}"
            expanded.append(item)
    return expanded


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run headless indoor VLN evaluation episodes.")
    parser.add_argument("--world-file", default=str(DEFAULT_WORLD))
    parser.add_argument("--tasks-file", default=str(DEFAULT_TASKS))
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--max-episodes", type=int, default=0, help="If > 0, stop after this many expanded episodes.")
    parser.add_argument("--robot-profile", default="rover", choices=["rover", "uav"])
    parser.add_argument("--model-name", default="")
    parser.add_argument("--default-timeout-sec", type=float, default=90.0)
    parser.add_argument("--startup-timeout-sec", type=float, default=180.0)
    parser.add_argument("--vlm-ready-timeout-sec", type=float, default=180.0)
    parser.add_argument("--reset-settle-sec", type=float, default=2.5)
    parser.add_argument("--wait-vlm-ready", action="store_true", default=True)
    parser.add_argument("--no-wait-vlm-ready", action="store_false", dest="wait_vlm_ready")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--gui", action="store_true", default=False)
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--vlm-backend", default="qwen")
    parser.add_argument("--grounding-backend", default="grounding_dino")
    parser.add_argument("--vlm-model-path", default=os.environ.get("VLN_QWEN_MODEL_PATH", "/var/tmp/vln_models/Qwen--Qwen2-VL-2B-Instruct"))
    parser.add_argument("--grounding-model-path", default=os.environ.get("VLN_GROUNDING_MODEL_PATH", "/var/tmp/vln_models/grounding-dino-base"))
    parser.add_argument(
        "--extra-pythonpath",
        default=str(ROOT / "venv310" / "lib" / "python3.8" / "site-packages"),
        help="Optional extra site-packages path injected into the launched ROS environment.",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    args.world_file = str(Path(args.world_file).resolve())
    args.tasks_file = str(Path(args.tasks_file).resolve())
    args.model_name = args.model_name or ("vln_uav" if args.robot_profile == "uav" else "vln_rover")
    args.default_spawn_z = 1.20 if args.robot_profile == "uav" else 0.14

    tasks = load_tasks(Path(args.tasks_file), args.repeat)
    if args.max_episodes > 0:
        tasks = tasks[: args.max_episodes]
    runner = HeadlessEval(args)
    startup_meta: Dict[str, Any] = {}

    try:
        runner.start_launch()
        runner.wait_for_master(args.startup_timeout_sec)
        runner.init_ros()
        startup_meta = runner.wait_for_startup()

        for idx, task in enumerate(tasks, start=1):
            if runner.launch_proc and runner.launch_proc.poll() is not None:
                raise RuntimeError("roslaunch exited before all episodes finished")
            runner.run_episode(task, idx)

        summary = runner.generate_summary(startup_meta)
        runner.write_summary_files(summary)

        latest = ROOT / "evaluate" / "latest"
        if latest.is_symlink() or latest.is_file():
            latest.unlink()
        elif latest.is_dir():
            shutil.rmtree(latest)
        latest.symlink_to(runner.run_dir.name)
        print(f"[evaluate] completed; summary: {runner.run_dir / 'summary.md'}")
        return 0
    except KeyboardInterrupt:
        partial = runner.generate_summary(startup_meta) if runner.collector.episodes else {
            "generated_at": iso_ts(),
            "run_dir": str(runner.run_dir),
            "startup": startup_meta,
            "episodes": [],
            "success_count": 0,
            "episode_count": 0,
            "findings": [],
            "recommended_improvements": [],
        }
        partial["interrupted"] = True
        runner.write_summary_files(partial)
        print(f"[evaluate] interrupted; partial summary: {runner.run_dir / 'summary.md'}", file=sys.stderr)
        return 130
    except Exception as exc:
        if runner.collector.episodes:
            partial = runner.generate_summary(startup_meta)
            partial["error"] = str(exc)
            runner.write_summary_files(partial)
        failure = {
            "generated_at": iso_ts(),
            "error": str(exc),
            "run_dir": str(runner.run_dir),
            "startup": startup_meta,
        }
        (runner.run_dir / "summary.json").write_text(safe_json_dumps(failure) + "\n", encoding="utf-8")
        (runner.run_dir / "summary.md").write_text(
            "# VLN Indoor Headless Evaluation\n\n"
            f"- Generated: {failure['generated_at']}\n"
            f"- Run dir: `{failure['run_dir']}`\n"
            f"- Error: `{failure['error']}`\n",
            encoding="utf-8",
        )
        print(f"[evaluate] failed: {exc}", file=sys.stderr)
        return 1
    finally:
        try:
            runner.stop_launch()
        finally:
            runner.close()


if __name__ == "__main__":
    raise SystemExit(main())
