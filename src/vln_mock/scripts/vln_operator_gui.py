#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import queue
import time
from datetime import datetime

import rospy
from geometry_msgs.msg import Point
from magv_vln_msgs.msg import ArucoInfo, Detection2DArray, VehicleStatus
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool, Int32, String as StringMsg

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, scrolledtext, ttk
except Exception as exc:
    raise RuntimeError("tkinter is required for vln_operator_gui") from exc

try:
    import cv2
    import numpy as np
    from PIL import Image as PILImage
    from PIL import ImageTk
except Exception:
    cv2 = None
    np = None
    PILImage = None
    ImageTk = None


class VlnOperatorGui:
    def __init__(self):
        rospy.init_node("vln_operator_gui", anonymous=False, disable_signals=True)

        self.instruction_topic = rospy.get_param("~instruction_topic", "/instruction")
        self.vln_status_topic = rospy.get_param("~vln_status_topic", "/vln_status")
        self.vlm_status_topic = rospy.get_param("~vlm_status_topic", "/VLM_Status")
        self.subtasks_topic = rospy.get_param("~subtasks_topic", "/subtasks")
        self.vlm_error_log_topic = rospy.get_param("~vlm_error_log_topic", "/vlm_error_log")
        self.grounding_status_topic = rospy.get_param("~grounding_status_topic", "/grounding_dino/status")
        self.grounding_detections_topic = rospy.get_param("~grounding_detections_topic", "/grounding_dino/detections")
        self.aruco_info_topic = rospy.get_param("~aruco_info_topic", "/aruco_info")
        self.final_status_topic = rospy.get_param("~final_status_topic", "/status")
        self.preview_image_topic = rospy.get_param("~preview_image_topic", "/magv/camera/image_compressed/compressed")
        self.preview_max_width = int(rospy.get_param("~preview_max_width", 640))
        self.preview_max_height = int(rospy.get_param("~preview_max_height", 360))

        self.topic_last_seen = {
            self.vln_status_topic: 0.0,
            self.vlm_status_topic: 0.0,
            self.subtasks_topic: 0.0,
            self.vlm_error_log_topic: 0.0,
            self.grounding_status_topic: 0.0,
            self.grounding_detections_topic: 0.0,
            self.aruco_info_topic: 0.0,
            self.final_status_topic: 0.0,
            self.preview_image_topic: 0.0,
        }

        self.event_queue = queue.Queue()
        self.command_history = []
        self.last_instruction = ""
        self.last_logged_state = None
        self.last_grounding_event = None
        self.last_subtasks_json = ""
        self.last_detection_summary = ""
        self.last_aruco_summary = ""
        self.latest_transcript_lines = []
        self.latest_preview_rgb = None
        self.preview_photo = None
        self.preview_frame_counter = 0
        self.latest_detections = []
        self.preview_enabled = cv2 is not None and np is not None and PILImage is not None and ImageTk is not None
        self.preview_resample = None
        if self.preview_enabled:
            self.preview_resample = getattr(getattr(PILImage, "Resampling", PILImage), "LANCZOS", None)

        self.root = tk.Tk()
        self.root.title("VLN Operator Console")
        self.root.geometry("1540x960")
        self.root.minsize(1200, 760)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.state_var = tk.StringVar(value="IDLE")
        self.goal_var = tk.StringVar(value="No")
        self.motion_var = tk.StringVar(value="Stopped")
        self.sensors_var = tk.StringVar(value="Unknown")
        self.nav_var = tk.StringVar(value="Unknown")
        self.vlm_ready_var = tk.StringVar(value="Unknown")
        self.final_status_var = tk.StringVar(value="Pending")
        self.subtask_progress_var = tk.StringVar(value="0 / 0")
        self.position_var = tk.StringVar(value="(n/a)")
        self.target_var = tk.StringVar(value="(n/a)")
        self.diagnostic_var = tk.StringVar(value="Waiting for status...")
        self.grounding_var = tk.StringVar(value="No status yet")
        self.preview_var = tk.StringVar(value="Camera preview waiting...")

        self._build_ui()
        self._setup_ros()
        self._append_chat("system", "GUI client ready. Enter a natural-language instruction to begin.")
        if not self.preview_enabled:
            self._append_chat("warn", "Camera preview dependencies are unavailable. Install pillow, numpy and opencv-python if you want embedded video preview.")
        self.root.after(120, self._drain_events)
        self.root.after(500, self._refresh_topic_health)
        rospy.on_shutdown(self._on_ros_shutdown)

    def _build_ui(self):
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        header = ttk.Frame(self.root, padding=10)
        header.grid(row=0, column=0, sticky="nsew")
        for idx in range(8):
            header.columnconfigure(idx, weight=1)

        self._build_header_card(header, 0, "VLN State", self.state_var)
        self._build_header_card(header, 1, "Goal", self.goal_var)
        self._build_header_card(header, 2, "Motion", self.motion_var)
        self._build_header_card(header, 3, "Sensors", self.sensors_var)
        self._build_header_card(header, 4, "Navigation", self.nav_var)
        self._build_header_card(header, 5, "VLM Ready", self.vlm_ready_var)
        self._build_header_card(header, 6, "Final", self.final_status_var)
        self._build_header_card(header, 7, "Subtasks", self.subtask_progress_var)

        body = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        body.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))

        left = ttk.Frame(body, padding=(0, 0, 6, 0))
        right = ttk.Frame(body, padding=(6, 0, 0, 0))
        body.add(left, weight=4)
        body.add(right, weight=3)

        left.columnconfigure(0, weight=1)
        left.rowconfigure(1, weight=1)
        left.rowconfigure(2, weight=0)

        transcript_label = ttk.Label(left, text="Conversation Feed", font=("", 12, "bold"))
        transcript_label.grid(row=0, column=0, sticky="w", pady=(0, 6))

        self.chat_text = scrolledtext.ScrolledText(left, wrap=tk.WORD, state=tk.DISABLED, font=("TkDefaultFont", 10))
        self.chat_text.grid(row=1, column=0, sticky="nsew")
        self.chat_text.tag_configure("user", foreground="#154d9b")
        self.chat_text.tag_configure("system", foreground="#1c6f42")
        self.chat_text.tag_configure("warn", foreground="#a15c00")
        self.chat_text.tag_configure("error", foreground="#a32121")
        self.chat_text.tag_configure("meta", foreground="#666666")

        input_frame = ttk.Frame(left, padding=(0, 8, 0, 0))
        input_frame.grid(row=2, column=0, sticky="nsew")
        input_frame.columnconfigure(0, weight=1)
        input_frame.columnconfigure(1, weight=0)

        self.instruction_entry = scrolledtext.ScrolledText(input_frame, height=5, wrap=tk.WORD)
        self.instruction_entry.grid(row=0, column=0, columnspan=2, sticky="nsew")
        self.instruction_entry.bind("<Control-Return>", self._on_ctrl_enter)

        actions = ttk.Frame(input_frame)
        actions.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        actions.columnconfigure(0, weight=1)

        history_row = ttk.Frame(actions)
        history_row.grid(row=0, column=0, sticky="ew")
        history_row.columnconfigure(1, weight=1)
        ttk.Label(history_row, text="Recent").grid(row=0, column=0, sticky="w", padx=(0, 8))
        self.history_combo = ttk.Combobox(history_row, state="readonly", values=self.command_history)
        self.history_combo.grid(row=0, column=1, sticky="ew")
        self.history_combo.bind("<<ComboboxSelected>>", self._on_history_selected)

        button_row = ttk.Frame(actions)
        button_row.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        for idx in range(8):
            button_row.columnconfigure(idx, weight=1)

        ttk.Button(button_row, text="Send", command=self._send_instruction).grid(row=0, column=0, sticky="ew", padx=(0, 6))
        ttk.Button(button_row, text="Resend Last", command=self._resend_last_instruction).grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Button(button_row, text="Clear Input", command=self._clear_input).grid(row=0, column=2, sticky="ew", padx=6)
        ttk.Button(button_row, text="Clear Feed", command=self._clear_feed).grid(row=0, column=3, sticky="ew", padx=6)
        ttk.Button(button_row, text="Save Feed", command=self._save_feed).grid(row=0, column=4, sticky="ew", padx=6)
        ttk.Button(button_row, text="Prompt: Bench", command=lambda: self._use_quick_prompt("go through the cone gate and stop near the bench")).grid(row=0, column=5, sticky="ew", padx=6)
        ttk.Button(button_row, text="Prompt: Marker", command=lambda: self._use_quick_prompt("find the ArUco marker board and stop in front of it")).grid(row=0, column=6, sticky="ew", padx=6)
        ttk.Button(button_row, text="Prompt: Finish", command=lambda: self._use_quick_prompt("move to the yellow finish zone after passing the tree")).grid(row=0, column=7, sticky="ew", padx=(6, 0))

        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)

        notebook = ttk.Notebook(right)
        notebook.grid(row=0, column=0, sticky="nsew")

        status_tab = ttk.Frame(notebook, padding=10)
        status_tab.columnconfigure(1, weight=1)
        notebook.add(status_tab, text="Task")

        ttk.Label(status_tab, text="Current Position").grid(row=0, column=0, sticky="nw")
        ttk.Label(status_tab, textvariable=self.position_var).grid(row=0, column=1, sticky="nw")
        ttk.Label(status_tab, text="Target Position").grid(row=1, column=0, sticky="nw", pady=(6, 0))
        ttk.Label(status_tab, textvariable=self.target_var).grid(row=1, column=1, sticky="nw", pady=(6, 0))
        ttk.Label(status_tab, text="Diagnostics").grid(row=2, column=0, sticky="nw", pady=(10, 0))
        ttk.Label(status_tab, textvariable=self.diagnostic_var, wraplength=500, justify=tk.LEFT).grid(row=2, column=1, sticky="nw", pady=(10, 0))

        ttk.Label(status_tab, text="Subtasks", font=("", 11, "bold")).grid(row=3, column=0, columnspan=2, sticky="w", pady=(16, 6))
        self.subtasks_text = scrolledtext.ScrolledText(status_tab, height=16, wrap=tk.WORD, state=tk.DISABLED)
        self.subtasks_text.grid(row=4, column=0, columnspan=2, sticky="nsew")
        status_tab.rowconfigure(4, weight=1)

        perception_tab = ttk.Frame(notebook, padding=10)
        perception_tab.columnconfigure(0, weight=1)
        perception_tab.rowconfigure(1, weight=1)
        perception_tab.rowconfigure(3, weight=1)
        notebook.add(perception_tab, text="Perception")

        ttk.Label(perception_tab, text="Grounding Status").grid(row=0, column=0, sticky="w")
        ttk.Label(perception_tab, textvariable=self.grounding_var, wraplength=560, justify=tk.LEFT).grid(row=1, column=0, sticky="nsew", pady=(4, 12))
        ttk.Label(perception_tab, text="Recent Detections", font=("", 11, "bold")).grid(row=2, column=0, sticky="w")
        self.detections_text = scrolledtext.ScrolledText(perception_tab, height=8, wrap=tk.WORD, state=tk.DISABLED)
        self.detections_text.grid(row=3, column=0, sticky="nsew", pady=(6, 12))
        ttk.Label(perception_tab, text="Recent ArUco Markers", font=("", 11, "bold")).grid(row=4, column=0, sticky="w")
        self.aruco_text = scrolledtext.ScrolledText(perception_tab, height=7, wrap=tk.WORD, state=tk.DISABLED)
        self.aruco_text.grid(row=5, column=0, sticky="nsew", pady=(6, 0))
        perception_tab.rowconfigure(5, weight=1)

        camera_tab = ttk.Frame(notebook, padding=10)
        camera_tab.columnconfigure(0, weight=1)
        camera_tab.rowconfigure(1, weight=1)
        notebook.add(camera_tab, text="Camera")

        camera_top = ttk.Frame(camera_tab)
        camera_top.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        camera_top.columnconfigure(0, weight=1)
        camera_top.columnconfigure(1, weight=0)
        ttk.Label(camera_top, textvariable=self.preview_var, justify=tk.LEFT).grid(row=0, column=0, sticky="w")
        ttk.Button(camera_top, text="Save Snapshot", command=self._save_snapshot).grid(row=0, column=1, sticky="e")

        self.preview_label = ttk.Label(camera_tab, anchor="center", relief=tk.SUNKEN)
        self.preview_label.grid(row=1, column=0, sticky="nsew")
        self.preview_label.configure(text="Waiting for image stream...")

        health_tab = ttk.Frame(notebook, padding=10)
        health_tab.columnconfigure(0, weight=1)
        health_tab.rowconfigure(1, weight=1)
        notebook.add(health_tab, text="Health")

        self.topic_tree = ttk.Treeview(health_tab, columns=("topic", "age", "health"), show="headings", height=8)
        self.topic_tree.heading("topic", text="Topic")
        self.topic_tree.heading("age", text="Last Update")
        self.topic_tree.heading("health", text="Health")
        self.topic_tree.column("topic", width=280, anchor=tk.W)
        self.topic_tree.column("age", width=120, anchor=tk.CENTER)
        self.topic_tree.column("health", width=100, anchor=tk.CENTER)
        self.topic_tree.grid(row=0, column=0, sticky="nsew")

        self.health_items = {}
        for topic in self.topic_last_seen:
            self.health_items[topic] = self.topic_tree.insert("", tk.END, values=(topic, "never", "offline"))

        ttk.Label(health_tab, text="Low-noise system log", font=("", 11, "bold")).grid(row=1, column=0, sticky="w", pady=(12, 6))
        self.log_text = scrolledtext.ScrolledText(health_tab, height=16, wrap=tk.WORD, state=tk.DISABLED)
        self.log_text.grid(row=2, column=0, sticky="nsew")
        health_tab.rowconfigure(2, weight=1)

    def _build_header_card(self, parent, column, title, variable):
        frame = ttk.LabelFrame(parent, text=title, padding=(10, 8))
        frame.grid(row=0, column=column, sticky="nsew", padx=4)
        ttk.Label(frame, textvariable=variable, font=("", 11, "bold")).pack(anchor="center")

    def _setup_ros(self):
        self.instruction_pub = rospy.Publisher(self.instruction_topic, StringMsg, queue_size=10, latch=True)

        self.subscribers = [
            rospy.Subscriber(self.vln_status_topic, VehicleStatus, self._on_vln_status, queue_size=10),
            rospy.Subscriber(self.vlm_status_topic, Bool, self._on_vlm_status, queue_size=10),
            rospy.Subscriber(self.subtasks_topic, StringMsg, self._on_subtasks, queue_size=10),
            rospy.Subscriber(self.vlm_error_log_topic, StringMsg, self._on_vlm_error_log, queue_size=50),
            rospy.Subscriber(self.grounding_status_topic, StringMsg, self._on_grounding_status, queue_size=20),
            rospy.Subscriber(self.grounding_detections_topic, Detection2DArray, self._on_detections, queue_size=10),
            rospy.Subscriber(self.aruco_info_topic, ArucoInfo, self._on_aruco_info, queue_size=10),
            rospy.Subscriber(self.final_status_topic, Int32, self._on_final_status, queue_size=10),
        ]
        if self.preview_enabled:
            self.subscribers.append(
                rospy.Subscriber(self.preview_image_topic, CompressedImage, self._on_preview_image, queue_size=1)
            )

    def _mark_topic(self, topic_name):
        self.topic_last_seen[topic_name] = time.time()

    def _on_vln_status(self, msg):
        self._mark_topic(self.vln_status_topic)
        self.event_queue.put(("vln_status", msg))

    def _on_vlm_status(self, msg):
        self._mark_topic(self.vlm_status_topic)
        self.event_queue.put(("vlm_status", bool(msg.data)))

    def _on_subtasks(self, msg):
        self._mark_topic(self.subtasks_topic)
        self.event_queue.put(("subtasks", msg.data))

    def _on_vlm_error_log(self, msg):
        self._mark_topic(self.vlm_error_log_topic)
        self.event_queue.put(("vlm_log", msg.data))

    def _on_grounding_status(self, msg):
        self._mark_topic(self.grounding_status_topic)
        self.event_queue.put(("grounding_status", msg.data))

    def _on_detections(self, msg):
        self._mark_topic(self.grounding_detections_topic)
        self.event_queue.put(("detections", msg))

    def _on_aruco_info(self, msg):
        self._mark_topic(self.aruco_info_topic)
        self.event_queue.put(("aruco", msg))

    def _on_final_status(self, msg):
        self._mark_topic(self.final_status_topic)
        self.event_queue.put(("final_status", int(msg.data)))

    def _on_preview_image(self, msg):
        if not self.preview_enabled:
            return
        try:
            self._mark_topic(self.preview_image_topic)
            np_arr = np.frombuffer(msg.data, np.uint8)
            bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if bgr is None:
                return
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            self.latest_preview_rgb = rgb
            self.preview_frame_counter += 1
            self.event_queue.put(("preview_image", {"shape": rgb.shape, "stamp": msg.header.stamp.to_sec() if msg.header.stamp else 0.0}))
        except Exception as exc:
            self.event_queue.put(("preview_error", str(exc)))

    def _drain_events(self):
        while True:
            try:
                event_type, payload = self.event_queue.get_nowait()
            except queue.Empty:
                break
            self._handle_event(event_type, payload)
        if not rospy.is_shutdown():
            self.root.after(120, self._drain_events)

    def _handle_event(self, event_type, payload):
        if event_type == "vln_status":
            self._handle_vln_status(payload)
        elif event_type == "vlm_status":
            self.vlm_ready_var.set("Ready" if payload else "Loading")
        elif event_type == "subtasks":
            self._handle_subtasks(payload)
        elif event_type == "vlm_log":
            self._append_log(payload)
        elif event_type == "grounding_status":
            self._handle_grounding_status(payload)
        elif event_type == "detections":
            self._handle_detections(payload)
        elif event_type == "aruco":
            self._handle_aruco(payload)
        elif event_type == "final_status":
            self._handle_final_status(payload)
        elif event_type == "preview_image":
            self._handle_preview_image(payload)
        elif event_type == "preview_error":
            self.preview_var.set(f"Camera preview error: {payload}")

    def _handle_vln_status(self, msg):
        state_name = self._state_to_name(msg.state)
        self.state_var.set(state_name)
        self.goal_var.set("Yes" if msg.has_goal else "No")
        self.motion_var.set("Moving" if msg.is_moving else "Stopped")
        self.sensors_var.set("Ready" if msg.sensors_ready else "Not Ready")
        self.nav_var.set("Ready" if msg.navigation_ready else "Not Ready")
        self.subtask_progress_var.set(f"{msg.current_subtask_index + 1 if msg.total_subtasks else 0} / {msg.total_subtasks}")
        self.position_var.set(self._format_point(msg.current_position))
        self.target_var.set(self._format_point(msg.target_position))
        self.diagnostic_var.set((msg.diagnostic_info or msg.state_description or "No diagnostics").strip())

        if msg.current_subtask_json and msg.current_subtask_json != self.last_subtasks_json:
            self.last_subtasks_json = msg.current_subtask_json
            self._set_text_widget(self.subtasks_text, self._pretty_json(msg.current_subtask_json))

        if msg.state != self.last_logged_state:
            self.last_logged_state = msg.state
            self._append_chat("system", f"State changed to {state_name}: {msg.state_description or msg.diagnostic_info or 'no extra description'}")

        if msg.completion_message:
            self.final_status_var.set(msg.completion_message)

    def _handle_subtasks(self, raw_json):
        pretty = self._pretty_json(raw_json)
        self.last_subtasks_json = raw_json
        self._set_text_widget(self.subtasks_text, pretty)
        try:
            subtasks = json.loads(raw_json)
            count = len(subtasks) if isinstance(subtasks, list) else 0
            self._append_chat("system", f"Instruction parsed into {count} subtasks.")
        except Exception:
            self._append_chat("warn", "Received /subtasks but could not parse JSON cleanly.")

    def _handle_grounding_status(self, payload):
        summary = payload
        try:
            data = json.loads(payload)
            event = data.get("event", "status")
            backend = data.get("backend", "")
            reason = data.get("reason", "")
            error = data.get("error", "")
            summary = f"event={event}"
            if backend:
                summary += f", backend={backend}"
            if reason:
                summary += f", reason={reason}"
            if error:
                summary += f", error={error}"
            if event != self.last_grounding_event and event not in ("not_ready",):
                self.last_grounding_event = event
                self._append_chat("system", f"Grounding status updated: {summary}")
        except Exception:
            pass
        self.grounding_var.set(summary)

    def _handle_detections(self, msg):
        lines = []
        parsed = []
        for det in msg.detections:
            bbox = det.bbox
            parsed.append(
                {
                    "label": det.label or det.id,
                    "score": float(det.score),
                    "cx": float(bbox.center.x),
                    "cy": float(bbox.center.y),
                    "w": float(bbox.size_x),
                    "h": float(bbox.size_y),
                }
            )
            lines.append(
                f"{det.label or det.id} | score={det.score:.2f} | "
                f"cx={bbox.center.x:.1f}, cy={bbox.center.y:.1f}, w={bbox.size_x:.1f}, h={bbox.size_y:.1f}"
            )
        self.latest_detections = parsed
        text = "\n".join(lines) if lines else "No detections in the latest frame."
        self._set_text_widget(self.detections_text, text)

        summary = ", ".join(line.split("|")[0].strip() for line in lines[:4]) if lines else "none"
        if summary != self.last_detection_summary:
            self.last_detection_summary = summary
            self._append_chat("system", f"Grounding detections: {summary}")

    def _handle_aruco(self, msg):
        lines = []
        for marker in msg.markers:
            pos = marker.pose.position
            lines.append(f"id={marker.id} @ x={pos.x:.2f}, y={pos.y:.2f}, z={pos.z:.2f}")
        text = "\n".join(lines) if lines else "No markers reported."
        self._set_text_widget(self.aruco_text, text)

        summary = ", ".join(lines[:3]) if lines else "none"
        if summary != self.last_aruco_summary:
            self.last_aruco_summary = summary
            self._append_chat("system", f"Aruco update: {summary}")

    def _handle_final_status(self, code):
        if code == 0:
            summary = "Completed successfully"
            self._append_chat("system", "Task finished successfully.")
        else:
            summary = f"Completed with status code {code}"
            self._append_chat("warn", f"Task ended with status code {code}.")
        self.final_status_var.set(summary)

    def _handle_preview_image(self, meta):
        if self.latest_preview_rgb is None or not self.preview_enabled:
            return
        try:
            frame = self.latest_preview_rgb.copy()
            self._draw_detection_overlay(frame)
            image = PILImage.fromarray(frame)
            if self.preview_resample is not None:
                image.thumbnail((self.preview_max_width, self.preview_max_height), self.preview_resample)
            else:
                image.thumbnail((self.preview_max_width, self.preview_max_height))
            self.preview_photo = ImageTk.PhotoImage(image=image)
            self.preview_label.configure(image=self.preview_photo, text="")
            h, w = self.latest_preview_rgb.shape[:2]
            self.preview_var.set(
                f"Live preview from {self.preview_image_topic} | frame {self.preview_frame_counter} | "
                f"{w}x{h} | detections={len(self.latest_detections)}"
            )
        except Exception as exc:
            self.preview_var.set(f"Camera preview render failed: {exc}")

    def _draw_detection_overlay(self, frame):
        if cv2 is None or frame is None:
            return
        for det in self.latest_detections:
            cx = det["cx"]
            cy = det["cy"]
            bw = max(2.0, det["w"])
            bh = max(2.0, det["h"])
            x1 = int(round(cx - bw / 2.0))
            y1 = int(round(cy - bh / 2.0))
            x2 = int(round(cx + bw / 2.0))
            y2 = int(round(cy + bh / 2.0))
            x1 = max(0, min(frame.shape[1] - 1, x1))
            y1 = max(0, min(frame.shape[0] - 1, y1))
            x2 = max(0, min(frame.shape[1] - 1, x2))
            y2 = max(0, min(frame.shape[0] - 1, y2))

            color = (44, 196, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f'{det["label"]} {det["score"]:.2f}'
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            text_x1 = x1
            text_y1 = max(0, y1 - th - baseline - 6)
            text_x2 = min(frame.shape[1] - 1, text_x1 + tw + 8)
            text_y2 = text_y1 + th + baseline + 6
            cv2.rectangle(frame, (text_x1, text_y1), (text_x2, text_y2), color, -1)
            cv2.putText(
                frame,
                label,
                (text_x1 + 4, text_y2 - baseline - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (20, 20, 20),
                1,
                cv2.LINE_AA,
            )

    def _refresh_topic_health(self):
        now = time.time()
        for topic, item in self.health_items.items():
            ts = self.topic_last_seen.get(topic, 0.0)
            if ts <= 0.0:
                age_text = "never"
                health = "offline"
            else:
                age = now - ts
                age_text = f"{age:.1f}s"
                if age < 2.0:
                    health = "fresh"
                elif age < 8.0:
                    health = "slow"
                else:
                    health = "stale"
            self.topic_tree.item(item, values=(topic, age_text, health))
        if not rospy.is_shutdown():
            self.root.after(500, self._refresh_topic_health)

    def _send_instruction(self):
        instruction = self.instruction_entry.get("1.0", tk.END).strip()
        if not instruction:
            messagebox.showinfo("Empty instruction", "Please enter a navigation instruction first.")
            return
        self.instruction_pub.publish(StringMsg(data=instruction))
        self.last_instruction = instruction
        self._remember_command(instruction)
        self._append_chat("user", instruction)
        self._append_chat("system", "Instruction published. Waiting for parsing and state updates...")

    def _resend_last_instruction(self):
        if not self.last_instruction:
            messagebox.showinfo("No instruction", "There is no previously sent instruction yet.")
            return
        self.instruction_pub.publish(StringMsg(data=self.last_instruction))
        self._append_chat("user", f"[resent] {self.last_instruction}")
        self._append_chat("system", "Last instruction re-published.")

    def _clear_input(self):
        self.instruction_entry.delete("1.0", tk.END)

    def _clear_feed(self):
        self.chat_text.configure(state=tk.NORMAL)
        self.chat_text.delete("1.0", tk.END)
        self.chat_text.configure(state=tk.DISABLED)
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.delete("1.0", tk.END)
        self.log_text.configure(state=tk.DISABLED)
        self.latest_transcript_lines = []
        self._append_chat("system", "Feed cleared.")

    def _save_feed(self):
        path = filedialog.asksaveasfilename(
            title="Save operator feed",
            defaultextension=".log",
            filetypes=[("Log files", "*.log"), ("Text files", "*.txt"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(self.latest_transcript_lines))
            self._append_chat("system", f"Feed saved to {path}")
        except Exception as exc:
            messagebox.showerror("Save failed", str(exc))

    def _save_snapshot(self):
        if self.latest_preview_rgb is None or not self.preview_enabled:
            messagebox.showinfo("No frame", "No camera frame is available yet.")
            return
        path = filedialog.asksaveasfilename(
            title="Save camera snapshot",
            defaultextension=".png",
            filetypes=[("PNG image", "*.png"), ("JPEG image", "*.jpg"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            image = PILImage.fromarray(self.latest_preview_rgb)
            image.save(path)
            self._append_chat("system", f"Saved camera snapshot to {path}")
        except Exception as exc:
            messagebox.showerror("Snapshot failed", str(exc))

    def _use_quick_prompt(self, text):
        self.instruction_entry.delete("1.0", tk.END)
        self.instruction_entry.insert("1.0", text)
        self.instruction_entry.focus_set()

    def _on_history_selected(self, _event):
        value = self.history_combo.get().strip()
        if not value:
            return
        self.instruction_entry.delete("1.0", tk.END)
        self.instruction_entry.insert("1.0", value)

    def _on_ctrl_enter(self, _event):
        self._send_instruction()
        return "break"

    def _remember_command(self, command):
        if command in self.command_history:
            self.command_history.remove(command)
        self.command_history.insert(0, command)
        self.command_history = self.command_history[:12]
        self.history_combo.configure(values=self.command_history)

    def _append_chat(self, kind, text):
        ts = datetime.now().strftime("%H:%M:%S")
        if kind == "user":
            prefix = f"[{ts}] User: "
            tag = "user"
        elif kind == "warn":
            prefix = f"[{ts}] Warning: "
            tag = "warn"
        elif kind == "error":
            prefix = f"[{ts}] Error: "
            tag = "error"
        else:
            prefix = f"[{ts}] System: "
            tag = "system"
        line = prefix + text
        self.latest_transcript_lines.append(line)
        self.chat_text.configure(state=tk.NORMAL)
        self.chat_text.insert(tk.END, line + "\n", tag)
        self.chat_text.configure(state=tk.DISABLED)
        self.chat_text.see(tk.END)

    def _append_log(self, text):
        self.latest_transcript_lines.append(text)
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, text + "\n", "meta")
        self.log_text.configure(state=tk.DISABLED)
        self.log_text.see(tk.END)

    def _set_text_widget(self, widget, text):
        widget.configure(state=tk.NORMAL)
        widget.delete("1.0", tk.END)
        widget.insert("1.0", text)
        widget.configure(state=tk.DISABLED)

    def _state_to_name(self, code):
        mapping = {
            VehicleStatus.STATE_INITIALIZING: "INITIALIZING",
            VehicleStatus.STATE_EXPLORATION: "EXPLORATION",
            VehicleStatus.STATE_NAVIGATION: "NAVIGATION",
            VehicleStatus.STATE_IDLE: "IDLE",
            VehicleStatus.STATE_ERROR: "ERROR",
            VehicleStatus.STATE_EMERGENCY_STOP: "EMERGENCY_STOP",
        }
        return mapping.get(code, f"UNKNOWN({code})")

    def _format_point(self, point_msg: Point):
        return f"({point_msg.x:.2f}, {point_msg.y:.2f}, {point_msg.z:.2f})"

    def _pretty_json(self, raw_text):
        try:
            return json.dumps(json.loads(raw_text), indent=2, ensure_ascii=False)
        except Exception:
            return raw_text

    def _on_ros_shutdown(self):
        try:
            self.root.after(0, self.root.destroy)
        except Exception:
            pass

    def _on_close(self):
        try:
            rospy.signal_shutdown("GUI window closed")
        finally:
            try:
                self.root.destroy()
            except Exception:
                pass

    def run(self):
        self.root.mainloop()


def main():
    app = VlnOperatorGui()
    app.run()


if __name__ == "__main__":
    main()
