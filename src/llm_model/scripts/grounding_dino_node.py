#!/catkin_ws/venv310/bin/python3
# -*- coding: utf-8 -*-

import os
import time
import threading

import rospy
import json
from std_msgs.msg import String as StringMsg
from sensor_msgs.msg import CompressedImage
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point

from magv_vln_msgs.msg import (
    VehicleStatus,
    Detection2DArray,
    Detection2D,
    BoundingBox2D,
)

import numpy as np
import cv2
from PIL import Image as PILImage

# Lazy import heavy deps
_auto_proc = None
_auto_model = None
_torch = None

try:
    import tf.transformations as tfs
except Exception:
    tfs = None


def _decode_compressed_image(msg):
    np_arr = np.frombuffer(msg.data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return img


def _to_pil_rgb(cv_bgr):
    return PILImage.fromarray(cv2.cvtColor(cv_bgr, cv2.COLOR_BGR2RGB))


class GroundingDINONode:
    def __init__(self):
        rospy.loginfo("[grounding_dino_node] Initializing...")

        # Params
        self.model_path = rospy.get_param(
            '~model_path', '/catkin_ws/src/llm_model/models/GroundingDino/grounding-dino-base'
        )
        self.box_threshold = float(rospy.get_param('~box_threshold', 0.3))
        self.text_threshold = float(rospy.get_param('~text_threshold', 0.25))
        self.image_topic = rospy.get_param(
            '~image_topic', '/magv/camera/image_compressed/compressed'
        )
        # run detection in these states; core_node 仅在 INITIALIZING 状态消费 yaw 结合检测
        self.enable_in_states = set([
            VehicleStatus.STATE_INITIALIZING,
            VehicleStatus.STATE_NAVIGATION,
        ])
        self.min_interval_sec = float(rospy.get_param('~min_interval_sec', 0.25))  # throttle ~4Hz
        self.force_enable = bool(rospy.get_param('~force_enable', False))  # debug: bypass state gate

        # Publishers
        self.detections_pub = rospy.Publisher(
            '/grounding_dino/detections', Detection2DArray, queue_size=1
        )
        self.status_pub = rospy.Publisher(
            '/grounding_dino/status', StringMsg, queue_size=1, latch=True
        )

        # Subscribers
        self.prompt_sub = rospy.Subscriber(
            '/grounding_dino/prompt', StringMsg, self._on_prompt, queue_size=10
        )
        self.image_sub = rospy.Subscriber(
            self.image_topic, CompressedImage, self._on_image, queue_size=1
        )
        self.state_sub = rospy.Subscriber(
            '/vln_status', VehicleStatus, self._on_status, queue_size=1
        )
        self.odom_sub = rospy.Subscriber(
            '/magv/odometry/gt', Odometry, self._on_odom, queue_size=1
        )

        # State
        self._latest_prompt = None
        self._latest_image = None
        self._latest_img_header = None
        self._current_state = VehicleStatus.STATE_IDLE
        self._current_yaw = 0.0  # 仅用于日志
        self._lock = threading.Lock()
        self._processing = False
        self._last_run_ts = 0.0

        # Deferred model loading to avoid blocking init
        load_delay = float(rospy.get_param('~load_delay_sec', 0.5))
        if load_delay > 0:
            rospy.Timer(rospy.Duration(load_delay), self._load_model_timer, oneshot=True)
            rospy.loginfo(
                f"[grounding_dino_node] Scheduled model load in {load_delay:.2f}s, model_path={self.model_path}"
            )
        else:
            rospy.loginfo(
                f"[grounding_dino_node] Starting model load immediately, model_path={self.model_path}"
            )
            self._load_model_timer(None)

    # ------------------------ ROS Callbacks ------------------------
    def _on_prompt(self, msg: StringMsg):
        text = (msg.data or '').strip()
        # 空字符串表示清空/禁用检测
        if not text:
            with self._lock:
                self._latest_prompt = None
            rospy.loginfo_throttle(5.0, "[grounding_dino_node] Prompt cleared; detection disabled until a new prompt arrives.")
            return
        # GroundingDINO 要求 lower + 以点号结束；支持多目标以句号分隔
        text = text.lower()
        if not text.endswith('.'):  # 避免重复多个点
            text = text + '.'
        with self._lock:
            self._latest_prompt = text
        rospy.loginfo_throttle(5.0, f"[grounding_dino_node] Received prompt: '{text}'")

    def _on_status(self, msg: VehicleStatus):
        with self._lock:
            self._current_state = msg.state

    def _on_odom(self, msg: Odometry):
        if tfs is None:
            return
        q = msg.pose.pose.orientation
        yaw = tfs.euler_from_quaternion([q.x, q.y, q.z, q.w])[2]
        with self._lock:
            self._current_yaw = float(yaw)

    def _on_image(self, msg: CompressedImage):
        try:
            img = _decode_compressed_image(msg)
            if img is None:
                rospy.logwarn_throttle(5.0, "[grounding_dino_node] Failed to decode image")
                return
            with self._lock:
                self._latest_image = img
                self._latest_img_header = msg.header
            self._maybe_trigger_detection()
        except Exception as e:
            rospy.logwarn_throttle(5.0, f"[grounding_dino_node] image callback exception: {e}")

    # ------------------------ Model Loading ------------------------
    def _load_model_timer(self, _):
        threading.Thread(target=self._load_model, daemon=True).start()

    def _load_model(self):
        global _auto_proc, _auto_model, _torch
        t0 = time.time()
        try:
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
            import torch
            _auto_proc = AutoProcessor
            _auto_model = AutoModelForZeroShotObjectDetection
            _torch = torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self._device = device
            self.processor = AutoProcessor.from_pretrained(self.model_path)
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_path).to(device)
            dt = time.time() - t0
            rospy.loginfo(
                f"[grounding_dino_node] Model loaded on {device} in {dt:.2f}s"
            )
            try:
                st = {
                    'event': 'model_loaded',
                    'device': device,
                    'model_path': self.model_path,
                    'load_time_sec': dt,
                }
                self.status_pub.publish(StringMsg(data=json.dumps(st)))
            except Exception:
                pass
        except Exception as e:
            rospy.logerr(f"[grounding_dino_node] Failed to load GroundingDINO: {e}")
            self.processor = None
            self.model = None
            try:
                st = {
                    'event': 'model_load_failed',
                    'error': str(e),
                    'model_path': self.model_path,
                }
                self.status_pub.publish(StringMsg(data=json.dumps(st)))
            except Exception:
                pass

    # ------------------------ Detection Flow ------------------------
    def _maybe_trigger_detection(self):
        with self._lock:
            has_model = (hasattr(self, 'processor') and self.processor is not None and 
                        hasattr(self, 'model') and self.model is not None)
            has_image = (self._latest_image is not None)
            has_prompt = (self._latest_prompt is not None)
            state_ok = (self.force_enable or (self._current_state in self.enable_in_states))
            busy = self._processing
            ready = (has_model and has_image and has_prompt and state_ok and not busy)
            last_ts = self._last_run_ts
        if not ready:
            # 每5秒提示一次缺失原因，便于快速定位，并发布状态
            reasons = []
            if not has_model:
                reasons.append('model_not_loaded')
            if not has_image:
                reasons.append('no_image')
            if not has_prompt:
                reasons.append('no_prompt')
            if not state_ok:
                reasons.append(f'state={self._current_state}')
            if busy:
                reasons.append('busy')
            msg = f"[grounding_dino_node] Not ready for detection: {', '.join(reasons)}"
            rospy.logdebug_throttle(5.0, msg)
            try:
                st = {
                    'event': 'not_ready',
                    'model_loaded': bool(has_model),
                    'has_image': bool(has_image),
                    'has_prompt': bool(has_prompt),
                    'state_ok': bool(state_ok),
                    'processing': bool(busy),
                    'reasons': reasons,
                }
                self.status_pub.publish(StringMsg(data=json.dumps(st)))
            except Exception:
                pass
            return
        if (time.time() - last_ts) < self.min_interval_sec:
            return
        threading.Thread(target=self._run_detection_once, daemon=True).start()

    def _run_detection_once(self):
        with self._lock:
            if self._processing:
                return
            self._processing = True
            img_bgr = self._latest_image.copy() if self._latest_image is not None else None
            header = self._latest_img_header
            prompt = self._latest_prompt
            yaw = self._current_yaw
        try:
            if img_bgr is None or prompt is None:
                return
            # Publish running status once per run
            try:
                st = {
                    'event': 'run_detection',
                    'state': int(self._current_state),
                    'prompt': str(prompt)[:200],
                }
                self.status_pub.publish(StringMsg(data=json.dumps(st)))
            except Exception:
                pass
            pil_img = _to_pil_rgb(img_bgr)
            w, h = pil_img.size
            inputs = self.processor(images=pil_img, text=prompt, return_tensors="pt")
            if hasattr(self, '_device') and self._device:
                inputs = {k: v.to(self._device) for k, v in inputs.items()}
            with _torch.no_grad():
                outputs = self.model(**inputs)
            # Post-process with robust API compatibility across transformers versions
            try:
                # Preferred: keyword args (newer versions)
                results = self.processor.post_process_grounded_object_detection(
                    outputs,
                    inputs["input_ids"],
                    box_threshold=self.box_threshold,
                    text_threshold=self.text_threshold,
                    target_sizes=[(h, w)],
                )
            except TypeError:
                try:
                    # Fallback: positional args (some versions don't accept keywords)
                    results = self.processor.post_process_grounded_object_detection(
                        outputs,
                        inputs["input_ids"],
                        self.box_threshold,
                        self.text_threshold,
                        [(h, w)],
                    )
                except TypeError:
                    try:
                        # Fallback: single threshold API (very old variants)
                        results = self.processor.post_process_grounded_object_detection(
                            outputs=outputs,
                            input_ids=inputs["input_ids"],
                            threshold=self.box_threshold,
                            target_sizes=[(h, w)],
                        )
                    except Exception as e:
                        rospy.logwarn_throttle(5.0, f"[grounding_dino_node] post_process compatibility failed: {e}")
                        results = [{"boxes": [], "scores": [], "labels": []}]
            detections = results[0]
            boxes = detections.get('boxes', [])
            scores = detections.get('scores', [])
            labels = detections.get('labels', [])

            msg = Detection2DArray()
            msg.header.stamp = rospy.Time.now()
            msg.header.frame_id = header.frame_id if header and header.frame_id else 'camera'
            out_list = []
            for i in range(len(boxes)):
                x1, y1, x2, y2 = [float(v) for v in boxes[i].tolist()]
                score = float(scores[i]) if i < len(scores) else 0.0
                label = str(labels[i]) if i < len(labels) else ''
                # Some transformers return label as tensor/int, ensure str
                label = (label if isinstance(label, str) else str(label)).strip()

                det = Detection2D()
                det.header = msg.header
                det.id = str(i)
                det.label = label
                det.score = score
                bbox = BoundingBox2D()
                center = Point()
                center.x = (x1 + x2) / 2.0
                center.y = (y1 + y2) / 2.0
                center.z = 0.0
                bbox.center = center
                bbox.size_x = max(0.0, x2 - x1)
                bbox.size_y = max(0.0, y2 - y1)
                det.bbox = bbox
                out_list.append(det)

            msg.detections = out_list
            self.detections_pub.publish(msg)
            rospy.loginfo_throttle(
                2.0,
                f"[grounding_dino_node] Published {len(out_list)} detections (yaw={yaw:.2f})"
            )
        except Exception as e:
            rospy.logwarn_throttle(5.0, f"[grounding_dino_node] Detection error: {e}")
        finally:
            with self._lock:
                self._processing = False
                self._last_run_ts = time.time()


def main():
    rospy.init_node('grounding_dino_node')
    node = GroundingDINONode()
    rospy.loginfo('[grounding_dino_node] Ready.')
    rospy.spin()


if __name__ == '__main__':
    main()
