#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import rospy
from PIL import Image as PILImage
from geometry_msgs.msg import Point
from magv_vln_msgs.msg import BoundingBox2D, Detection2D, Detection2DArray, VehicleStatus
from nav_msgs.msg import Odometry
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String as StringMsg

try:
    import tf.transformations as tfs
except Exception:
    tfs = None


def _decode_compressed_image(msg):
    np_arr = np.frombuffer(msg.data, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)


def _to_pil_rgb(cv_bgr):
    return PILImage.fromarray(cv2.cvtColor(cv_bgr, cv2.COLOR_BGR2RGB))


class GroundingDINONode:
    def __init__(self):
        rospy.loginfo('[grounding_dino_node] Initializing...')

        self.requested_backend = str(rospy.get_param('~backend', 'auto')).strip().lower()
        self.model_path = str(
            rospy.get_param(
                '~model_path',
                os.path.join(os.path.dirname(__file__), '..', 'models', 'GroundingDino', 'grounding-dino-base'),
            )
        ).strip()
        self.fallback_model_path = str(rospy.get_param('~fallback_model_path', '')).strip()
        self.box_threshold = float(rospy.get_param('~box_threshold', 0.3))
        self.text_threshold = float(rospy.get_param('~text_threshold', 0.25))
        self.image_topic = rospy.get_param('~image_topic', '/magv/camera/image_compressed/compressed')
        self.odometry_topic = rospy.get_param('~odometry_topic', '/magv/odometry/gt')
        self.vln_status_topic = rospy.get_param('~vln_status_topic', '/vln_status')
        self.prompt_topic = rospy.get_param('~prompt_topic', '/grounding_dino/prompt')
        self.detections_topic = rospy.get_param('~detections_topic', '/grounding_dino/detections')
        self.status_topic = rospy.get_param('~status_topic', '/grounding_dino/status')
        self.min_interval_sec = float(rospy.get_param('~min_interval_sec', 0.25))
        self.force_enable = bool(rospy.get_param('~force_enable', False))
        self.florence_max_new_tokens = int(rospy.get_param('~florence_max_new_tokens', 256))
        self.florence_num_beams = int(rospy.get_param('~florence_num_beams', 3))
        self.enable_in_states = {
            VehicleStatus.STATE_INITIALIZING,
            VehicleStatus.STATE_NAVIGATION,
        }

        self.detections_pub = rospy.Publisher(self.detections_topic, Detection2DArray, queue_size=1)
        self.status_pub = rospy.Publisher(self.status_topic, StringMsg, queue_size=1, latch=True)

        self.prompt_sub = rospy.Subscriber(self.prompt_topic, StringMsg, self._on_prompt, queue_size=10)
        self.image_sub = rospy.Subscriber(self.image_topic, CompressedImage, self._on_image, queue_size=1)
        self.state_sub = rospy.Subscriber(self.vln_status_topic, VehicleStatus, self._on_status, queue_size=1)
        self.odom_sub = rospy.Subscriber(self.odometry_topic, Odometry, self._on_odom, queue_size=1)

        self.backend = 'offline'
        self.processor = None
        self.model = None
        self._torch = None
        self._device = 'cpu'
        self._torch_dtype = None
        self._latest_prompt = None
        self._latest_image = None
        self._latest_img_header = None
        self._current_state = VehicleStatus.STATE_IDLE
        self._current_yaw = 0.0
        self._lock = threading.Lock()
        self._processing = False
        self._last_run_ts = 0.0

        self.load_delay_sec = float(rospy.get_param('~load_delay_sec', 0.5))
        self._schedule_model_loading()
        rospy.loginfo('[grounding_dino_node] Scheduled model load in %.2fs', self.load_delay_sec)

    def _publish_status(self, payload):
        try:
            self.status_pub.publish(StringMsg(data=json.dumps(payload)))
        except Exception:
            pass

    def _detect_model_type(self, model_dir):
        cfg_path = Path(model_dir) / 'config.json'
        if not cfg_path.exists():
            return ''
        try:
            cfg = json.loads(cfg_path.read_text())
        except Exception:
            return ''
        model_type = str(cfg.get('model_type', '')).strip().lower()
        if model_type:
            return model_type
        archs = [str(x).lower() for x in cfg.get('architectures', [])]
        if any('florence' in x for x in archs):
            return 'florence2'
        return ''

    def _grounding_model_exists(self, model_dir):
        if not model_dir or not os.path.isdir(model_dir):
            return False
        return True

    def _on_prompt(self, msg: StringMsg):
        text = (msg.data or '').strip()
        if not text:
            with self._lock:
                self._latest_prompt = None
            rospy.loginfo_throttle(5.0, '[grounding_dino_node] Prompt cleared; detection disabled.')
            return
        text = text.lower()
        if not text.endswith('.'):
            text += '.'
        with self._lock:
            self._latest_prompt = text
        rospy.loginfo_throttle(5.0, "[grounding_dino_node] Received prompt: '%s'", text)

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
                rospy.logwarn_throttle(5.0, '[grounding_dino_node] Failed to decode image')
                return
            with self._lock:
                self._latest_image = img
                self._latest_img_header = msg.header
            self._maybe_trigger_detection()
        except Exception as e:
            rospy.logwarn_throttle(5.0, f'[grounding_dino_node] image callback exception: {e}')

    def _schedule_model_loading(self):
        threading.Thread(target=self._load_model_after_delay, daemon=True).start()

    def _load_model_after_delay(self):
        if self.load_delay_sec > 0:
            time.sleep(self.load_delay_sec)
        if rospy.is_shutdown():
            return
        threading.Thread(target=self._load_model, daemon=True).start()

    def _load_model(self):
        t0 = time.time()
        try:
            import torch

            self._torch = torch
            self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self._torch_dtype = torch.float16 if self._device == 'cuda' else torch.float32

            if self.requested_backend in ('auto', 'grounding_dino', 'groundingdino') and self._grounding_model_exists(self.model_path):
                from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

                self.processor = AutoProcessor.from_pretrained(self.model_path, local_files_only=True)
                self.model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_path, local_files_only=True).to(self._device)
                self.backend = 'grounding_dino'
            elif self.requested_backend in ('auto', 'florence', 'florence2') and self.fallback_model_path and self._detect_model_type(self.fallback_model_path) == 'florence2':
                from transformers import AutoModelForCausalLM, AutoProcessor

                self.processor = AutoProcessor.from_pretrained(self.fallback_model_path, trust_remote_code=True, local_files_only=True)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.fallback_model_path,
                    trust_remote_code=True,
                    local_files_only=True,
                    torch_dtype=self._torch_dtype,
                ).to(self._device)
                self.backend = 'florence2'
            else:
                raise RuntimeError(
                    f'No supported detection backend found. requested={self.requested_backend}, '
                    f'grounding_path={self.model_path}, florence_path={self.fallback_model_path}'
                )

            self.model.eval()
            dt = time.time() - t0
            rospy.loginfo('[grounding_dino_node] Model backend=%s loaded on %s in %.2fs', self.backend, self._device, dt)
            self._publish_status(
                {
                    'event': 'model_loaded',
                    'backend': self.backend,
                    'device': self._device,
                    'model_path': self.model_path if self.backend == 'grounding_dino' else self.fallback_model_path,
                    'load_time_sec': dt,
                }
            )
        except Exception as e:
            self.processor = None
            self.model = None
            self.backend = 'offline'
            rospy.logerr(f'[grounding_dino_node] Failed to load detection backend: {e}')
            self._publish_status({'event': 'model_load_failed', 'error': str(e)})

    def _maybe_trigger_detection(self):
        with self._lock:
            has_model = self.processor is not None and self.model is not None
            has_image = self._latest_image is not None
            has_prompt = self._latest_prompt is not None
            state_ok = self.force_enable or (self._current_state in self.enable_in_states)
            busy = self._processing
            ready = has_model and has_image and has_prompt and state_ok and not busy
            last_ts = self._last_run_ts

        if not ready:
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
            self._publish_status(
                {
                    'event': 'not_ready',
                    'backend': self.backend,
                    'reasons': reasons,
                    'state_ok': state_ok,
                }
            )
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
            if self.backend == 'grounding_dino':
                detections = self._run_grounding_dino(img_bgr, prompt)
            elif self.backend == 'florence2':
                detections = self._run_florence_grounding(img_bgr, prompt)
            else:
                detections = []
            self._publish_detections(header, detections, yaw)
        except Exception as e:
            rospy.logwarn_throttle(5.0, f'[grounding_dino_node] Detection error: {e}')
        finally:
            with self._lock:
                self._processing = False
                self._last_run_ts = time.time()

    def _run_grounding_dino(self, img_bgr, prompt):
        pil_img = _to_pil_rgb(img_bgr)
        w, h = pil_img.size
        inputs = self.processor(images=pil_img, text=prompt, return_tensors='pt')
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with self._torch.no_grad():
            outputs = self.model(**inputs)

        try:
            results = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs['input_ids'],
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                target_sizes=[(h, w)],
            )
        except TypeError:
            try:
                results = self.processor.post_process_grounded_object_detection(
                    outputs,
                    inputs['input_ids'],
                    self.box_threshold,
                    self.text_threshold,
                    [(h, w)],
                )
            except TypeError:
                results = self.processor.post_process_grounded_object_detection(
                    outputs=outputs,
                    input_ids=inputs['input_ids'],
                    threshold=self.box_threshold,
                    target_sizes=[(h, w)],
                )

        det = results[0]
        boxes = det.get('boxes', [])
        scores = det.get('scores', [])
        labels = det.get('labels', [])
        out = []
        for idx in range(len(boxes)):
            x1, y1, x2, y2 = [float(v) for v in boxes[idx].tolist()]
            score = float(scores[idx]) if idx < len(scores) else 0.0
            label = str(labels[idx]) if idx < len(labels) else ''
            out.append((label.strip(), score, x1, y1, x2, y2))
        return out

    def _run_florence_grounding(self, img_bgr, prompt):
        pil_img = _to_pil_rgb(img_bgr)
        w, h = pil_img.size
        phrase = prompt.strip().strip('.')
        if not phrase:
            return []

        task_prompt = '<CAPTION_TO_PHRASE_GROUNDING>' + phrase
        inputs = self.processor(text=task_prompt, images=pil_img, return_tensors='pt').to(self._device, self._torch_dtype)
        with self._torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs['input_ids'],
                pixel_values=inputs['pixel_values'],
                max_new_tokens=self.florence_max_new_tokens,
                do_sample=False,
                num_beams=self.florence_num_beams,
            )
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed = self.processor.post_process_generation(
            generated_text,
            task='<CAPTION_TO_PHRASE_GROUNDING>',
            image_size=(w, h),
        )
        results = parsed.get('<CAPTION_TO_PHRASE_GROUNDING>', {}) if isinstance(parsed, dict) else {}
        boxes = results.get('bboxes', [])
        labels = results.get('labels', [])
        out = []
        for idx, box in enumerate(boxes):
            if len(box) != 4:
                continue
            x1, y1, x2, y2 = [float(v) for v in box]
            label = str(labels[idx]) if idx < len(labels) else phrase
            out.append((label.strip(), 1.0, x1, y1, x2, y2))
        return out

    def _publish_detections(self, header, detections, yaw):
        msg = Detection2DArray()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = header.frame_id if header and header.frame_id else 'camera'

        out_list = []
        for idx, (label, score, x1, y1, x2, y2) in enumerate(detections):
            det = Detection2D()
            det.header = msg.header
            det.id = str(idx)
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
        self._publish_status({'event': 'published', 'backend': self.backend, 'count': len(out_list)})
        rospy.loginfo_throttle(2.0, '[grounding_dino_node] Published %d detections (backend=%s yaw=%.2f)', len(out_list), self.backend, yaw)


def main():
    rospy.init_node('grounding_dino_node')
    _ = GroundingDINONode()
    rospy.loginfo('[grounding_dino_node] Ready.')
    rospy.spin()


if __name__ == '__main__':
    main()
