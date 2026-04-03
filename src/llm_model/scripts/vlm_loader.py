#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import re
import threading
import time
import uuid
from pathlib import Path

import numpy as np
import rospy
from PIL import Image as PILImage
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool, String as StringMsg

os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')


class VLMModelLoaderNode:
    """
    职责：
    - 启动时加载可用的视觉/多模态模型
    - 加载成功后在 /VLM_Status 发布 True（latched）
    - 订阅 /vlm_query，处理文本或图文请求，发布到 /vlm_response
    - 当模型不可用时，为文本解析和简单视觉路由提供安全降级
    """

    def __init__(self):
        self.status_pub = rospy.Publisher('/VLM_Status', Bool, queue_size=1, latch=True)
        self.vlm_response_pub = rospy.Publisher('/vlm_response', StringMsg, queue_size=10)
        self.error_log_pub = rospy.Publisher('/vlm_error_log', StringMsg, queue_size=10)

        self.vlm_query_sub = rospy.Subscriber('/vlm_query', StringMsg, self._on_vlm_query, queue_size=10)
        self.image_sub = None

        self._model = None
        self._processor = None
        self._torch = None
        self._device = 'cpu'
        self._torch_dtype = None
        self.backend = 'offline'
        self.model_loaded = False
        self.last_error_message = None
        self.latest_cv_image = None

        self.requested_backend = str(rospy.get_param('~backend', 'auto')).strip().lower()
        self.model_path = str(rospy.get_param('~model_path', '')).strip()
        self.default_florence_model_path = str(
            rospy.get_param(
                '~default_florence_model_path',
                os.path.join(os.path.dirname(__file__), '..', 'models', 'microsoft--Florence-2-large'),
            )
        ).strip()
        self.image_topic = str(rospy.get_param('~image_topic', '/magv/camera/image_compressed/compressed')).strip()
        self.load_delay_sec = float(rospy.get_param('~load_delay_sec', 0.5))
        self.publish_ready_without_model = bool(rospy.get_param('~publish_ready_without_model', False))

        self.image_sub = rospy.Subscriber(self.image_topic, CompressedImage, self._on_image, queue_size=1)
        self._schedule_model_loading()
        rospy.loginfo('VLM Loader initialized. Scheduling model load...')

    def _log_error(self, level, message):
        import datetime

        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        log_msg = f'[{timestamp}] [{level}] VLM_LOADER: {message}'
        self.error_log_pub.publish(StringMsg(data=log_msg))

    def _on_image(self, msg: CompressedImage):
        try:
            import cv2

            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if cv_image is not None:
                self.latest_cv_image = cv_image
        except Exception as e:
            rospy.logerr(f'VLM Loader: Failed to decode compressed image: {e}')

    def _schedule_model_loading(self):
        threading.Thread(target=self._load_model_after_delay, daemon=True).start()

    def _load_model_after_delay(self):
        if self.load_delay_sec > 0:
            time.sleep(self.load_delay_sec)
        if rospy.is_shutdown():
            return
        self._delayed_model_loading()

    def _delayed_model_loading(self):
        rospy.loginfo('VLM Loader: starting model loading...')
        try:
            self._load_model()
            self.model_loaded = True
            self.last_error_message = None
            self._log_error('SUCCESS', f'Model loaded successfully (backend={self.backend})')
            self.status_pub.publish(Bool(data=True))
        except Exception as e:
            self.model_loaded = False
            self.last_error_message = str(e)
            self.backend = 'offline'
            rospy.logerr(f'VLM Loader: model load failed: {e}')
            self._log_error('ERROR', f'Model load failed: {e}')
            if self.publish_ready_without_model:
                self._log_error('WARNING', 'Publishing ready=True with offline-only fallback enabled')
                self.status_pub.publish(Bool(data=True))

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
        if any('qwen2_5_vl' in x or 'qwen2vl' in x or 'qwen2_vl' in x for x in archs):
            return 'qwen2_5_vl'
        return ''

    def _resolve_backend_and_path(self):
        candidates = []
        if self.model_path:
            candidates.append(self.model_path)
        if self.default_florence_model_path and self.default_florence_model_path not in candidates:
            candidates.append(self.default_florence_model_path)

        existing = []
        for path in candidates:
            if path and os.path.isdir(path):
                existing.append(path)

        if self.requested_backend in ('qwen', 'qwen2', 'qwen2_5_vl', 'qwen2_vl'):
            for path in existing:
                if 'qwen' in self._detect_model_type(path):
                    return 'qwen', path
            raise RuntimeError(f'No local Qwen-VL model found. candidates={candidates}')

        if self.requested_backend in ('florence', 'florence2'):
            for path in existing:
                if self._detect_model_type(path) == 'florence2':
                    return 'florence2', path
            raise RuntimeError(f'No local Florence-2 model found. candidates={candidates}')

        for path in existing:
            mtype = self._detect_model_type(path)
            if 'qwen' in mtype:
                return 'qwen', path
        for path in existing:
            if self._detect_model_type(path) == 'florence2':
                return 'florence2', path

        raise RuntimeError(f'No supported local VLM model found. candidates={candidates}')

    def _load_model(self):
        self._log_error('INFO', 'Importing ML dependencies...')
        try:
            import torch
            import transformers  # noqa: F401
        except Exception as e:
            raise RuntimeError(f'ML deps import failed: {e}')

        self._torch = torch
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._torch_dtype = torch.float16 if self._device == 'cuda' else torch.float32
        backend, model_dir = self._resolve_backend_and_path()
        self.backend = backend
        self.model_path = model_dir

        if backend == 'qwen':
            self._load_qwen_model(model_dir)
        elif backend == 'florence2':
            self._load_florence_model(model_dir)
        else:
            raise RuntimeError(f'Unsupported backend resolved: {backend}')

    def _accelerate_available(self):
        try:
            import accelerate  # noqa: F401
            return True
        except Exception:
            return False

    def _move_batch_to_device(self, inputs):
        moved = {}
        for key, value in inputs.items():
            if not hasattr(value, 'to'):
                moved[key] = value
                continue

            if self._device == 'cuda' and getattr(value, 'is_floating_point', lambda: False)():
                moved[key] = value.to(device=self._device, dtype=self._torch_dtype)
            else:
                moved[key] = value.to(self._device)
        return moved

    def _load_qwen_model(self, model_dir):
        try:
            from transformers.models.qwen2_5_vl import (
                Qwen2_5_VLForConditionalGeneration,
                Qwen2_5_VLProcessor as QwenProcessor,
            )
            model_api = 'qwen2_5_vl'
        except Exception:
            from transformers.models.qwen2_vl import (
                Qwen2VLForConditionalGeneration as Qwen2_5_VLForConditionalGeneration,
                Qwen2VLProcessor as QwenProcessor,
            )
            model_api = 'qwen2_vl'

        self._log_error('INFO', f'Using transformers API: {model_api}')
        self._processor = QwenProcessor.from_pretrained(model_dir, trust_remote_code=True, local_files_only=True)
        use_device_map = self._device == 'cuda' and self._accelerate_available()
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_dir,
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=self._torch_dtype,
            device_map='auto' if use_device_map else None,
        )
        has_device_map = bool(getattr(self._model, 'hf_device_map', None) or getattr(self._model, 'device_map', None))
        if not has_device_map:
            self._model.to(self._device)
        self._model.eval()
        rospy.loginfo('VLM Loader: loaded Qwen backend from %s', model_dir)

    def _load_florence_model(self, model_dir):
        from transformers import AutoModelForCausalLM, AutoProcessor

        self._processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True, local_files_only=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=self._torch_dtype,
        ).to(self._device)
        self._model.eval()
        rospy.loginfo('VLM Loader: loaded Florence-2 backend from %s', model_dir)

    def _on_vlm_query(self, msg: StringMsg):
        try:
            payload = json.loads(msg.data)
        except Exception as e:
            rospy.logerr(f'VLM Loader: bad /vlm_query JSON: {e}')
            return

        request_id = payload.get('request_id') or str(uuid.uuid4())
        need_image = bool(payload.get('need_image', False))
        qtype = payload.get('type', 'instruction')
        text = (payload.get('text') or '').strip()

        if ('type' not in payload) and ('query' in payload):
            qtype = 'vision_query'
            text = str(payload.get('query') or '').strip()
            need_image = bool(payload.get('image_available', True))

        if qtype == 'instruction_offline':
            response = {'request_id': request_id, 'subtasks': self._safe_offline_parse(text)}
            self.vlm_response_pub.publish(StringMsg(data=json.dumps(response)))
            return

        if qtype.startswith('instruction') and self.backend != 'qwen':
            response = {
                'request_id': request_id,
                'subtasks': self._safe_offline_parse(text),
                'backend': self.backend,
            }
            self.vlm_response_pub.publish(StringMsg(data=json.dumps(response)))
            return

        try:
            if need_image:
                if not self.model_loaded or self._model is None or self._processor is None:
                    response = self._offline_vision_query(payload)
                elif self.backend == 'qwen':
                    response = self._run_qwen_vision(text)
                else:
                    response = self._offline_vision_query(payload)
            else:
                if not self.model_loaded or self.backend != 'qwen':
                    response = {'subtasks': self._safe_offline_parse(text), 'backend': self.backend}
                else:
                    response = self._run_qwen_text(text)
        except Exception as e:
            response = {'request_id': request_id, 'error': f'inference_failed: {e}'}

        if isinstance(response, dict):
            response.setdefault('request_id', request_id)
        self.vlm_response_pub.publish(StringMsg(data=json.dumps(response)))

    def _run_qwen_text(self, instruction):
        messages = [{"role": "user", "content": [{"type": "text", "text": instruction}]}]
        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors='pt',
        ).to(self._model.device)

        gen_kwargs = {
            'max_new_tokens': int(rospy.get_param('~llm_max_new_tokens', 128)),
            'do_sample': bool(rospy.get_param('~llm_do_sample', False)),
        }
        if gen_kwargs['do_sample']:
            gen_kwargs['temperature'] = float(rospy.get_param('~llm_temperature', 0.7))

        with self._torch.no_grad():
            outputs = self._model.generate(inputs, **gen_kwargs)
        gen_ids = outputs[0][inputs.shape[1]:]
        tokenizer = getattr(self._processor, 'tokenizer', None)
        response_text = tokenizer.decode(gen_ids, skip_special_tokens=True) if tokenizer is not None else self._processor.decode(gen_ids, skip_special_tokens=True)
        try:
            data = json.loads(response_text.strip())
        except Exception:
            data = self._extract_json(response_text)
        if not isinstance(data, list):
            data = []
        return {'subtasks': data}

    def _run_qwen_vision(self, query_text):
        if self.latest_cv_image is None:
            return {'error': 'no_image'}
        try:
            import cv2

            rgb = cv2.cvtColor(self.latest_cv_image, cv2.COLOR_BGR2RGB)
            pil_image = PILImage.fromarray(rgb)
        except Exception as e:
            return {'error': f'image_convert_failed: {e}'}

        inputs = self._processor(text=query_text, images=[pil_image], return_tensors='pt')
        inputs = self._move_batch_to_device(inputs)
        gen_kwargs = {
            'max_new_tokens': int(rospy.get_param('~llm_max_new_tokens', 128)),
            'do_sample': bool(rospy.get_param('~llm_do_sample', False)),
        }
        if gen_kwargs['do_sample']:
            gen_kwargs['temperature'] = float(rospy.get_param('~llm_temperature', 0.7))

        with self._torch.no_grad():
            outputs = self._model.generate(**inputs, **gen_kwargs)

        tokenizer = getattr(self._processor, 'tokenizer', None)
        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True) if tokenizer is not None else self._processor.decode(outputs[0], skip_special_tokens=True)
        data = self._extract_json(response_text)
        return {'vision_result': data, 'vision_result_text': response_text}

    def _offline_vision_query(self, payload):
        markers = payload.get('aruco_markers') or []
        if len(markers) == 1:
            return {
                'target_found': True,
                'target_aruco_id': markers[0].get('id'),
                'reason': 'single_marker_fallback',
                'backend': self.backend,
            }

        query_text = str(payload.get('query') or payload.get('text') or '')
        ids = re.findall(r'\b(\d{1,3})\b', query_text)
        if ids:
            wanted = int(ids[0])
            for marker in markers:
                if int(marker.get('id')) == wanted:
                    return {
                        'target_found': True,
                        'target_aruco_id': wanted,
                        'reason': 'query_id_match_fallback',
                        'backend': self.backend,
                    }

        return {
            'target_found': False,
            'reason': 'no_supported_vqa_backend',
            'backend': self.backend,
        }

    def _extract_json(self, text):
        m = re.search(r'\[.*\]', text, re.DOTALL)
        if not m:
            return []
        try:
            return json.loads(m.group(0))
        except Exception:
            return []

    def _safe_offline_parse(self, instruction):
        try:
            return self._offline_parse_subtasks(instruction)
        except Exception as e:
            self._log_error('ERROR', f'offline parse failed: {e}')
            return []

    def _offline_parse_subtasks(self, instruction):
        text = (instruction or '').strip().lower()
        if not text:
            return []

        dir_map = {
            'forward': [r'forward', r'ahead', r'straight', r'go on', r'move on', r'向前', r'前进', r'直走', r'直行'],
            'backward': [r'backward', r'back', r'reverse', r'向后', r'后退'],
            'left': [r'left', r'to the left', r'leftward', r'向左', r'左转', r'左侧', r'左边'],
            'right': [r'right', r'to the right', r'rightward', r'向右', r'右转', r'右侧', r'右边'],
        }

        directions = []
        for direction, patterns in dir_map.items():
            for pattern in patterns:
                if re.search(rf'\b{pattern}\b', text):
                    directions.append(direction)
                    break

        goal = None
        match = re.search(r'\bto\s+the\s+([a-z_\- ]{2,})', text)
        if not match:
            match = re.search(r'\bat\s+the\s+([a-z_\- ]{2,})', text)
        if not match:
            match = re.search(r'\bto\s+([a-z_\- ]{2,})', text)
        if not match:
            match = re.search(r'到([\u4e00-\u9fa5a-z0-9_\-]{1,8})(?:那|旁边|那里|位置)?', text)
        if not match:
            match = re.search(r'去([\u4e00-\u9fa5a-z0-9_\-]{1,8})(?:那|旁边|那里|位置)?', text)
        if match:
            goal = match.group(1).strip().split(' ')[0]

        if not directions:
            directions = ['forward']

        return [{f'subtask_{idx}': direction, 'goal': goal} for idx, direction in enumerate(directions, start=1)]


def main():
    rospy.init_node('vlm_loader', anonymous=False)
    _ = VLMModelLoaderNode()
    rospy.spin()


if __name__ == '__main__':
    main()
