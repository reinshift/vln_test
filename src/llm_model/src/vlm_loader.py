#!/catkin_ws/venv310/bin/python3
# -*- coding: utf-8 -*-

import os
import re
import json
import math
import rospy
import uuid
from std_msgs.msg import String as StringMsg, Bool
from sensor_msgs.msg import CompressedImage
from PIL import Image as PILImage
import numpy as np

class VLMModelLoaderNode:
    """
    仅负责：
    - 启动时加载 VLM（模型与处理器）
    - 加载成功后在 /VLM_Status 发布 True（latched）
    - 订阅 /vlm_query，处理文本或图文请求，发布到 /vlm_response（携带 request_id 回传）
    - 可选订阅摄像头图像（compressed），以便图文推理
    """
    def __init__(self):
        # Publishers (latched True once loaded)
        self.status_pub = rospy.Publisher('/VLM_Status', Bool, queue_size=1, latch=True)
        self.vlm_response_pub = rospy.Publisher('/vlm_response', StringMsg, queue_size=10)
        self.error_log_pub = rospy.Publisher('/vlm_error_log', StringMsg, queue_size=10)

        # Subscribers
        self.vlm_query_sub = rospy.Subscriber('/vlm_query', StringMsg, self._on_vlm_query, queue_size=10)
        self.image_sub = rospy.Subscriber('/magv/camera/image_compressed/compressed', CompressedImage, self._on_image, queue_size=1)

        # State
        self._model = None
        self._processor = None
        self.model_loaded = False
        self.last_error_message = None
        self.latest_cv_image = None

        # Params
        self.model_path = rospy.get_param('~model_path', os.path.join(os.path.dirname(__file__), '..', 'models', 'Qwen2.5-VL-7B-Instruct'))
        self.load_delay_sec = float(rospy.get_param('~load_delay_sec', 1.0))

        # Delay load to avoid blocking init
        rospy.Timer(rospy.Duration(self.load_delay_sec), self._delayed_model_loading, oneshot=True)
        rospy.loginfo('VLM Loader initialized. Scheduling model load...')

    def _log_error(self, level, message):
        import datetime
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        log_msg = f"[{timestamp}] [{level}] VLM_LOADER: {message}"
        self.error_log_pub.publish(StringMsg(data=log_msg))

    def _on_image(self, msg: CompressedImage):
        try:
            import cv2
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if cv_image is not None:
                self.latest_cv_image = cv_image
        except Exception as e:
            rospy.logerr(f"VLM Loader: Failed to decode compressed image: {e}")

    def _delayed_model_loading(self, _):
        rospy.loginfo('VLM Loader: starting model loading...')
        try:
            self._load_model()
            self.model_loaded = True
            self.last_error_message = None
            rospy.loginfo('VLM Loader: model loaded successfully.')
            self._log_error('SUCCESS', 'Model loaded successfully')
            # Publish latched True exactly once
            self.status_pub.publish(Bool(data=True))
        except Exception as e:
            self.model_loaded = False
            self.last_error_message = str(e)
            rospy.logerr(f'VLM Loader: model load failed: {e}')
            self._log_error('ERROR', f'Model load failed: {e}')
            # 发布 False 以明确状态（非 latched True）——这里不发布，以免卡住旧订阅者；仅在成功时发布 True

    def _load_model(self):
        # Import deps lazily
        self._log_error('INFO', 'Importing ML dependencies...')
        try:
            import torch
            import safetensors
            import transformers
            # generation utils warmup
            try:
                import transformers.generation.utils as _gen_utils  # noqa: F401
            except Exception as ge:
                self._log_error('WARNING', f'Pre-import transformers.generation.utils failed: {ge}')

            # Prefer Qwen2.5-VL API, fallback to Qwen2-VL
            try:
                from transformers.models.qwen2_5_vl import (
                    Qwen2_5_VLProcessor as AutoProcessor,
                    Qwen2_5_VLForConditionalGeneration,
                )
                model_api = 'qwen2_5_vl'
            except Exception:
                from transformers.models.qwen2_vl import (
                    Qwen2VLProcessor as AutoProcessor,
                    Qwen2VLForConditionalGeneration as Qwen2_5_VLForConditionalGeneration,
                )
                model_api = 'qwen2_vl'
            self._log_error('INFO', f'Using transformers API: {model_api}')
        except Exception as e:
            raise RuntimeError(f'ML deps import failed: {e}')

        model_dir = self.model_path
        if not os.path.exists(model_dir):
            raise RuntimeError(f'Model directory not found: {model_dir}')

        import torch  # type: ignore
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        torch_dtype = torch.float16 if device == 'cuda' else torch.float32
        self._processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_dir,
            torch_dtype=torch_dtype,
            device_map='auto' if torch.cuda.is_available() else None,
        )
        # If accelerate device_map present, skip .to(device)
        has_device_map = bool(getattr(self._model, 'hf_device_map', None) or getattr(self._model, 'device_map', None))
        if not has_device_map:
            self._model.to(device)
        self._model.eval()

    def _on_vlm_query(self, msg: StringMsg):
        # Expect JSON: {"request_id": str, "type": "instruction"|"vision_query", "text": str, "need_image": bool}
        try:
            payload = json.loads(msg.data)
        except Exception as e:
            rospy.logerr(f'VLM Loader: bad /vlm_query JSON: {e}')
            return

        request_id = payload.get('request_id') or str(uuid.uuid4())
        need_image = bool(payload.get('need_image', False))
        qtype = payload.get('type', 'instruction')
        text = (payload.get('text') or '').strip()

        if not self.model_loaded or self._model is None or self._processor is None:
            resp = {
                'request_id': request_id,
                'error': 'VLM model unavailable',
                'reason': self.last_error_message or 'not_loaded',
            }
            self.vlm_response_pub.publish(StringMsg(data=json.dumps(resp)))
            return

        try:
            if need_image:
                response = self._run_vision(text)
            else:
                response = self._run_text(text)
        except Exception as e:
            response = {'request_id': request_id, 'error': f'inference_failed: {e}'}
        # Ensure request_id is present
        if isinstance(response, dict):
            response.setdefault('request_id', request_id)
        self.vlm_response_pub.publish(StringMsg(data=json.dumps(response)))

    def _run_text(self, instruction: str):
        import torch
        # build prompt/messages for parsing subtasks
        messages = self._build_messages(instruction)
        inputs = self._processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors='pt'
        ).to(self._model.device)
        gen_kwargs = {
            'max_new_tokens': int(rospy.get_param('~llm_max_new_tokens', 128)),
            'do_sample': bool(rospy.get_param('~llm_do_sample', False)),
        }
        if gen_kwargs['do_sample']:
            try:
                gen_kwargs['temperature'] = float(rospy.get_param('~llm_temperature', 0.7))
            except Exception:
                pass
        with torch.no_grad():
            outputs = self._model.generate(inputs, **gen_kwargs)
        gen_ids = outputs[0][inputs.shape[1]:]
        tokenizer = getattr(self._processor, 'tokenizer', None)
        if tokenizer is not None:
            response = tokenizer.decode(gen_ids, skip_special_tokens=True)
        else:
            response = self._processor.decode(gen_ids, skip_special_tokens=True)
        # Try parse JSON array, fallback to []
        try:
            data = json.loads(response.strip())
        except Exception:
            data = self._extract_json(response)
        if not isinstance(data, list):
            data = []
        return {'subtasks': data}

    def _run_vision(self, query_text: str):
        import torch
        if self.latest_cv_image is None:
            return {'error': 'no_image'}
        try:
            import cv2
            rgb = cv2.cvtColor(self.latest_cv_image, cv2.COLOR_BGR2RGB)
            pil_image = PILImage.fromarray(rgb)
        except Exception as e:
            return {'error': f'image_convert_failed: {e}'}
        inputs = self._processor(text=query_text, images=[pil_image], return_tensors='pt').to(self._model.device)
        gen_kwargs = {
            'max_new_tokens': int(rospy.get_param('~llm_max_new_tokens', 128)),
            'do_sample': bool(rospy.get_param('~llm_do_sample', False)),
        }
        if gen_kwargs['do_sample']:
            try:
                gen_kwargs['temperature'] = float(rospy.get_param('~llm_temperature', 0.7))
            except Exception:
                pass
        with torch.no_grad():
            outputs = self._model.generate(**inputs, **gen_kwargs)
        tokenizer = getattr(self._processor, 'tokenizer', None)
        if tokenizer is not None:
            response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        else:
            response_text = self._processor.decode(outputs[0], skip_special_tokens=True)
        data = self._extract_json(response_text)
        return {'vision_result': data}

    def _build_messages(self, instruction: str):
        sys_preamble = (
            'You are a navigation instruction parser. Convert instructions to a JSON array of movement steps. '
            'Direction options: forward, backward, left, right. '
            'Goal should be a specific target if mentioned, otherwise null. '
            'Response must be JSON only, no additional text.'
        )
        messages = [{"role": "system", "content": [{"type": "text", "text": sys_preamble}]}]
        messages.append({"role": "user", "content": [{"type": "text", "text": instruction}]})
        return messages

    def _extract_json(self, text: str):
        m = re.search(r'\[.*\]', text, re.DOTALL)
        if not m:
            return []
        try:
            return json.loads(m.group(0))
        except Exception:
            return []

def main():
    rospy.init_node('vlm_loader', anonymous=False)
    _ = VLMModelLoaderNode()
    rospy.spin()

if __name__ == '__main__':
    main()
