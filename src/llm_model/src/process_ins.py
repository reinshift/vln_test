#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import rospy
from std_msgs.msg import String as StringMsg, Bool
from sensor_msgs.msg import Image, CompressedImage
from PIL import Image as PILImage
import numpy as np

class InstructionProcessorNode:
    def __init__(self):
        # Initialize ROS components
        self.sub = rospy.Subscriber("/instruction", StringMsg, self._on_instruction, queue_size=1)
        self.vlm_query_sub = rospy.Subscriber("/vlm_query", StringMsg, self._on_vlm_query, queue_size=1)
        self.status_pub = rospy.Publisher("/VLM_Status", Bool, queue_size=1)
        self.task_pub = rospy.Publisher("/subtasks", StringMsg, queue_size=1, latch=True)
        self.vlm_response_pub = rospy.Publisher("/vlm_response", StringMsg, queue_size=1)
        # 详细错误日志发布器
        self.error_log_pub = rospy.Publisher("/vlm_error_log", StringMsg, queue_size=10)
        # Subscribe to compressed image topic from rosbag
        self.image_sub = rospy.Subscriber("/magv/camera/image_compressed/compressed", CompressedImage, self._on_image, queue_size=1)

        # Model components
        self._model = None
        self._processor = None
        self.model_loaded = False
        self.last_error_message = None  # 记录最近一次模型加载错误
        self.latest_cv_image = None

        # Start status publisher timer FIRST
        self.status_timer = rospy.Timer(rospy.Duration(2.0), self._publish_status)
        
        # 立即检测环境状态
        self._check_environment()
        
        # 确保节点完全初始化后再尝试加载模型
        rospy.loginfo("Instruction processor initialized. Starting model loading...")
        
        # 延迟加载模型，避免阻塞节点初始化
        rospy.Timer(rospy.Duration(1.0), self._delayed_model_loading, oneshot=True)
        
        rospy.loginfo("Instruction processor ready. Model loading scheduled.")

    def _publish_status(self, event):
        """Publish model status"""
        self.status_pub.publish(Bool(data=self.model_loaded))

    def _on_image(self, msg: CompressedImage):
        """Decode and store the latest image (from CompressedImage)"""
        try:
            import cv2
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if cv_image is None:
                rospy.logwarn("Failed to decode compressed image: None result")
                return
            self.latest_cv_image = cv_image
        except Exception as e:
            rospy.logerr(f"Failed to decode compressed image: {e}")

    def _on_instruction(self, msg: StringMsg):
        """Process incoming instruction. Always attempt fallback parsing even if model is not loaded."""
        instruction = (msg.data or "").strip()
        rospy.loginfo("Received instruction: %s", instruction)

        subtasks = None
        if self.model_loaded:
            try:
                subtasks = self._process_with_llm(instruction)
            except Exception as e:
                rospy.logerr("LLM processing failed: %s", e)

        if not subtasks:
            rospy.logwarn("Using fallback parser for instruction (model_loaded=%s)", self.model_loaded)
            subtasks = self._fallback_parse(instruction)

        # Publish subtasks to new topic (latched)
        task_msg = StringMsg(data=json.dumps(subtasks))
        self.task_pub.publish(task_msg)
        rospy.loginfo("Published subtasks: %s", json.dumps(subtasks, indent=2))

    def _on_vlm_query(self, msg: StringMsg):
        """Process VLM query for ArUco marker analysis"""
        if not self.model_loaded:
            # 当模型不可用时，发布一个标准化的错误响应，避免下游阻塞
            rospy.logerr("Model not loaded, responding with error to /vlm_response")
            error_resp = {
                "error": "VLM model unavailable",
                "reason": self.last_error_message or "not_loaded",
                "target_found": False,
                "confidence": 0.0
            }
            self.vlm_response_pub.publish(StringMsg(data=json.dumps(error_resp)))
            return

        try:
            query_data = json.loads(msg.data)
            rospy.loginfo("Processing VLM query: %s", query_data.get("query", ""))

            # Process the query and generate response
            response = self._process_vlm_query(query_data)

            # Publish response
            response_msg = StringMsg(data=json.dumps(response))
            self.vlm_response_pub.publish(response_msg)
            rospy.loginfo("Published VLM response: %s", json.dumps(response, indent=2))

        except json.JSONDecodeError as e:
            rospy.logerr("Failed to parse VLM query: %s", e)
        except Exception as e:
            rospy.logerr("VLM query processing failed: %s", e)

    def _check_environment(self):
        """检测虚拟环境和基础依赖状态"""
        import sys
        import os
        
        # 检测虚拟环境
        venv_path = os.environ.get('VIRTUAL_ENV')
        if venv_path:
            self._log_error("INFO", f"Virtual environment detected: {venv_path}")
        else:
            self._log_error("WARNING", "No virtual environment detected")
        
        # 检测Python路径
        self._log_error("INFO", f"Python executable: {sys.executable}")
        self._log_error("INFO", f"Python version: {sys.version}")
        
        # 检测基础依赖
        dependencies = [
            ('torch', 'PyTorch'),
            ('transformers', 'Transformers'),
            ('safetensors', 'SafeTensors'),
            ('safetensors.torch', 'SafeTensors Torch'),
            ('PIL', 'Pillow'),
            ('cv2', 'OpenCV'),
            ('numpy', 'NumPy')
        ]
        
        for module, name in dependencies:
            try:
                __import__(module)
                self._log_error("INFO", f"{name} import: SUCCESS")
            except Exception as e:
                # 宽松处理：仅记录日志，不抛出异常，避免如cv2部分初始化导致的AttributeError阻断
                self._log_error("ERROR", f"{name} import: FAILED - {str(e)}")

    def _delayed_model_loading(self, event):
        """延迟加载模型，避免阻塞节点初始化"""
        rospy.loginfo("Starting delayed VLM model loading...")
        self._log_error("INFO", "VLM model loading started")
        try:
            self._load_model()
            self.model_loaded = True
            rospy.loginfo("VLM model loaded successfully!")
            self._log_error("SUCCESS", "VLM model loaded successfully")
            self.last_error_message = None
        except Exception as e:
            rospy.logerr("Failed to load VLM model: %s", e)
            self._log_error("ERROR", f"VLM model loading failed: {str(e)}")
            self.model_loaded = False
            self.last_error_message = str(e)
        
        rospy.loginfo("Model loading completed. Status: %s", "LOADED" if self.model_loaded else "FAILED")

    def _log_error(self, level, message):
        """发布详细错误日志到话题"""
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        log_msg = f"[{timestamp}] [{level}] VLM_MODEL: {message}"
        self.error_log_pub.publish(StringMsg(data=log_msg))

    def _load_model(self):
        """Load model from local directory. Import heavy deps lazily so node can run without them."""
        try:
            self._log_error("INFO", "Importing ML dependencies...")
            
            # 逐个检测关键依赖
            try:
                import torch
                self._log_error("INFO", f"PyTorch version: {torch.__version__}")
                self._log_error("INFO", f"CUDA available: {torch.cuda.is_available()}")
                if torch.cuda.is_available():
                    self._log_error("INFO", f"CUDA device count: {torch.cuda.device_count()}")
            except ImportError as e:
                self._log_error("ERROR", f"PyTorch import failed: {str(e)}")
                raise
            
            try:
                import safetensors
                self._log_error("INFO", f"SafeTensors version: {safetensors.__version__}")
                # 显式检测 safetensors.torch 子模块
                try:
                    import safetensors.torch as st_torch  # noqa: F401
                    self._log_error("INFO", "SafeTensors Torch submodule: AVAILABLE")
                except ImportError as ee:
                    msg = (
                        f"SafeTensors Torch submodule missing: {ee}. "
                        "Please install/repair safetensors with: `pip install -U safetensors` "
                        "(ensure it matches your PyTorch/CUDA build)."
                    )
                    self._log_error("ERROR", msg)
                    raise
            except ImportError as e:
                self._log_error("ERROR", f"SafeTensors import failed: {str(e)}")
                raise
            
            try:
                import transformers
                self._log_error("INFO", f"Transformers version: {transformers.__version__}")
                # 预导入 generation.utils 以满足部分版本对 GenerationMixin 的懒加载依赖
                try:
                    import transformers.generation.utils as _gen_utils  # noqa: F401
                    self._log_error("INFO", "Pre-import transformers.generation.utils: SUCCESS")
                except Exception as ge:
                    self._log_error("WARNING", f"Pre-import transformers.generation.utils failed: {ge}")

                # 优先使用子模块路径导入，兼容4.55的导出行为
                try:
                    from transformers.models.qwen2_5_vl import (
                        Qwen2_5_VLProcessor as AutoProcessor,
                        Qwen2_5_VLForConditionalGeneration,
                    )
                    self._log_error("INFO", "Imported Qwen2_5_VL classes from submodule path")
                except Exception as sub_e:
                    self._log_error("WARNING", f"Submodule import failed: {sub_e}")
                    # 回退1：尝试顶层命名
                    try:
                        from transformers import Qwen2_5_VLProcessor as AutoProcessor, Qwen2_5_VLForConditionalGeneration
                        self._log_error("INFO", "Imported Qwen2_5_VL classes from top-level namespace")
                    except Exception as top_e:
                        self._log_error("WARNING", f"Top-level import failed: {top_e}")
                        # 回退2：兼容旧命名（Qwen2VL）
                        try:
                            from transformers.models.qwen2_vl import (
                                Qwen2VLProcessor as AutoProcessor,
                                Qwen2VLForConditionalGeneration as Qwen2_5_VLForConditionalGeneration,
                            )
                            self._log_error("INFO", "Imported Qwen2VL classes as fallback")
                        except Exception as old_e:
                            self._log_error("ERROR", f"Transformers import failed: {old_e}")
                            raise
            except ImportError as e:
                self._log_error("ERROR", f"Transformers import failed: {str(e)}")
                raise
                
            self._log_error("INFO", "All ML dependencies imported successfully")
        except Exception as e:
            self._log_error("ERROR", f"ML dependencies import failed: {str(e)}")
            raise RuntimeError("ML dependencies not available: {}".format(e))

        default_model_path = os.path.join(os.path.dirname(__file__), "..", "models", "Qwen2.5-VL-7B-Instruct")
        model_dir = rospy.get_param("~model_path", default_model_path)
        self._log_error("INFO", f"Model path: {model_dir}")
        
        # 检查模型文件是否存在
        if not os.path.exists(model_dir):
            self._log_error("ERROR", f"Model directory not found: {model_dir}")
            # 列出父目录内容以帮助调试
            parent_dir = os.path.dirname(model_dir)
            if os.path.exists(parent_dir):
                try:
                    contents = os.listdir(parent_dir)
                    self._log_error("INFO", f"Parent directory contents: {contents}")
                except Exception as e:
                    self._log_error("ERROR", f"Cannot list parent directory: {str(e)}")
            raise RuntimeError(f"Model directory not found: {model_dir}")
        
        # 检查模型目录内容
        try:
            model_files = os.listdir(model_dir)
            self._log_error("INFO", f"Model directory contains {len(model_files)} files")
            key_files = [f for f in model_files if f.endswith(('.json', '.safetensors', '.bin'))]
            self._log_error("INFO", f"Key model files: {key_files[:10]}...")  # 只显示前10个
        except Exception as e:
            self._log_error("ERROR", f"Cannot list model directory: {str(e)}")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if device == "cuda" else torch.float32
        self._log_error("INFO", f"Using device: {device}, dtype: {torch_dtype}")

        try:
            # Load processor and model
            self._log_error("INFO", "Loading processor...")
            import psutil
            mem_before = psutil.virtual_memory().percent
            self._log_error("INFO", f"Memory usage before processor loading: {mem_before}%")
            
            self._processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
            self._log_error("INFO", "Processor loaded successfully")
            
            mem_after_processor = psutil.virtual_memory().percent
            self._log_error("INFO", f"Memory usage after processor: {mem_after_processor}%")
            
            self._log_error("INFO", "Loading model...")
            self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_dir,
                torch_dtype=torch_dtype,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            mem_after_model = psutil.virtual_memory().percent
            self._log_error("INFO", f"Memory usage after model loading: {mem_after_model}%")
            
            self._log_error("INFO", "Model loaded, moving to device...")
            # 若使用了 accelerate 的 device_map（例如 device_map="auto"），则不要显式 .to(device)
            # 否则会触发：RuntimeError: You can't move a model that has some modules offloaded to cpu or disk.
            try:
                has_device_map = bool(getattr(self._model, "hf_device_map", None) or getattr(self._model, "device_map", None))
            except Exception:
                has_device_map = False

            if has_device_map:
                self._log_error("INFO", "Accelerate device_map detected; skip explicit .to(device)")
            else:
                self._model.to(device)

            self._model.eval()
            
            mem_final = psutil.virtual_memory().percent
            self._log_error("INFO", f"Final memory usage: {mem_final}%")
            self._log_error("INFO", "Model setup completed")
        except Exception as e:
            self._log_error("ERROR", f"Model loading step failed: {str(e)}")
            # 记录当前内存状态
            try:
                import psutil
                mem_current = psutil.virtual_memory().percent
                self._log_error("ERROR", f"Memory usage at failure: {mem_current}%")
            except:
                pass
            raise

    def _process_with_llm(self, instruction: str):
        """使用聊天模板与参数化生成解析指令为 JSON。"""
        import torch
        import time

        max_new, do_sample, temperature, use_examples = self._get_gen_params()
        messages = self._build_messages(instruction, use_examples=use_examples)

        tokenizer = getattr(self._processor, "tokenizer", None)
        if tokenizer is None:
            rospy.logwarn("Processor has no tokenizer; falling back to processor.decode")

        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self._model.device)

        # 生成参数
        gen_kwargs = {
            "max_new_tokens": int(max_new),
            "do_sample": bool(do_sample),
        }
        if do_sample:
            try:
                gen_kwargs["temperature"] = float(temperature)
            except Exception:
                pass

        # 获取 eos/pad
        eos_id = getattr(getattr(self._model, "generation_config", None), "eos_token_id", None)
        pad_id = getattr(getattr(self._model, "generation_config", None), "pad_token_id", eos_id)
        if eos_id is not None:
            gen_kwargs["eos_token_id"] = eos_id
        if pad_id is not None:
            gen_kwargs["pad_token_id"] = pad_id

        t0 = time.time()
        with torch.no_grad():
            outputs = self._model.generate(inputs, **gen_kwargs)
        t1 = time.time()
        self._log_error("INFO", f"LLM generate latency: {t1 - t0:.2f}s")

        # 仅解码新生成tokens
        gen_ids = outputs[0][inputs.shape[1]:]
        if tokenizer is not None:
            response = tokenizer.decode(gen_ids, skip_special_tokens=True)
        else:
            response = self._processor.decode(gen_ids, skip_special_tokens=True)

        # 直接尝试解析 JSON，否则提取
        try:
            return json.loads(response.strip())
        except json.JSONDecodeError:
            rospy.logwarn("JSON decode failed, attempting extraction")
            return self._extract_json(response)

    def _extract_json(self, text: str):
        """Extract JSON from text response"""
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if not match:
            return []
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return []

    def _get_gen_params(self):
        """从 ROS 参数读取生成配置，提供默认值。"""
        try:
            max_new = rospy.get_param("~llm_max_new_tokens", 128)
            do_sample = rospy.get_param("~llm_do_sample", False)
            temperature = rospy.get_param("~llm_temperature", 0.7)
            use_examples = rospy.get_param("~llm_use_examples", True)
        except Exception:
            max_new, do_sample, temperature, use_examples = 128, False, 0.7, True
        return max_new, do_sample, temperature, use_examples

    def _build_messages(self, instruction: str, use_examples: bool = True):
        """构建符合 Qwen 聊天模板的 messages。"""
        sys_preamble = (
            "You are a navigation instruction parser. Convert instructions to a JSON array of movement steps. "
            "Direction options: forward, backward, left, right. "
            "Goal should be a specific target if mentioned, otherwise null. "
            "Response must be JSON only, no additional text."
        )
        messages = [{"role": "system", "content": [{"type": "text", "text": sys_preamble}]}]

        if use_examples:
            examples = [
                {"input": "move forward to the tree, turn right, go straight and stop at the traffic cone",
                 "output": [{"subtask_1": "forward", "goal": "tree"},
                            {"subtask_2": "right", "goal": None},
                            {"subtask_3": "forward", "goal": "traffic cone"}]},
                {"input": "turn back, go straight to the tree, turn right, move until reach the bench",
                 "output": [{"subtask_1": "backward", "goal": None},
                            {"subtask_2": "forward", "goal": "tree"},
                            {"subtask_3": "right", "goal": None},
                            {"subtask_4": "forward", "goal": "bench"}]},
                {"input": "head to your right hand side and go to the bench",
                 "output": [{"subtask_1": "right", "goal": None},
                            {"subtask_2": "forward", "goal": "bench"}]}
            ]
            ex_text = "Examples:\n" + "\n\n".join([
                f"Instruction: {ex['input']}\nOutput: {json.dumps(ex['output'])}" for ex in examples
            ])
            messages.append({"role": "user", "content": [{"type": "text", "text": ex_text}]})

        messages.append({"role": "user", "content": [{"type": "text", "text": instruction}]})
        return messages

    def _fallback_parse(self, instruction: str):
        """Fallback heuristic parser"""
        # 方向关键字优先级：先匹配更具体/否定性的，再匹配泛化的forward，避免“go backward”被“go”误判为forward
        dir_map = {
            "backward": ["turn back", "backward", "turn around", "back"],
            "left": ["turn left", "left side", "left"],
            "right": ["turn right", "right hand side", "right side", "right"],
            # forward 不再包含过度泛化的 "go"/"move"，以减少歧义
            "forward": ["forward", "straight", "ahead", "move until"],
        }

        # Find direction-target pairs
        pairs = []
        subtask_counter = 1

        # Process instruction in sections
        sections = re.split(r'[,;]\s*|\s+and\s+|\s+then\s+', instruction)
        for section in sections:
            section = section.lower().strip()
            if not section:
                continue

            current_direction = None
            # Identify direction（按优先级顺序匹配，先匹配更具体的）
            for direction in ["backward", "left", "right", "forward"]:
                keywords = dir_map[direction]
                if any(keyword in section for keyword in keywords):
                    current_direction = direction
                    break

            # Identify target
            current_target = None
            if current_direction:
                # A simple way to extract a target is to remove keywords and see what's left
                temp_section = section
                for keyword in dir_map[current_direction]:
                    temp_section = temp_section.replace(keyword, "")
                current_target = temp_section.strip(" .")

                pairs.append({
                    f"subtask_{subtask_counter}": current_direction,
                    "goal": current_target if current_target else None
                })
                subtask_counter += 1

        return pairs if pairs else [
            {"subtask_1": "forward", "goal": "destination"},
            {"subtask_2": "stop", "goal": None}
        ]

    def _process_vlm_query(self, query_data):
        """Process VLM query about ArUco markers and target objects"""
        import torch
        query = query_data.get("query", "")
        aruco_markers = query_data.get("aruco_markers", [])

        if self.latest_cv_image is None:
            rospy.logwarn("No image received, cannot process VLM query")
            return {"error": "No image available"}

        # Prepare inputs for the model
        prompt = f'Analyze the image to answer the following query: "{query}". '
        prompt += f'Consider the detected ArUco markers at these locations: {json.dumps(aruco_markers)}. '
        prompt += 'Is the target object visible near any of these markers? Respond in JSON format with fields: "target_found" (boolean), "confidence" (float), and "target_description" (string).'

        # Convert OpenCV BGR image to PIL RGB image
        try:
            import cv2
            rgb_image = cv2.cvtColor(self.latest_cv_image, cv2.COLOR_BGR2RGB)
            pil_image = PILImage.fromarray(rgb_image)
        except Exception as e:
            rospy.logerr(f"Failed to convert image: {e}")
            return {"error": "Image conversion failed"}

        # Process with VLM（保持现有输入方式，但统一生成/解码与参数化）
        import time
        max_new, do_sample, temperature, _use_examples = self._get_gen_params()
        inputs = self._processor(text=prompt, images=[pil_image], return_tensors="pt").to(self._model.device)
        gen_kwargs = {
            "max_new_tokens": int(max_new),
            "do_sample": bool(do_sample),
        }
        if do_sample:
            try:
                gen_kwargs["temperature"] = float(temperature)
            except Exception:
                pass
        t0 = time.time()
        with torch.no_grad():
            outputs = self._model.generate(**inputs, **gen_kwargs)
        t1 = time.time()
        self._log_error("INFO", f"VLM generate latency: {t1 - t0:.2f}s")

        tokenizer = getattr(self._processor, "tokenizer", None)
        if tokenizer is not None:
            response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        else:
            response_text = self._processor.decode(outputs[0], skip_special_tokens=True)

        # Extract and return JSON
        try:
            # Extract JSON part from the response
            json_response = self._extract_json(response_text)
            if isinstance(json_response, list) and json_response:
                return json_response[0] # Assuming the first element is the desired dict
            elif isinstance(json_response, dict):
                return json_response
            else:
                return {"error": "Failed to parse VLM response", "raw_response": response_text}
        except Exception as e:
            rospy.logerr(f"Error processing VLM response: {e}")
            return {"error": "VLM response processing failed", "raw_response": response_text}


def main():
    rospy.init_node("instruction_processor", anonymous=False)
    node = InstructionProcessorNode()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Node shutdown")


if __name__ == "__main__":
    main()