#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import rospy
from std_msgs.msg import String as StringMsg, Bool
from sensor_msgs.msg import Image
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from cv_bridge import CvBridge
from PIL import Image as PILImage

class InstructionProcessorNode:
    def __init__(self):
        # Initialize ROS components
        self.sub = rospy.Subscriber("/instruction", StringMsg, self._on_instruction, queue_size=1)
        self.vlm_query_sub = rospy.Subscriber("/vlm_query", StringMsg, self._on_vlm_query, queue_size=1)
        self.status_pub = rospy.Publisher("/VLM_Status", Bool, queue_size=1)
        self.task_pub = rospy.Publisher("/subtasks", StringMsg, queue_size=1)
        self.vlm_response_pub = rospy.Publisher("/vlm_response", StringMsg, queue_size=1)
        self.image_sub = rospy.Subscriber("/camera/image_raw", Image, self._on_image, queue_size=1)

        # Model components
        self._model = None
        self._processor = None
        self.model_loaded = False
        self.bridge = CvBridge()
        self.latest_image = None

        # Load model at startup
        rospy.loginfo("Loading VLM model at startup...")
        try:
            self._load_model()
            self.model_loaded = True
            rospy.loginfo("VLM model loaded successfully!")
        except Exception as e:
            rospy.logerr("Failed to load VLM model: %s", e)
            self.model_loaded = False

        # Start status publisher timer
        self.status_timer = rospy.Timer(rospy.Duration(2.0), self._publish_status)
        rospy.loginfo("Instruction processor ready. Model status: %s", "LOADED" if self.model_loaded else "FAILED")

    def _publish_status(self, event):
        """Publish model status"""
        self.status_pub.publish(Bool(data=self.model_loaded))

    def _on_image(self, msg: Image):
        """Store the latest image"""
        self.latest_image = msg

    def _on_instruction(self, msg: StringMsg):
        """Process incoming instruction"""
        if not self.model_loaded:
            rospy.logerr("Model not loaded, skipping instruction")
            return

        instruction = (msg.data or "").strip()
        rospy.loginfo("Processing: %s", instruction)

        try:
            subtasks = self._process_with_llm(instruction)
        except Exception as e:
            rospy.logerr("LLM processing failed: %s", e)
            subtasks = self._fallback_parse(instruction)

        # Publish subtasks to new topic
        task_msg = StringMsg(data=json.dumps(subtasks))
        self.task_pub.publish(task_msg)
        rospy.loginfo("Published subtasks: %s", json.dumps(subtasks, indent=2))

    def _on_vlm_query(self, msg: StringMsg):
        """Process VLM query for ArUco marker analysis"""
        if not self.model_loaded:
            rospy.logerr("Model not loaded, skipping VLM query")
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

    def _load_model(self):
        """Load model from local directory"""
        default_model_path = os.path.join(os.path.dirname(__file__), "..", "models", "Qwen2.5-VL-7B-Instruct")
        model_dir = rospy.get_param("~model_path", default_model_path)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if device == "cuda" else torch.float32

        # Load processor and model
        self._processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_dir,
            torch_dtype=torch_dtype,
            device_map="auto" if torch.cuda.is_available() else None
        )
        self._model.to(device)
        self._model.eval()

    def _process_with_llm(self, instruction: str):
        """Process instruction with VL model"""
        # Improved prompt with examples
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

        # Construct prompt
        prompt = (
            "You are a navigation instruction parser. Convert instructions to a JSON array of movement steps. "
            "Direction options: forward, backward, left, right. "
            "Goal should be a specific target if mentioned, otherwise null. "
            "Response must be JSON only, no additional text.\n\n"
            "Examples:\n"
        )

        for ex in examples:
            prompt += f"Instruction: {ex['input']}\nOutput: {json.dumps(ex['output'])}\n\n"

        prompt += f"Instruction: {instruction}\nOutput:"

        # Generate response
        inputs = self._processor(text=prompt, padding=True, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=False,
                temperature=0.1
            )

        # Decode and parse response
        response = self._processor.decode(outputs[0], skip_special_tokens=True)
        json_str = response.split("Output:")[-1].strip()

        try:
            return json.loads(json_str)
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

    def _fallback_parse(self, instruction: str):
        """Fallback heuristic parser"""
        # Mapping of direction keywords
        # Mapping of direction keywords
        dir_map = {
            "forward": ["forward", "straight", "ahead", "move until", "go", "move"],
            "backward": ["turn back", "back", "backward", "turn around"],
            "left": ["left", "left side"],
            "right": ["right", "right side", "right hand side"], # Added "right hand side"
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
            # Identify direction
            for direction, keywords in dir_map.items():
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
        query = query_data.get("query", "")
        aruco_markers = query_data.get("aruco_markers", [])

        if self.latest_image is None:
            rospy.logwarn("No image received, cannot process VLM query")
            return {"error": "No image available"}

        # Prepare inputs for the model
        prompt = f'Analyze the image to answer the following query: "{query}". '
        prompt += f'Consider the detected ArUco markers at these locations: {json.dumps(aruco_markers)}. '
        prompt += 'Is the target object visible near any of these markers? Respond in JSON format with fields: "target_found" (boolean), "confidence" (float), and "target_description" (string).'

        # Convert ROS Image to something the model can use (e.g., PIL Image)
        try:
            cv_image = self.bridge.imgmsg_to_cv2(self.latest_image, desired_encoding="rgb8")
            pil_image = PILImage.fromarray(cv_image)
        except Exception as e:
            rospy.logerr(f"Failed to convert image: {e}")
            return {"error": "Image conversion failed"}

        # Process with VLM
        inputs = self._processor(text=prompt, images=[pil_image], return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            outputs = self._model.generate(**inputs, max_new_tokens=100, do_sample=False)

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