#!/catkin_ws/venv310/bin/python3
# -*- coding: utf-8 -*-

import json
import re
import time
import uuid
import threading
import rospy
from std_msgs.msg import String as StringMsg, Bool

class InstructionProcessorLite:
    """
    职责：
    - 启动并持续监听（可能是 latched 的）/instruction
    - 订阅 /VLM_Status，若收到指令且需要等待 VLM 加载：等待其从 False->True（带超时参数）
    - 将带有系统提示词的文本请求封装为 /vlm_query（JSON，含 request_id），等待 /vlm_response
    - 收到对应响应后，将解析出的子任务发布到 /subtasks（latched）
    """
    def __init__(self):
        # Params
        self.wait_vlm_ready_sec = float(rospy.get_param('~wait_vlm_ready_sec', 15.0))
        self.system_prompt = rospy.get_param('~system_prompt', (
            'You are a navigation instruction subtask parser. Convert instructions to a JSON array. '\
            'Each array element is an object with keys: subtask_N (a direction string, one of "forward", "backward", "left", "right") '\
            'and goal (a string target or null). Start N from 1 and increment by 1. '\
            'Respond with JSON only, no extra text. '\
            '\n\nExamples:\n'\
            'Instruction: "move forward and stop at the tree"\n'\
            'Output: [\n'\
            '  {"subtask_1": "forward", "goal": "tree"}\n'\
            ']\n\n'\
            'Instruction: "head to your right hand side and go to the bench"\n'\
            'Output: [\n'\
            '  {"subtask_1": "right", "goal": null},\n'\
            '  {"subtask_2": "forward", "goal": "bench"}\n'\
            ']'
        ))
        # Active retry fetch config for /instruction
        self.instruction_wait_timeout = float(rospy.get_param('~instruction_wait_timeout', 2.0))  # each wait timeout
        self.instruction_retry_interval = float(rospy.get_param('~instruction_retry_interval', 0.5))  # pause between attempts

        # Pubs/Subs
        self.query_pub = rospy.Publisher('/vlm_query', StringMsg, queue_size=10)
        self.subtasks_pub = rospy.Publisher('/subtasks', StringMsg, queue_size=1, latch=True)
        self.error_log_pub = rospy.Publisher('/vlm_error_log', StringMsg, queue_size=10)

        self.vlm_status = False
        self.vlm_status_sub = rospy.Subscriber('/VLM_Status', Bool, self._on_vlm_status, queue_size=1)
        self.vlm_resp_sub = rospy.Subscriber('/vlm_response', StringMsg, self._on_vlm_response, queue_size=10)
        self.ins_sub = rospy.Subscriber('/instruction', StringMsg, self._on_instruction, queue_size=1)

        # State for correlating responses
        self.pending_request_id = None
        self.pending_instruction = None
        self.pending_sent_time = None

        # Received flag to stop retry worker once we get any instruction
        self._received_instruction_once = False

        # Start background retry worker to actively fetch instruction if none arrives (handles late publishers and latched)
        self._retry_thread = threading.Thread(target=self._instruction_retry_worker, name='ins_retry_worker', daemon=True)
        self._retry_thread.start()

        rospy.loginfo('ins_processor initialized.')

    def _log(self, level: str, message: str):
        import datetime
        ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        self.error_log_pub.publish(StringMsg(data=f'[{ts}] [{level}] INS_PROC: {message}'))

    def _on_vlm_status(self, msg: Bool):
        self.vlm_status = bool(getattr(msg, 'data', False))

    def _wait_vlm_transition_true(self, max_wait: float) -> bool:
        """等待 VLM 从 False -> True；若当前已 True 则直接返回 True。"""
        if self.vlm_status:
            return True
        start = rospy.Time.now()
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            if self.vlm_status:
                return True
            elapsed = (rospy.Time.now() - start).to_sec()
            if elapsed >= max_wait:
                return False
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                break
        return False

    def _on_instruction(self, msg: StringMsg):
        instruction = (msg.data or '').strip()
        if not instruction:
            rospy.logwarn('ins_processor: received empty instruction')
            return
        rospy.loginfo('ins_processor: got instruction: %s', instruction)
        self._log('INFO', f'Received instruction: {instruction}')
        self._received_instruction_once = True

        # Wait for VLM ready transition to True (configurable)
        ok = self._wait_vlm_transition_true(self.wait_vlm_ready_sec)
        if not ok:
            self._log('WARNING', f'VLM not ready within {self.wait_vlm_ready_sec}s; proceeding to request anyway')

        # Build prompt
        user_text = f'{self.system_prompt}\n\nInstruction: {instruction}'
        request_id = str(uuid.uuid4())
        payload = {
            'request_id': request_id,
            'type': 'instruction',
            'text': user_text,
            'need_image': False,
        }
        self.pending_request_id = request_id
        self.pending_instruction = instruction
        self.pending_sent_time = time.time()

        self.query_pub.publish(StringMsg(data=json.dumps(payload)))
        self._log('INFO', f'Sent /vlm_query request_id={request_id}')

    def _instruction_retry_worker(self):
        """Actively wait for /instruction if none received yet. Works with latched publishers and late publishers."""
        # Short initial delay to allow normal subscriber path first
        time.sleep(0.5)
        while not rospy.is_shutdown() and not self._received_instruction_once:
            try:
                msg = rospy.wait_for_message('/instruction', StringMsg, timeout=self.instruction_wait_timeout)
                if msg and not self._received_instruction_once:
                    # Reuse the same processing path
                    self._on_instruction(msg)
                    break
            except rospy.ROSException:
                # timeout -> retry after interval
                pass
            except Exception as e:
                self._log('WARNING', f'instruction retry worker error: {e}')
            time.sleep(self.instruction_retry_interval)

    def _on_vlm_response(self, msg: StringMsg):
        # Parse response JSON
        try:
            data = json.loads(msg.data)
        except Exception as e:
            self._log('ERROR', f'Invalid /vlm_response JSON: {e}')
            return

        rid = str(data.get('request_id', ''))
        if self.pending_request_id and rid and rid != self.pending_request_id:
            # Not for us
            return

        # Prefer structured {'subtasks': [...]}
        subtasks = data.get('subtasks')
        if subtasks is None:
            # Try to extract from text-like fields
            if isinstance(data.get('vision_result'), list):
                subtasks = []
            else:
                text = data if isinstance(data, str) else json.dumps(data)
                subtasks = self._extract_json(text)
        if not isinstance(subtasks, list):
            subtasks = []

        # Publish downstream
        self.subtasks_pub.publish(StringMsg(data=json.dumps(subtasks)))
        self._log('INFO', f'Published subtasks (N={len(subtasks)}) for request_id={rid}')

        # Clear pending
        self.pending_request_id = None
        self.pending_instruction = None
        self.pending_sent_time = None

    def _extract_json(self, text: str):
        m = re.search(r'\[.*\]', text, re.DOTALL)
        if not m:
            return []
        try:
            return json.loads(m.group(0))
        except Exception:
            return []


def main():
    rospy.init_node('ins_processor', anonymous=False)
    _ = InstructionProcessorLite()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main()
