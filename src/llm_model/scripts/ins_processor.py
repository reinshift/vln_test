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
        self.require_vlm_ready = bool(rospy.get_param('~require_vlm_ready', True))
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
        # Response wait timeout for /vlm_response
        self.response_timeout_sec = float(rospy.get_param('~vlm_response_timeout_sec', 8.0))

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
        self.pending_mode = None  # 'online' | 'offline'
        self._pending_timer = None  # rospy.Timer
        # Dedupe recent instruction
        self.last_instruction = None
        self.last_instruction_time = 0.0

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
        # If model just became ready and we have a pending instruction not yet sent, send it now
        if self.vlm_status and self.pending_request_id is None and self.pending_instruction:
            self._log('INFO', 'VLM became ready; sending buffered instruction')
            self._send_instruction_to_vlm(self.pending_instruction)

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

        # Simple dedupe within 5 seconds if identical instruction already processed or pending
        now = time.time()
        if (self.last_instruction == instruction) and (now - self.last_instruction_time < 5.0):
            self._log('INFO', 'Duplicate instruction received within 5s window; ignoring')
            return
        self.last_instruction = instruction
        self.last_instruction_time = now

        # Wait for VLM ready transition to True (configurable)
        ok = self._wait_vlm_transition_true(self.wait_vlm_ready_sec)
        if not ok and self.require_vlm_ready:
            # Fallback: trigger offline parsing via vlm_loader to avoid blocking downstream
            self._log('WARNING', f'VLM not ready within {self.wait_vlm_ready_sec}s; falling back to offline parsing')
            request_id = str(uuid.uuid4())
            payload = {
                'request_id': request_id,
                'type': 'instruction_offline',
                'text': instruction,
                'need_image': False,
            }
            self.pending_request_id = request_id
            self.pending_instruction = instruction
            self.pending_sent_time = time.time()
            self.pending_mode = 'offline'
            self.query_pub.publish(StringMsg(data=json.dumps(payload)))
            self._log('INFO', f'Sent offline /vlm_query request_id={request_id}')
            self._start_response_timer()
            return

        # Send now (either ready or not required to be ready)
        self._send_instruction_to_vlm(instruction)

    def _send_instruction_to_vlm(self, instruction: str):
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
        self.pending_mode = 'online'
        self.query_pub.publish(StringMsg(data=json.dumps(payload)))
        self._log('INFO', f'Sent /vlm_query request_id={request_id}')
        self._start_response_timer()

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
        if isinstance(data, dict) and data.get('error'):
            self._log('WARNING', f"VLM response error: {data.get('error')} (reason={data.get('reason', '')})")
            return
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

        # Filter out forward-without-object subtasks per requirement
        filtered = self._filter_forward_none(subtasks)
        if len(filtered) != len(subtasks):
            self._log('INFO', f'Filtered forward-without-goal subtasks: {len(subtasks) - len(filtered)} removed')
        subtasks = filtered

        if len(subtasks) == 0:
            # Do not publish empty subtasks; log and clear pending to avoid soft-deadlock
            self._log('WARNING', 'Empty subtasks parsed after filtering; not publishing /subtasks (clearing pending)')
            self._clear_pending()
            return

        # Publish downstream
        self.subtasks_pub.publish(StringMsg(data=json.dumps(subtasks)))
        self._log('INFO', f'Published subtasks (N={len(subtasks)}) for request_id={rid}')

        # Clear pending
        self._clear_pending()

    def _extract_json(self, text: str):
        m = re.search(r'\[.*\]', text, re.DOTALL)
        if not m:
            return []
        try:
            return json.loads(m.group(0))
        except Exception:
            return []

    def _filter_forward_none(self, subtasks):
        """
        过滤掉：方向为 'forward' 且 object/goal 为空(None/""/"null") 的子任务。
        兼容多种字段：
          - 方向键：任一包含 'subtask' 的键，或 'direction'
          - 目标键：'goal' 或 'object'
        """
        def _get_dir_and_goal(item):
            if not isinstance(item, dict):
                return None, None
            # direction
            direction = None
            for k, v in item.items():
                if isinstance(k, str) and 'subtask' in k.lower():
                    direction = str(v).strip().lower() if v is not None else None
                    break
            if direction is None:
                v = item.get('direction')
                direction = str(v).strip().lower() if v is not None else None
            # goal/object
            goal = item.get('goal')
            if goal is None and 'object' in item:
                goal = item.get('object')
            # Normalize textual nulls
            if isinstance(goal, str):
                g = goal.strip().lower()
                if g in ('', 'null', 'none', 'nil'):  # treat as None
                    goal = None
            return direction, goal

        out = []
        for it in subtasks:
            direction, goal = _get_dir_and_goal(it)
            if direction == 'forward' and goal is None:
                continue
            out.append(it)
        return out

    def _start_response_timer(self):
        # cancel existing timer
        if self._pending_timer is not None:
            try:
                self._pending_timer.shutdown()
            except Exception:
                pass
            self._pending_timer = None
        if self.response_timeout_sec <= 0:
            return
        self._pending_timer = rospy.Timer(rospy.Duration(self.response_timeout_sec), self._on_response_timeout, oneshot=True)

    def _on_response_timeout(self, _):
        # If still pending, handle timeout
        if not self.pending_request_id:
            return
        mode = self.pending_mode or 'online'
        instr = self.pending_instruction or ''
        if mode == 'online':
            # Fallback to offline once
            self._log('WARNING', f'Response timeout for online request_id={self.pending_request_id}; falling back to offline parsing')
            request_id = str(uuid.uuid4())
            payload = {
                'request_id': request_id,
                'type': 'instruction_offline',
                'text': instr,
                'need_image': False,
            }
            self.pending_request_id = request_id
            self.pending_sent_time = time.time()
            self.pending_mode = 'offline'
            self.query_pub.publish(StringMsg(data=json.dumps(payload)))
            self._log('INFO', f'Sent offline /vlm_query (timeout fallback) request_id={request_id}')
            self._start_response_timer()
        else:
            # Offline also timed out -> log and clear pending
            self._log('ERROR', f'Response timeout for offline request_id={self.pending_request_id}; giving up and clearing pending')
            self._clear_pending()

    def _clear_pending(self):
        self.pending_request_id = None
        self.pending_instruction = None
        self.pending_sent_time = None
        self.pending_mode = None
        if self._pending_timer is not None:
            try:
                self._pending_timer.shutdown()
            except Exception:
                pass
            self._pending_timer = None


def main():
    rospy.init_node('ins_processor', anonymous=False)
    _ = InstructionProcessorLite()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main()
