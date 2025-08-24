#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import json
import re
from std_msgs.msg import String, Bool
from threading import Lock
import time

class EnglishInstructionProcessor:
    """Enhanced instruction processor specifically for English instructions in competition"""
    
    def __init__(self):
        rospy.init_node('english_instruction_processor', anonymous=True)
        rospy.loginfo("Enhanced English Instruction Processor initializing...")
        
        # Publishers
        self.subtasks_pub = rospy.Publisher('/subtasks', String, queue_size=1)
        self.status_pub = rospy.Publisher('/EIP_Status', Bool, queue_size=1)
        
        # Subscribers
        self.instruction_sub = rospy.Subscriber('/instruction', String, self.instruction_callback, queue_size=1)
        
        # Thread safety
        self.processing_lock = Lock()
        
        # Build vocabulary and pattern databases
        self.build_navigation_vocabulary()
        self.build_pattern_database()
        
        # State
        self.last_instruction = ""
        self.processing_time_limit = 2.0  # 2 seconds max processing time
        
        rospy.loginfo("Enhanced English Instruction Processor ready")
    
    def build_navigation_vocabulary(self):
        """Build comprehensive English navigation vocabulary"""
        
        # Movement directions
        self.direction_keywords = {
            'forward': [
                'forward', 'ahead', 'straight', 'onward', 'continue', 
                'proceed', 'advance', 'move forward', 'go forward',
                'go straight', 'keep going', 'march forward'
            ],
            'backward': [
                'backward', 'back', 'reverse', 'retreat', 'return',
                'go back', 'move back', 'step back', 'backwards'
            ],
            'left': [
                'left', 'turn left', 'go left', 'head left', 'veer left',
                'bear left', 'swing left', 'pivot left', 'rotate left'
            ],
            'right': [
                'right', 'turn right', 'go right', 'head right', 'veer right',
                'bear right', 'swing right', 'pivot right', 'rotate right'
            ]
        }
        
        # Action verbs
        self.action_keywords = {
            'navigate': [
                'go', 'move', 'travel', 'proceed', 'advance', 'head',
                'walk', 'drive', 'navigate', 'approach', 'reach'
            ],
            'turn': [
                'turn', 'rotate', 'pivot', 'swing', 'veer', 'bear',
                'curve', 'bend', 'angle'
            ],
            'stop': [
                'stop', 'halt', 'pause', 'wait', 'rest', 'end',
                'finish', 'arrive', 'reach'
            ],
            'search': [
                'find', 'look', 'search', 'locate', 'seek', 'spot',
                'identify', 'detect', 'discover'
            ]
        }
        
        # Landmark/target objects (competition specific)
        self.landmark_keywords = {
            'tree': [
                'tree', 'trees', 'oak', 'pine', 'plant', 'vegetation'
            ],
            'cone': [
                'cone', 'traffic cone', 'orange cone', 'pylon', 'marker cone'
            ],
            'wall': [
                'wall', 'brick wall', 'barrier', 'fence', 'boundary'
            ],
            'marker': [
                'marker', 'aruco', 'marker board', 'target', 'sign',
                'board', 'panel', 'square'
            ],
            'barrel': [
                'barrel', 'drum', 'container', 'cylinder'
            ],
            'bench': [
                'bench', 'seat', 'table'
            ],
            'hydrant': [
                'hydrant', 'fire hydrant', 'water hydrant'
            ],
            'trash': [
                'trash', 'bin', 'garbage', 'waste', 'trash bin'
            ],
            'truck': [
                'truck', 'trailer', 'vehicle', 'tractor'
            ]
        }
        
        # Spatial relationships
        self.spatial_keywords = {
            'near': ['near', 'close to', 'next to', 'beside', 'by', 'around'],
            'far': ['far', 'distant', 'away from', 'beyond'],
            'between': ['between', 'among', 'in between'],
            'in_front': ['in front of', 'before', 'ahead of'],
            'behind': ['behind', 'after', 'past'],
            'at': ['at', 'to', 'toward', 'towards']
        }
        
        # Distance/magnitude modifiers
        self.magnitude_keywords = {
            'short': ['short', 'brief', 'quick', 'little', 'small'],
            'long': ['long', 'far', 'extended', 'big', 'large'],
            'very': ['very', 'quite', 'extremely', 'really', 'super']
        }
    
    def build_pattern_database(self):
        """Build common instruction patterns for faster parsing"""
        
        self.instruction_patterns = [
            # Basic movement patterns
            {
                'pattern': r'(go|move|proceed)\s+(forward|straight|ahead)',
                'template': {'action': 'navigate', 'direction': 'forward'}
            },
            {
                'pattern': r'turn\s+(left|right)',
                'template': {'action': 'turn', 'direction': '{direction}'}
            },
            {
                'pattern': r'(go|move)\s+(left|right)',
                'template': {'action': 'navigate', 'direction': '{direction}'}
            },
            
            # Goal-oriented patterns
            {
                'pattern': r'(go|move|navigate)\s+to\s+(?:the\s+)?(\w+)',
                'template': {'action': 'navigate', 'goal': '{goal}'}
            },
            {
                'pattern': r'(find|locate|search for)\s+(?:the\s+)?(\w+)',
                'template': {'action': 'search', 'goal': '{goal}'}
            },
            {
                'pattern': r'(approach|reach)\s+(?:the\s+)?(\w+)',
                'template': {'action': 'navigate', 'goal': '{goal}'}
            },
            
            # Stopping patterns
            {
                'pattern': r'stop\s+(?:at|near|by)\s+(?:the\s+)?(\w+)',
                'template': {'action': 'navigate', 'goal': '{goal}', 'final': True}
            },
            
            # Complex patterns with direction and goal
            {
                'pattern': r'(go|move)\s+(forward|straight|ahead)\s+(?:to|toward|until)\s+(?:the\s+)?(\w+)',
                'template': {'action': 'navigate', 'direction': 'forward', 'goal': '{goal}'}
            },
            {
                'pattern': r'turn\s+(left|right)\s+(?:at|near|by)\s+(?:the\s+)?(\w+)',
                'template': {'action': 'turn', 'direction': '{direction}', 'goal': '{goal}'}
            }
        ]
    
    def instruction_callback(self, msg):
        """Process incoming English instruction"""
        with self.processing_lock:
            start_time = time.time()
            instruction = msg.data.strip().lower()
            
            if instruction == self.last_instruction:
                rospy.loginfo("Duplicate instruction, skipping")
                return
            
            self.last_instruction = instruction
            rospy.loginfo(f"Processing English instruction: '{instruction}'")
            
            try:
                # Multi-strategy parsing
                subtasks = self.parse_instruction_multiway(instruction)
                
                if subtasks:
                    # Publish subtasks
                    subtasks_msg = String()
                    subtasks_msg.data = json.dumps(subtasks)
                    self.subtasks_pub.publish(subtasks_msg)
                    
                    # Publish status
                    status_msg = Bool()
                    status_msg.data = True
                    self.status_pub.publish(status_msg)
                    
                    processing_time = time.time() - start_time
                    rospy.loginfo(f"Parsed into {len(subtasks)} subtasks in {processing_time:.3f}s")
                    rospy.loginfo(f"Subtasks: {subtasks}")
                    
                else:
                    rospy.logwarn("Failed to parse instruction")
                    self.publish_failure()
                    
            except Exception as e:
                rospy.logerr(f"Error processing instruction: {e}")
                self.publish_failure()
    
    def parse_instruction_multiway(self, instruction):
        """Use multiple parsing strategies and vote on results"""
        
        results = []
        
        # Strategy 1: Pattern matching
        pattern_result = self.parse_with_patterns(instruction)
        if pattern_result:
            results.append(('pattern', pattern_result))
        
        # Strategy 2: Keyword extraction
        keyword_result = self.parse_with_keywords(instruction)
        if keyword_result:
            results.append(('keyword', keyword_result))
        
        # Strategy 3: Sentence structure analysis
        structure_result = self.parse_with_structure(instruction)
        if structure_result:
            results.append(('structure', structure_result))
        
        # Strategy 4: Simplified fallback
        fallback_result = self.parse_fallback(instruction)
        if fallback_result:
            results.append(('fallback', fallback_result))
        
        # Voting and combination
        if results:
            return self.combine_parsing_results(results)
        else:
            return None
    
    def parse_with_patterns(self, instruction):
        """Parse using predefined patterns"""
        
        for pattern_def in self.instruction_patterns:
            pattern = pattern_def['pattern']
            template = pattern_def['template']
            
            match = re.search(pattern, instruction, re.IGNORECASE)
            if match:
                # Fill template with matched groups
                subtask = template.copy()
                
                # Replace placeholders with matched groups
                for key, value in subtask.items():
                    if isinstance(value, str) and '{' in value:
                        if '{direction}' in value:
                            if len(match.groups()) >= 1:
                                subtask[key] = match.group(1)
                        elif '{goal}' in value:
                            if len(match.groups()) >= 2:
                                goal = self.normalize_landmark(match.group(2))
                                subtask[key] = goal if goal else match.group(2)
                            elif len(match.groups()) >= 1:
                                goal = self.normalize_landmark(match.group(1))
                                subtask[key] = goal if goal else match.group(1)
                
                return [subtask]
        
        return None
    
    def parse_with_keywords(self, instruction):
        """Parse by extracting keywords"""
        
        # Tokenize instruction
        tokens = re.findall(r'\b\w+\b', instruction.lower())
        
        # Extract actions
        actions = []
        for action, keywords in self.action_keywords.items():
            if any(keyword in instruction for keyword in keywords):
                actions.append(action)
        
        # Extract directions
        directions = []
        for direction, keywords in self.direction_keywords.items():
            if any(keyword in instruction for keyword in keywords):
                directions.append(direction)
        
        # Extract landmarks
        landmarks = []
        for landmark, keywords in self.landmark_keywords.items():
            if any(keyword in instruction for keyword in keywords):
                landmarks.append(landmark)
        
        # Build subtasks from extracted keywords
        subtasks = []
        
        if actions or directions:
            action = actions[0] if actions else 'navigate'
            
            if directions and landmarks:
                # Both direction and landmark specified
                subtask = {
                    'action': action,
                    'direction': directions[0],
                    'goal': landmarks[0]
                }
                subtasks.append(subtask)
            elif directions:
                # Only direction specified
                subtask = {
                    'action': action,
                    'direction': directions[0]
                }
                subtasks.append(subtask)
            elif landmarks:
                # Only landmark specified
                subtask = {
                    'action': action,
                    'goal': landmarks[0]
                }
                subtasks.append(subtask)
        
        return subtasks if subtasks else None
    
    def parse_with_structure(self, instruction):
        """Parse by analyzing sentence structure"""
        
        # Split instruction into clauses
        clauses = re.split(r'[,;]\s*(?:and\s+)?(?:then\s+)?', instruction)
        subtasks = []
        
        for clause in clauses:
            clause = clause.strip()
            if not clause:
                continue
            
            # Analyze each clause
            subtask = self.analyze_clause(clause)
            if subtask:
                subtasks.append(subtask)
        
        return subtasks if subtasks else None
    
    def analyze_clause(self, clause):
        """Analyze a single clause to extract action and parameters"""
        
        # Remove common prefixes and suffixes
        clause = re.sub(r'^(please\s+|now\s+|then\s+)', '', clause)
        clause = re.sub(r'(please|thank you)$', '', clause)
        
        # Default subtask
        subtask = {'action': 'navigate'}
        
        # Extract action
        for action, keywords in self.action_keywords.items():
            if any(re.search(r'\b' + re.escape(keyword) + r'\b', clause) for keyword in keywords):
                subtask['action'] = action
                break
        
        # Extract direction
        for direction, keywords in self.direction_keywords.items():
            if any(re.search(r'\b' + re.escape(keyword) + r'\b', clause) for keyword in keywords):
                subtask['direction'] = direction
                break
        
        # Extract goal/landmark
        for landmark, keywords in self.landmark_keywords.items():
            if any(re.search(r'\b' + re.escape(keyword) + r'\b', clause) for keyword in keywords):
                subtask['goal'] = landmark
                break
        
        # Check for stop/final conditions
        if any(word in clause for word in ['stop', 'halt', 'end', 'finish']):
            subtask['final'] = True
        
        return subtask
    
    def parse_fallback(self, instruction):
        """Simplified fallback parsing"""
        
        # Very basic parsing as last resort
        subtask = {'action': 'navigate'}
        
        if 'forward' in instruction or 'straight' in instruction:
            subtask['direction'] = 'forward'
        elif 'left' in instruction:
            subtask['direction'] = 'left'
        elif 'right' in instruction:
            subtask['direction'] = 'right'
        elif 'back' in instruction:
            subtask['direction'] = 'backward'
        
        # Check for common landmarks
        if 'tree' in instruction:
            subtask['goal'] = 'tree'
        elif 'cone' in instruction:
            subtask['goal'] = 'cone'
        elif 'marker' in instruction:
            subtask['goal'] = 'marker'
        elif 'wall' in instruction:
            subtask['goal'] = 'wall'
        
        return [subtask]
    
    def combine_parsing_results(self, results):
        """Combine results from multiple parsing strategies"""
        
        if len(results) == 1:
            return results[0][1]
        
        # Simple combination: prefer pattern matching, then keyword extraction
        for strategy, result in results:
            if strategy == 'pattern' and result:
                return result
        
        for strategy, result in results:
            if strategy == 'keyword' and result:
                return result
        
        # Fallback to first available result
        return results[0][1] if results else None
    
    def normalize_landmark(self, landmark):
        """Normalize landmark name to standard vocabulary"""
        
        landmark = landmark.lower().strip()
        
        for standard_name, variants in self.landmark_keywords.items():
            if landmark in variants:
                return standard_name
        
        return landmark
    
    def publish_failure(self):
        """Publish failure status"""
        status_msg = Bool()
        status_msg.data = False
        self.status_pub.publish(status_msg)

if __name__ == '__main__':
    try:
        processor = EnglishInstructionProcessor()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
