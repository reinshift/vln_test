#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import json
import math
import numpy as np
from std_msgs.msg import String
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import Odometry
from aruco_detector.msg import ArucoInfo, ArucoMarker
from magv_vln_msgs.msg import DetectedObjectArray, VehicleStatus
from threading import Lock
import time
from collections import defaultdict

class MarkerManager:
    """Enhanced marker management for multi-marker competition scenarios"""
    
    def __init__(self):
        rospy.init_node('marker_manager', anonymous=True)
        rospy.loginfo("Multi-Marker Manager initializing...")
        
        # Publishers
        self.target_marker_pub = rospy.Publisher('/target_marker', String, queue_size=1)
        self.marker_guidance_pub = rospy.Publisher('/marker_guidance', String, queue_size=1)
        
        # Subscribers
        self.aruco_sub = rospy.Subscriber('/aruco_info', ArucoInfo, self.aruco_callback, queue_size=1)
        self.objects_sub = rospy.Subscriber('/detected_objects', DetectedObjectArray, self.objects_callback, queue_size=1)
        self.odometry_sub = rospy.Subscriber('/magv/odometry/gt', Odometry, self.odometry_callback, queue_size=1)
        self.instruction_sub = rospy.Subscriber('/instruction', String, self.instruction_callback, queue_size=1)
        self.vln_status_sub = rospy.Subscriber('/vln_status', VehicleStatus, self.status_callback, queue_size=1)
        
        # Thread safety
        self.data_lock = Lock()
        
        # State
        self.current_pose = None
        self.current_instruction = ""
        self.task_active = False
        
        # Marker database
        self.detected_markers = {}  # {marker_id: MarkerInfo}
        self.marker_features = {}   # {marker_id: surrounding_features}
        self.marker_history = defaultdict(list)  # Historical detections
        
        # Spatial database
        self.spatial_relationships = {}  # Object relationships
        self.landmark_positions = {}     # Known landmark positions
        
        # Parameters
        self.marker_timeout = 30.0       # Marker considered stale after 30s
        self.confidence_threshold = 0.7  # Minimum confidence for marker selection
        self.spatial_tolerance = 2.0     # Spatial relationship tolerance (meters)
        
        rospy.loginfo("Multi-Marker Manager ready")
    
    def aruco_callback(self, msg):
        """Process ArUco marker detections"""
        with self.data_lock:
            current_time = rospy.Time.now()
            
            for marker in msg.markers:
                marker_id = marker.id
                
                # Create marker info
                marker_info = {
                    'id': marker_id,
                    'position': marker.pose.position,
                    'orientation': marker.pose.orientation,
                    'confidence': getattr(marker, 'confidence', 1.0),
                    'timestamp': current_time,
                    'detection_count': 1
                }
                
                # Update or add marker
                if marker_id in self.detected_markers:
                    # Update existing marker with weighted average
                    existing = self.detected_markers[marker_id]
                    existing['detection_count'] += 1
                    
                    # Weighted position update
                    weight = 0.3  # New detection weight
                    existing['position'].x = (1-weight) * existing['position'].x + weight * marker.pose.position.x
                    existing['position'].y = (1-weight) * existing['position'].y + weight * marker.pose.position.y
                    existing['position'].z = (1-weight) * existing['position'].z + weight * marker.pose.position.z
                    
                    existing['timestamp'] = current_time
                    existing['confidence'] = max(existing['confidence'], marker_info['confidence'])
                else:
                    self.detected_markers[marker_id] = marker_info
                
                # Add to history
                self.marker_history[marker_id].append({
                    'position': marker.pose.position,
                    'timestamp': current_time,
                    'robot_position': self.current_pose.position if self.current_pose else Point()
                })
                
                # Limit history size
                if len(self.marker_history[marker_id]) > 10:
                    self.marker_history[marker_id].pop(0)
            
            # Clean up stale markers
            self.cleanup_stale_markers()
            
            # Update spatial relationships
            self.update_spatial_relationships()
            
            # Analyze and publish guidance if task is active
            if self.task_active:
                self.analyze_and_publish_guidance()
    
    def objects_callback(self, msg):
        """Process detected objects for spatial context"""
        with self.data_lock:
            # Update landmark positions
            for obj in msg.objects:
                label = obj.label
                position = obj.position
                
                if label not in self.landmark_positions:
                    self.landmark_positions[label] = []
                
                # Add position if not too close to existing ones
                is_new_position = True
                for existing_pos in self.landmark_positions[label]:
                    if self.calculate_distance(position, existing_pos) < 1.0:  # 1m threshold
                        is_new_position = False
                        break
                
                if is_new_position:
                    self.landmark_positions[label].append(position)
                    
                    # Limit positions per landmark
                    if len(self.landmark_positions[label]) > 5:
                        self.landmark_positions[label].pop(0)
            
            # Update marker features based on nearby objects
            self.update_marker_features()
    
    def odometry_callback(self, msg):
        """Update current robot pose"""
        self.current_pose = msg.pose.pose
    
    def instruction_callback(self, msg):
        """Process new instruction to understand target"""
        self.current_instruction = msg.data.lower()
        
        # Extract location hints from instruction
        location_hints = self.extract_location_hints(self.current_instruction)
        
        if location_hints and self.detected_markers:
            target_marker = self.find_target_marker(location_hints)
            if target_marker:
                self.publish_target_marker(target_marker)
    
    def status_callback(self, msg):
        """Monitor VLN system status"""
        self.task_active = (msg.state == VehicleStatus.STATE_NAVIGATION or 
                           msg.state == VehicleStatus.STATE_INITIALIZING)
    
    def cleanup_stale_markers(self):
        """Remove markers that haven't been seen recently"""
        current_time = rospy.Time.now()
        stale_markers = []
        
        for marker_id, marker_info in self.detected_markers.items():
            time_diff = (current_time - marker_info['timestamp']).to_sec()
            if time_diff > self.marker_timeout:
                stale_markers.append(marker_id)
        
        for marker_id in stale_markers:
            del self.detected_markers[marker_id]
            rospy.loginfo(f"Removed stale marker {marker_id}")
    
    def update_spatial_relationships(self):
        """Build spatial relationship graph between markers and landmarks"""
        if not self.current_pose:
            return
        
        self.spatial_relationships = {}
        
        for marker_id, marker_info in self.detected_markers.items():
            relationships = []
            marker_pos = marker_info['position']
            
            # Find nearby landmarks
            for landmark_type, positions in self.landmark_positions.items():
                for landmark_pos in positions:
                    distance = self.calculate_distance(marker_pos, landmark_pos)
                    if distance < self.spatial_tolerance:
                        relationships.append({
                            'type': 'near',
                            'object': landmark_type,
                            'distance': distance,
                            'direction': self.calculate_relative_direction(marker_pos, landmark_pos)
                        })
            
            # Find relationships with other markers
            for other_id, other_info in self.detected_markers.items():
                if other_id != marker_id:
                    distance = self.calculate_distance(marker_pos, other_info['position'])
                    if distance < 5.0:  # 5m threshold for marker relationships
                        relationships.append({
                            'type': 'marker_near',
                            'object': f'marker_{other_id}',
                            'distance': distance,
                            'direction': self.calculate_relative_direction(marker_pos, other_info['position'])
                        })
            
            self.spatial_relationships[marker_id] = relationships
    
    def update_marker_features(self):
        """Update marker features based on surrounding objects"""
        for marker_id, marker_info in self.detected_markers.items():
            features = {
                'nearby_objects': [],
                'scene_description': ""
            }
            
            marker_pos = marker_info['position']
            
            # Find objects near this marker
            for landmark_type, positions in self.landmark_positions.items():
                for landmark_pos in positions:
                    distance = self.calculate_distance(marker_pos, landmark_pos)
                    if distance < 3.0:  # 3m radius for features
                        features['nearby_objects'].append({
                            'type': landmark_type,
                            'distance': distance,
                            'direction': self.calculate_relative_direction(marker_pos, landmark_pos)
                        })
            
            # Generate scene description
            if features['nearby_objects']:
                objects_list = [obj['type'] for obj in features['nearby_objects']]
                features['scene_description'] = f"marker near {', '.join(set(objects_list))}"
            else:
                features['scene_description'] = "marker in open area"
            
            self.marker_features[marker_id] = features
    
    def extract_location_hints(self, instruction):
        """Extract location hints from natural language instruction"""
        location_hints = {
            'landmarks': [],
            'spatial_relations': [],
            'descriptors': []
        }
        
        # Common landmarks
        landmarks = ['tree', 'cone', 'wall', 'bench', 'barrel', 'truck', 'hydrant']
        for landmark in landmarks:
            if landmark in instruction:
                location_hints['landmarks'].append(landmark)
        
        # Spatial relations
        if 'near' in instruction or 'next to' in instruction or 'beside' in instruction:
            location_hints['spatial_relations'].append('near')
        if 'far' in instruction or 'away from' in instruction:
            location_hints['spatial_relations'].append('far')
        if 'between' in instruction:
            location_hints['spatial_relations'].append('between')
        
        # Descriptors
        if 'first' in instruction or 'closest' in instruction:
            location_hints['descriptors'].append('closest')
        if 'last' in instruction or 'farthest' in instruction:
            location_hints['descriptors'].append('farthest')
        if 'left' in instruction:
            location_hints['descriptors'].append('left')
        if 'right' in instruction:
            location_hints['descriptors'].append('right')
        
        return location_hints
    
    def find_target_marker(self, location_hints):
        """Find the most likely target marker based on location hints"""
        if not self.detected_markers:
            return None
        
        candidates = []
        
        for marker_id, marker_info in self.detected_markers.items():
            score = 0.0
            
            # Base confidence score
            score += marker_info['confidence'] * 10
            
            # Detection stability score
            score += min(marker_info['detection_count'], 10) * 2
            
            # Spatial relationship matching
            if location_hints['landmarks']:
                relationships = self.spatial_relationships.get(marker_id, [])
                for hint_landmark in location_hints['landmarks']:
                    for rel in relationships:
                        if rel['object'] == hint_landmark:
                            if 'near' in location_hints['spatial_relations']:
                                score += 20 / (1 + rel['distance'])  # Higher score for closer objects
                            else:
                                score += 10
            
            # Distance-based scoring
            if self.current_pose and 'closest' in location_hints['descriptors']:
                distance = self.calculate_distance(marker_info['position'], self.current_pose.position)
                score += 30 / (1 + distance)  # Prefer closer markers
            elif self.current_pose and 'farthest' in location_hints['descriptors']:
                distance = self.calculate_distance(marker_info['position'], self.current_pose.position)
                score += distance * 2  # Prefer farther markers
            
            # Directional scoring
            if self.current_pose and ('left' in location_hints['descriptors'] or 'right' in location_hints['descriptors']):
                direction = self.calculate_relative_direction(self.current_pose.position, marker_info['position'])
                if 'left' in location_hints['descriptors'] and direction in ['left', 'front_left', 'back_left']:
                    score += 15
                elif 'right' in location_hints['descriptors'] and direction in ['right', 'front_right', 'back_right']:
                    score += 15
            
            candidates.append((marker_id, score, marker_info))
        
        # Sort by score and return best candidate
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        if candidates and candidates[0][1] > 5.0:  # Minimum score threshold
            return candidates[0][2]  # Return marker_info
        
        return None
    
    def analyze_and_publish_guidance(self):
        """Analyze current situation and publish guidance"""
        if not self.detected_markers:
            return
        
        guidance = {
            'marker_count': len(self.detected_markers),
            'markers': [],
            'recommendation': '',
            'confidence': 0.0
        }
        
        # Sort markers by confidence and detection stability
        sorted_markers = sorted(
            self.detected_markers.items(),
            key=lambda x: (x[1]['confidence'] * x[1]['detection_count']),
            reverse=True
        )
        
        for marker_id, marker_info in sorted_markers:
            marker_data = {
                'id': marker_id,
                'position': {
                    'x': marker_info['position'].x,
                    'y': marker_info['position'].y,
                    'z': marker_info['position'].z
                },
                'confidence': marker_info['confidence'],
                'detection_count': marker_info['detection_count'],
                'features': self.marker_features.get(marker_id, {}),
                'relationships': self.spatial_relationships.get(marker_id, [])
            }
            
            if self.current_pose:
                distance = self.calculate_distance(marker_info['position'], self.current_pose.position)
                marker_data['distance'] = distance
                marker_data['direction'] = self.calculate_relative_direction(
                    self.current_pose.position, marker_info['position']
                )
            
            guidance['markers'].append(marker_data)
        
        # Generate recommendation
        if len(guidance['markers']) == 1:
            guidance['recommendation'] = f"Single marker detected (ID: {guidance['markers'][0]['id']})"
            guidance['confidence'] = guidance['markers'][0]['confidence']
        elif len(guidance['markers']) > 1:
            best_marker = guidance['markers'][0]
            guidance['recommendation'] = f"Multiple markers detected, best candidate: ID {best_marker['id']}"
            guidance['confidence'] = best_marker['confidence']
        
        # Publish guidance
        guidance_msg = String()
        guidance_msg.data = json.dumps(guidance)
        self.marker_guidance_pub.publish(guidance_msg)
    
    def publish_target_marker(self, marker_info):
        """Publish the selected target marker"""
        target_data = {
            'target_marker_id': marker_info['id'],
            'position': {
                'x': marker_info['position'].x,
                'y': marker_info['position'].y,
                'z': marker_info['position'].z
            },
            'confidence': marker_info['confidence'],
            'features': self.marker_features.get(marker_info['id'], {}),
            'selection_reason': 'instruction_analysis'
        }
        
        target_msg = String()
        target_msg.data = json.dumps(target_data)
        self.target_marker_pub.publish(target_msg)
        
        rospy.loginfo(f"Selected target marker: ID {marker_info['id']} with confidence {marker_info['confidence']}")
    
    def calculate_distance(self, pos1, pos2):
        """Calculate Euclidean distance between two points"""
        dx = pos1.x - pos2.x
        dy = pos1.y - pos2.y
        dz = pos1.z - pos2.z
        return math.sqrt(dx*dx + dy*dy + dz*dz)
    
    def calculate_relative_direction(self, from_pos, to_pos):
        """Calculate relative direction from one position to another"""
        dx = to_pos.x - from_pos.x
        dy = to_pos.y - from_pos.y
        
        angle = math.atan2(dy, dx)
        
        # Normalize to 8 directions
        angle_degrees = math.degrees(angle)
        if angle_degrees < 0:
            angle_degrees += 360
        
        if 337.5 <= angle_degrees or angle_degrees < 22.5:
            return 'front'
        elif 22.5 <= angle_degrees < 67.5:
            return 'front_right'
        elif 67.5 <= angle_degrees < 112.5:
            return 'right'
        elif 112.5 <= angle_degrees < 157.5:
            return 'back_right'
        elif 157.5 <= angle_degrees < 202.5:
            return 'back'
        elif 202.5 <= angle_degrees < 247.5:
            return 'back_left'
        elif 247.5 <= angle_degrees < 292.5:
            return 'left'
        else:  # 292.5 <= angle_degrees < 337.5
            return 'front_left'

if __name__ == '__main__':
    try:
        manager = MarkerManager()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
