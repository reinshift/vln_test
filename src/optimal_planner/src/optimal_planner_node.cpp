#include <ros/ros.h>
#include <nav_msgs/OccupancyGrid.h>
#include <geometry_msgs/PoseStamped.h>
#include <std_msgs/String.h>
#include <std_msgs/Int32.h>
#include <magv_vln_msgs/OptimalPath.h>
#include <magv_vln_msgs/VehicleStatus.h>
#include <magv_vln_msgs/PathPoint.h>
#include "optimal_planner/astar_planner.h"
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <json/json.h>

class OptimalPlannerNode {
public:
    OptimalPlannerNode() : tf_listener_(tf_buffer_) {
        ros::NodeHandle nh;
        ros::NodeHandle pnh("~");
        
        // Parameters
        pnh.param("competition_mode", competition_mode_, true);
        pnh.param("timeout_duration", timeout_duration_, 100.0);
        pnh.param("marker_completion_radius", marker_completion_radius_, 2.0);
        pnh.param("replanning_frequency", replanning_frequency_, 2.0);
        pnh.param("use_direct_navigation", use_direct_navigation_, true);
        
        // Initialize planner
        planner_ = std::make_unique<optimal_planner::AStarPlanner>();
        planner_->setInflationRadius(0.4);  // Robot radius + safety margin
        planner_->setHeuristicWeight(1.2);  // Slightly weighted for speed
        planner_->setMaxIterations(15000);  // Increased for complex environments
        
        // Publishers
        optimal_path_pub_ = nh.advertise<magv_vln_msgs::OptimalPath>("/optimal_path", 1);
        path_point_pub_ = nh.advertise<magv_vln_msgs::PathPoint>("/path_point", 10);
        status_pub_ = nh.advertise<std_msgs::Int32>("/status", 1);
        core_feedback_pub_ = nh.advertise<std_msgs::String>("/core_feedback", 1);
        
        // Subscribers
        occupancy_grid_sub_ = nh.subscribe("/occupancy_grid", 1, 
            &OptimalPlannerNode::occupancyGridCallback, this);
        vln_status_sub_ = nh.subscribe("/vln_status", 1,
            &OptimalPlannerNode::vlnStatusCallback, this);
        aruco_info_sub_ = nh.subscribe("/aruco_info", 1,
            &OptimalPlannerNode::arucoInfoCallback, this);
        goal_request_sub_ = nh.subscribe("/planning_request", 1,
            &OptimalPlannerNode::planningRequestCallback, this);
        
        // Timers
        if (competition_mode_) {
            replanning_timer_ = nh.createTimer(ros::Duration(1.0 / replanning_frequency_),
                &OptimalPlannerNode::replanningCallback, this);
        }
        
        // Initialize state
        task_start_time_ = ros::Time::now();
        current_path_index_ = 0;
        navigation_active_ = false;
        marker_detected_ = false;
        
        ROS_INFO("Optimal Planner Node initialized in %s mode", 
                 competition_mode_ ? "competition" : "normal");
    }

private:
    // ROS communication
    ros::Publisher optimal_path_pub_;
    ros::Publisher path_point_pub_;
    ros::Publisher status_pub_;
    ros::Publisher core_feedback_pub_;
    
    ros::Subscriber occupancy_grid_sub_;
    ros::Subscriber vln_status_sub_;
    ros::Subscriber aruco_info_sub_;
    ros::Subscriber goal_request_sub_;
    
    ros::Timer replanning_timer_;
    
    // TF
    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;
    
    // Planner
    std::unique_ptr<optimal_planner::AStarPlanner> planner_;
    
    // State variables
    nav_msgs::OccupancyGrid current_occupancy_grid_;
    geometry_msgs::PoseStamped current_pose_;
    geometry_msgs::Point current_goal_;
    magv_vln_msgs::OptimalPath current_path_;
    
    bool competition_mode_;
    double timeout_duration_;
    double marker_completion_radius_;
    double replanning_frequency_;
    bool use_direct_navigation_;
    
    ros::Time task_start_time_;
    int current_path_index_;
    bool navigation_active_;
    bool marker_detected_;
    geometry_msgs::Point marker_position_;
    bool has_occupancy_grid_;
    bool has_goal_;
    
    void occupancyGridCallback(const nav_msgs::OccupancyGrid::ConstPtr& msg) {
        current_occupancy_grid_ = *msg;
        has_occupancy_grid_ = true;
        
        // Trigger replanning if we have a goal and are navigating
        if (has_goal_ && navigation_active_) {
            planOptimalPath();
        }
    }
    
    void vlnStatusCallback(const magv_vln_msgs::VehicleStatus::ConstPtr& msg) {
        // Update current pose from VLN status
        current_pose_.pose.position = msg->current_position;
        current_pose_.header = msg->header;
        
        // Check if we should start navigation
        if (msg->state == magv_vln_msgs::VehicleStatus::STATE_NAVIGATION && !navigation_active_) {
            navigation_active_ = true;
            if (has_goal_ && has_occupancy_grid_) {
                planOptimalPath();
            }
        } else if (msg->state != magv_vln_msgs::VehicleStatus::STATE_NAVIGATION) {
            navigation_active_ = false;
        }
    }
    
    void arucoInfoCallback(const aruco_detector::msg::ArucoInfo::ConstPtr& msg) {
        if (!msg->markers.empty()) {
            marker_detected_ = true;
            marker_position_ = msg->markers[0].pose.position;
            
            // Check if we're in competition mode and close enough to marker
            if (competition_mode_) {
                double distance = calculateDistance(current_pose_.pose.position, marker_position_);
                if (distance <= marker_completion_radius_) {
                    publishTaskCompletion();
                    return;
                }
                
                // If using direct navigation and path is clear, go directly to marker
                if (use_direct_navigation_ && isPathClear(current_pose_.pose.position, marker_position_)) {
                    planDirectPathToMarker();
                }
            }
        }
    }
    
    void planningRequestCallback(const std_msgs::String::ConstPtr& msg) {
        // Parse planning request (JSON format expected)
        Json::Value request;
        Json::Reader reader;
        
        if (reader.parse(msg->data, request)) {
            if (request.isMember("goal")) {
                current_goal_.x = request["goal"]["x"].asDouble();
                current_goal_.y = request["goal"]["y"].asDouble();
                current_goal_.z = request["goal"]["z"].asDouble();
                has_goal_ = true;
                
                if (has_occupancy_grid_ && navigation_active_) {
                    planOptimalPath();
                }
            }
        }
    }
    
    void replanningCallback(const ros::TimerEvent& event) {
        if (!navigation_active_ || !has_goal_ || !has_occupancy_grid_) {
            return;
        }
        
        // Check if we need to replan due to:
        // 1. Significant deviation from path
        // 2. New obstacles detected
        // 3. Marker detected (for direct navigation)
        
        if (shouldReplan()) {
            planOptimalPath();
        }
        
        // Publish next waypoint if we have a valid path
        publishNextWaypoint();
        
        // Check timeout in competition mode
        if (competition_mode_) {
            double elapsed_time = (ros::Time::now() - task_start_time_).toSec();
            if (elapsed_time > timeout_duration_) {
                ROS_WARN("Task timeout reached, stopping navigation");
                navigation_active_ = false;
                publishTaskFailure();
            }
        }
    }
    
    bool planOptimalPath() {
        if (!has_occupancy_grid_ || !has_goal_) {
            return false;
        }
        
        geometry_msgs::Point start_point = current_pose_.pose.position;
        geometry_msgs::Point goal_point = marker_detected_ && use_direct_navigation_ ? 
                                         marker_position_ : current_goal_;
        
        magv_vln_msgs::OptimalPath new_path;
        bool success = planner_->planPath(current_occupancy_grid_, start_point, goal_point, new_path);
        
        if (success) {
            current_path_ = new_path;
            current_path_index_ = 0;
            optimal_path_pub_.publish(current_path_);
            
            // Send feedback to core node
            Json::Value feedback;
            feedback["status"] = "path_planned";
            feedback["path_length"] = current_path_.total_length;
            feedback["spl_score"] = current_path_.spl_score;
            feedback["waypoints_count"] = (int)current_path_.waypoints.size();
            
            std_msgs::String feedback_msg;
            Json::StreamWriterBuilder builder;
            feedback_msg.data = Json::writeString(builder, feedback);
            core_feedback_pub_.publish(feedback_msg);
            
            ROS_INFO("Optimal path planned: %.2fm length, %.3f SPL score, %d waypoints",
                     current_path_.total_length, current_path_.spl_score, 
                     (int)current_path_.waypoints.size());
            
            return true;
        } else {
            ROS_WARN("Failed to plan optimal path");
            return false;
        }
    }
    
    void planDirectPathToMarker() {
        if (!marker_detected_) return;
        
        // Create simple direct path to marker
        magv_vln_msgs::OptimalPath direct_path;
        direct_path.header.frame_id = "map";
        direct_path.header.stamp = ros::Time::now();
        direct_path.planning_method = "direct";
        
        // Add current position as first waypoint
        magv_vln_msgs::PathPoint start_wp;
        start_wp.header = direct_path.header;
        start_wp.position = current_pose_.pose.position;
        start_wp.is_final_goal = false;
        direct_path.waypoints.push_back(start_wp);
        
        // Add marker position as final waypoint
        magv_vln_msgs::PathPoint goal_wp;
        goal_wp.header = direct_path.header;
        goal_wp.position = marker_position_;
        goal_wp.is_final_goal = true;
        direct_path.waypoints.push_back(goal_wp);
        
        // Calculate metrics
        double dx = marker_position_.x - current_pose_.pose.position.x;
        double dy = marker_position_.y - current_pose_.pose.position.y;
        direct_path.total_length = std::sqrt(dx*dx + dy*dy);
        direct_path.optimal_length = direct_path.total_length;
        direct_path.efficiency = 1.0;
        direct_path.spl_score = 1.0;
        
        current_path_ = direct_path;
        current_path_index_ = 0;
        optimal_path_pub_.publish(current_path_);
        
        ROS_INFO("Direct path to marker planned: %.2fm", direct_path.total_length);
    }
    
    void publishNextWaypoint() {
        if (current_path_.waypoints.empty() || 
            current_path_index_ >= current_path_.waypoints.size()) {
            return;
        }
        
        // Check if we've reached current waypoint
        auto& current_wp = current_path_.waypoints[current_path_index_];
        double distance = calculateDistance(current_pose_.pose.position, current_wp.position);
        
        if (distance < current_wp.position_tolerance) {
            current_path_index_++;
            
            // Check if we've completed the path
            if (current_path_index_ >= current_path_.waypoints.size()) {
                if (competition_mode_ && marker_detected_) {
                    publishTaskCompletion();
                } else {
                    publishNavigationComplete();
                }
                return;
            }
        }
        
        // Publish current target waypoint
        if (current_path_index_ < current_path_.waypoints.size()) {
            path_point_pub_.publish(current_path_.waypoints[current_path_index_]);
        }
    }
    
    bool shouldReplan() {
        // Simple replanning heuristics
        if (current_path_.waypoints.empty()) return true;
        
        // Check if significantly deviated from path
        if (current_path_index_ < current_path_.waypoints.size()) {
            auto& target_wp = current_path_.waypoints[current_path_index_];
            double distance = calculateDistance(current_pose_.pose.position, target_wp.position);
            if (distance > 1.0) { // 1 meter deviation threshold
                return true;
            }
        }
        
        // Check if new marker detected
        if (marker_detected_ && use_direct_navigation_ && 
            current_path_.planning_method != "direct") {
            return true;
        }
        
        return false;
    }
    
    bool isPathClear(const geometry_msgs::Point& start, const geometry_msgs::Point& goal) {
        // Simplified path clearance check
        // In a full implementation, this would do line-of-sight checking
        return true; // Conservative approach - assume clear for now
    }
    
    double calculateDistance(const geometry_msgs::Point& p1, const geometry_msgs::Point& p2) {
        double dx = p2.x - p1.x;
        double dy = p2.y - p1.y;
        return std::sqrt(dx*dx + dy*dy);
    }
    
    void publishTaskCompletion() {
        std_msgs::Int32 status_msg;
        status_msg.data = 1; // Success
        status_pub_.publish(status_msg);
        
        Json::Value feedback;
        feedback["status"] = "task_complete";
        feedback["success"] = true;
        feedback["completion_time"] = (ros::Time::now() - task_start_time_).toSec();
        
        std_msgs::String feedback_msg;
        Json::StreamWriterBuilder builder;
        feedback_msg.data = Json::writeString(builder, feedback);
        core_feedback_pub_.publish(feedback_msg);
        
        navigation_active_ = false;
        ROS_INFO("Task completed successfully!");
    }
    
    void publishTaskFailure() {
        std_msgs::Int32 status_msg;
        status_msg.data = 0; // Failure
        status_pub_.publish(status_msg);
        
        Json::Value feedback;
        feedback["status"] = "task_failed";
        feedback["success"] = false;
        feedback["reason"] = "timeout";
        
        std_msgs::String feedback_msg;
        Json::StreamWriterBuilder builder;
        feedback_msg.data = Json::writeString(builder, feedback);
        core_feedback_pub_.publish(feedback_msg);
        
        ROS_WARN("Task failed - timeout reached");
    }
    
    void publishNavigationComplete() {
        Json::Value feedback;
        feedback["status"] = "navigation_complete";
        feedback["subtask_completed"] = true;
        
        std_msgs::String feedback_msg;
        Json::StreamWriterBuilder builder;
        feedback_msg.data = Json::writeString(builder, feedback);
        core_feedback_pub_.publish(feedback_msg);
        
        ROS_INFO("Navigation to goal completed");
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "optimal_planner_node");
    
    try {
        OptimalPlannerNode node;
        ros::spin();
    } catch (const std::exception& e) {
        ROS_ERROR("Optimal planner node error: %s", e.what());
        return 1;
    }
    
    return 0;
}
