#include <ros/ros.h>
#include <std_msgs/Header.h>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/Twist.h>
#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/LaserScan.h>
#include <sensor_msgs/PointCloud2.h>
#include <magv_vln_msgs/VehicleStatus.h>
#include <magv_vln_msgs/PositionCommand.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <cmath>
#include <limits>

class VehicleStatusManager
{
private:
    ros::NodeHandle nh_;
    ros::NodeHandle private_nh_;
    
    // Publishers
    ros::Publisher status_pub_;
    
    // Subscribers
    ros::Subscriber odom_sub_;
    ros::Subscriber cmd_sub_;
    ros::Subscriber laser_sub_;
    ros::Subscriber goal_sub_;
    ros::Subscriber pointcloud_sub_;
    
    // Timer for periodic status updates
    ros::Timer status_timer_;
    
    // Current vehicle status
    magv_vln_msgs::VehicleStatus current_status_;
    
    // Internal state variables
    bool sensors_initialized_;
    bool navigation_initialized_;
    bool has_received_odom_;
    bool has_received_laser_;
    bool has_received_pointcloud_;
    bool is_moving_;
    bool has_active_goal_;
    bool emergency_stop_triggered_;
    
    geometry_msgs::Point last_position_;
    geometry_msgs::Point current_goal_;
    ros::Time last_movement_time_;
    ros::Time last_state_change_time_;

    // Emergency stop variables
    double min_obstacle_distance_;
    ros::Time last_pointcloud_time_;
    uint8_t previous_state_;  // Store state before emergency stop

    // Parameters
    double status_publish_rate_;
    double movement_threshold_;
    double idle_timeout_;
    double emergency_stop_distance_;
    double pointcloud_timeout_;

public:
    VehicleStatusManager() : private_nh_("~"),
                           sensors_initialized_(false),
                           navigation_initialized_(false),
                           has_received_odom_(false),
                           has_received_laser_(false),
                           has_received_pointcloud_(false),
                           is_moving_(false),
                           has_active_goal_(false),
                           emergency_stop_triggered_(false),
                           min_obstacle_distance_(std::numeric_limits<double>::max()),
                           previous_state_(magv_vln_msgs::VehicleStatus::STATE_IDLE)
    {
        // Load parameters
        private_nh_.param("status_publish_rate", status_publish_rate_, 2.0);
        private_nh_.param("movement_threshold", movement_threshold_, 0.1);
        private_nh_.param("idle_timeout", idle_timeout_, 30.0);
        private_nh_.param("emergency_stop_distance", emergency_stop_distance_, 0.5);  // 0.5m default
        private_nh_.param("pointcloud_timeout", pointcloud_timeout_, 2.0);  // 2s timeout
        
        // Initialize publishers
        status_pub_ = nh_.advertise<magv_vln_msgs::VehicleStatus>("/vln_status", 10);
        
        // Initialize subscribers
        odom_sub_ = nh_.subscribe("/odom", 10, &VehicleStatusManager::odomCallback, this);
        cmd_sub_ = nh_.subscribe("/cmd_vel", 10, &VehicleStatusManager::cmdVelCallback, this);
        laser_sub_ = nh_.subscribe("/scan", 10, &VehicleStatusManager::laserCallback, this);
        goal_sub_ = nh_.subscribe("/move_base_simple/goal", 10, &VehicleStatusManager::goalCallback, this);
        pointcloud_sub_ = nh_.subscribe("magv/scan/3d", 10, &VehicleStatusManager::pointcloudCallback, this);
        
        // Initialize status timer
        status_timer_ = nh_.createTimer(ros::Duration(1.0/status_publish_rate_), 
                                       &VehicleStatusManager::statusTimerCallback, this);
        
        // Initialize vehicle status
        initializeStatus();
        
        ROS_INFO("Vehicle Status Manager initialized");
    }
    
    void initializeStatus()
    {
        current_status_.header.frame_id = "base_link";
        current_status_.state = magv_vln_msgs::VehicleStatus::STATE_INITIALIZING;
        current_status_.state_description = "Initializing";
        current_status_.is_moving = false;
        current_status_.has_goal = false;
        current_status_.sensors_ready = false;
        current_status_.navigation_ready = false;
        current_status_.last_state_change = ros::Time::now();
        current_status_.diagnostic_info = "System starting up";
        
        last_state_change_time_ = ros::Time::now();
        last_movement_time_ = ros::Time::now();
    }
    
    void odomCallback(const nav_msgs::Odometry::ConstPtr& msg)
    {
        if (!has_received_odom_) {
            has_received_odom_ = true;
            ROS_INFO("Received first odometry message");
        }
        
        // Update current position
        current_status_.current_position = msg->pose.pose.position;
        
        // TODO: Implement movement detection logic
        // Check if vehicle is moving based on odometry
        // Example logic (to be refined):
        /*
        double distance_moved = sqrt(
            pow(msg->pose.pose.position.x - last_position_.x, 2) +
            pow(msg->pose.pose.position.y - last_position_.y, 2)
        );
        
        if (distance_moved > movement_threshold_) {
            is_moving_ = true;
            last_movement_time_ = ros::Time::now();
        } else if ((ros::Time::now() - last_movement_time_).toSec() > 2.0) {
            is_moving_ = false;
        }
        */
        
        last_position_ = msg->pose.pose.position;
    }
    
    void cmdVelCallback(const geometry_msgs::Twist::ConstPtr& msg)
    {
        // TODO: Implement movement detection based on velocity commands
        // Example logic:
        /*
        double linear_vel = sqrt(pow(msg->linear.x, 2) + pow(msg->linear.y, 2));
        double angular_vel = fabs(msg->angular.z);
        
        if (linear_vel > 0.01 || angular_vel > 0.01) {
            is_moving_ = true;
            last_movement_time_ = ros::Time::now();
        }
        */
    }
    
    void laserCallback(const sensor_msgs::LaserScan::ConstPtr& msg)
    {
        if (!has_received_laser_) {
            has_received_laser_ = true;
            ROS_INFO("Received first laser scan message");
        }
        
        // TODO: Implement sensor health check
        // Example logic:
        /*
        // Check if laser data is valid
        bool laser_healthy = true;
        for (const auto& range : msg->ranges) {
            if (std::isnan(range) || std::isinf(range)) {
                // Some invalid readings are normal, but too many indicate problems
            }
        }
        */
    }
    
    void goalCallback(const geometry_msgs::PoseStamped::ConstPtr& msg)
    {
        has_active_goal_ = true;
        current_goal_ = msg->pose.position;

        ROS_INFO("Received new navigation goal: [%.2f, %.2f]",
                 current_goal_.x, current_goal_.y);

        // TODO: Implement goal validation logic
        // Example logic:
        /*
        // Validate if goal is reachable
        // Check if goal is within map bounds
        // Check if goal is not in obstacle
        */
    }

    void pointcloudCallback(const sensor_msgs::PointCloud2::ConstPtr& msg)
    {
        if (!has_received_pointcloud_) {
            has_received_pointcloud_ = true;
            ROS_INFO("Received first pointcloud message");
        }

        last_pointcloud_time_ = ros::Time::now();

        // Convert ROS PointCloud2 to PCL
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
        try {
            pcl::fromROSMsg(*msg, *cloud);
        } catch (const std::exception& e) {
            ROS_ERROR("PCL conversion failed: %s", e.what());
            return;
        }

        // Calculate minimum distance to obstacles
        min_obstacle_distance_ = calculateMinObstacleDistance(cloud);

        // Check for emergency stop condition
        if (min_obstacle_distance_ < emergency_stop_distance_) {
            if (!emergency_stop_triggered_) {
                // Store current state before switching to emergency stop
                previous_state_ = current_status_.state;
                emergency_stop_triggered_ = true;
                ROS_WARN("EMERGENCY STOP TRIGGERED! Obstacle detected at %.2fm (threshold: %.2fm)",
                         min_obstacle_distance_, emergency_stop_distance_);
            }
        } else {
            // Clear emergency stop if distance is safe again
            if (emergency_stop_triggered_ && min_obstacle_distance_ > emergency_stop_distance_ + 0.1) {
                emergency_stop_triggered_ = false;
                ROS_INFO("Emergency stop cleared. Distance: %.2fm", min_obstacle_distance_);
            }
        }


    }

    double calculateMinObstacleDistance(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud)
    {
        double min_distance = std::numeric_limits<double>::max();

        // Vehicle position is at origin (0, 0, 0) in base_link frame
        for (const auto& point : cloud->points) {
            // Skip invalid points
            if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z)) {
                continue;
            }

            // Calculate 2D distance (ignore z for ground-based vehicle)
            double distance = sqrt(point.x * point.x + point.y * point.y);

            // Consider all points above ground level
            if (point.z > 0.0) {
                if (distance < min_distance) {
                    min_distance = distance;
                }
            }
        }

        return min_distance;
    }
    
    void statusTimerCallback(const ros::TimerEvent& event)
    {
        updateVehicleState();
        publishStatus();
    }
    
    void updateVehicleState()
    {
        uint8_t new_state = current_status_.state;
        std::string new_description = current_status_.state_description;
        std::string diagnostic = "";
        
        // Update sensor and navigation readiness (for normal operations)
        current_status_.sensors_ready = has_received_odom_ && has_received_laser_;
        current_status_.navigation_ready = current_status_.sensors_ready; // Add more conditions as needed

        // Check for emergency stop condition first (highest priority)
        // Emergency stop only needs pointcloud data, not other sensors
        if (has_received_pointcloud_ && emergency_stop_triggered_) {
            if (current_status_.state != magv_vln_msgs::VehicleStatus::STATE_EMERGENCY_STOP) {
                new_state = magv_vln_msgs::VehicleStatus::STATE_EMERGENCY_STOP;
                new_description = "Emergency Stop - Obstacle Too Close";
                diagnostic = "Obstacle detected at " + std::to_string(min_obstacle_distance_) + "m (threshold: " + std::to_string(emergency_stop_distance_) + "m)";
            }
        } else if (has_received_pointcloud_ && current_status_.state == magv_vln_msgs::VehicleStatus::STATE_EMERGENCY_STOP && !emergency_stop_triggered_) {
            // Recovery from emergency stop - return to previous state or idle
            new_state = (previous_state_ == magv_vln_msgs::VehicleStatus::STATE_EMERGENCY_STOP) ?
                        magv_vln_msgs::VehicleStatus::STATE_IDLE : previous_state_;
            new_description = "Recovered from Emergency Stop";
            diagnostic = "Safe distance restored: " + std::to_string(min_obstacle_distance_) + "m";
        }
        
        // TODO: Implement comprehensive state transition logic
        // This is a basic framework - expand based on your specific requirements

        // Only proceed with normal state logic if not in emergency stop or emergency stop not triggered
        if (!emergency_stop_triggered_ && current_status_.state != magv_vln_msgs::VehicleStatus::STATE_EMERGENCY_STOP) {
            switch (current_status_.state) {
            case magv_vln_msgs::VehicleStatus::STATE_INITIALIZING:
                diagnostic = "Waiting for sensors and navigation stack";
                
                // TODO: Add initialization completion logic
                /*
                if (current_status_.sensors_ready && current_status_.navigation_ready) {
                    // Check if enough time has passed for initialization
                    if ((ros::Time::now() - last_state_change_time_).toSec() > 5.0) {
                        new_state = magv_vln_msgs::VehicleStatus::STATE_IDLE;
                        new_description = "Ready - Idle";
                    }
                }
                */
                break;
                
            case magv_vln_msgs::VehicleStatus::STATE_IDLE:
                diagnostic = "System ready, waiting for commands";
                
                // TODO: Add transition to exploration or navigation
                /*
                if (has_active_goal_) {
                    new_state = magv_vln_msgs::VehicleStatus::STATE_NAVIGATION;
                    new_description = "Navigating to goal";
                } else if (should_start_exploration()) {
                    new_state = magv_vln_msgs::VehicleStatus::STATE_EXPLORATION;
                    new_description = "Exploring environment";
                }
                */
                break;
                
            case magv_vln_msgs::VehicleStatus::STATE_EXPLORATION:
                diagnostic = "Actively exploring the environment";
                
                // TODO: Add exploration completion/interruption logic
                /*
                if (has_active_goal_) {
                    new_state = magv_vln_msgs::VehicleStatus::STATE_NAVIGATION;
                    new_description = "Switching to navigation mode";
                } else if (exploration_completed()) {
                    new_state = magv_vln_msgs::VehicleStatus::STATE_IDLE;
                    new_description = "Exploration completed";
                }
                */
                break;
                
            case magv_vln_msgs::VehicleStatus::STATE_NAVIGATION:
                diagnostic = "Navigating to target position";
                
                // TODO: Add navigation completion/failure logic
                /*
                if (goal_reached()) {
                    new_state = magv_vln_msgs::VehicleStatus::STATE_IDLE;
                    new_description = "Goal reached";
                    has_active_goal_ = false;
                } else if (navigation_failed()) {
                    new_state = magv_vln_msgs::VehicleStatus::STATE_ERROR;
                    new_description = "Navigation failed";
                }
                */
                break;
                
            case magv_vln_msgs::VehicleStatus::STATE_ERROR:
                diagnostic = "Error state - manual intervention may be required";

                // TODO: Add error recovery logic
                /*
                if (error_resolved()) {
                    new_state = magv_vln_msgs::VehicleStatus::STATE_IDLE;
                    new_description = "Error resolved";
                }
                */
                break;

            case magv_vln_msgs::VehicleStatus::STATE_EMERGENCY_STOP:
                // Emergency stop state - vehicle should be stopped
                // This state is handled by the emergency stop logic above
                // No additional transitions needed here as emergency stop has highest priority
                if (!emergency_stop_triggered_) {
                    // This case is handled above in the emergency stop recovery logic
                }
                break;

            default:
                diagnostic = "Unknown state";
                break;
            }
        }
        
        // Update state if changed
        if (new_state != current_status_.state) {
            ROS_INFO("State transition: %s -> %s", 
                     current_status_.state_description.c_str(), 
                     new_description.c_str());
            
            current_status_.state = new_state;
            current_status_.state_description = new_description;
            current_status_.last_state_change = ros::Time::now();
            last_state_change_time_ = ros::Time::now();
        }
        
        // Update other status fields
        current_status_.is_moving = is_moving_;
        current_status_.has_goal = has_active_goal_;
        current_status_.target_position = current_goal_;
        current_status_.diagnostic_info = diagnostic;
    }
    
    void publishStatus()
    {
        current_status_.header.stamp = ros::Time::now();
        status_pub_.publish(current_status_);
        
        // Log status periodically
        static ros::Time last_log_time = ros::Time::now();
        if ((ros::Time::now() - last_log_time).toSec() > 2.0) {
            ROS_INFO("Vehicle Status: %s | Moving: %s | Goal: %s | Sensors: %s", 
                     current_status_.state_description.c_str(),
                     current_status_.is_moving ? "Yes" : "No",
                     current_status_.has_goal ? "Yes" : "No",
                     current_status_.sensors_ready ? "Ready" : "Not Ready");
            last_log_time = ros::Time::now();
        }
    }
    
    // TODO: Implement helper functions for state logic
    /*
    bool should_start_exploration() {
        // Add logic to determine when to start exploration
        return false;
    }
    
    bool exploration_completed() {
        // Add logic to determine when exploration is complete
        return false;
    }
    
    bool goal_reached() {
        // Add logic to determine when navigation goal is reached
        return false;
    }
    
    bool navigation_failed() {
        // Add logic to determine when navigation has failed
        return false;
    }
    
    bool error_resolved() {
        // Add logic to determine when error conditions are resolved
        return false;
    }
    */
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "vehicle_status_manager");
    
    try {
        VehicleStatusManager status_manager;
        ros::spin();
    }
    catch (const std::exception& e) {
        ROS_ERROR("Vehicle Status Manager error: %s", e.what());
        return 1;
    }
    
    return 0;
}
