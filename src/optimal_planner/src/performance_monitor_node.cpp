#include <ros/ros.h>
#include <std_msgs/String.h>
#include <std_msgs/Int32.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/Twist.h>
#include <magv_vln_msgs/PerformanceMetrics.h>
#include <magv_vln_msgs/VehicleStatus.h>
#include <magv_vln_msgs/OptimalPath.h>
#include <aruco_detector/ArucoInfo.h>
#include <json/json.h>
#include <chrono>
#include <deque>

class PerformanceMonitorNode {
public:
    PerformanceMonitorNode() {
        ros::NodeHandle nh;
        ros::NodeHandle pnh("~");
        
        // Parameters
        pnh.param("competition_mode", competition_mode_, true);
        pnh.param("timeout_duration", timeout_duration_, 100.0);
        pnh.param("marker_completion_radius", marker_completion_radius_, 2.0);
        pnh.param("monitoring_frequency", monitoring_frequency_, 5.0);
        pnh.param("performance_history_size", performance_history_size_, 50);
        
        // Publishers
        performance_pub_ = nh.advertise<magv_vln_msgs::PerformanceMetrics>("/performance_metrics", 1);
        alert_pub_ = nh.advertise<std_msgs::String>("/performance_alerts", 1);
        
        // Subscribers
        odometry_sub_ = nh.subscribe("/magv/odometry/gt", 1, 
            &PerformanceMonitorNode::odometryCallback, this);
        vln_status_sub_ = nh.subscribe("/vln_status", 1,
            &PerformanceMonitorNode::vlnStatusCallback, this);
        cmd_vel_sub_ = nh.subscribe("/magv/omni_drive_controller/cmd_vel", 1,
            &PerformanceMonitorNode::cmdVelCallback, this);
        optimal_path_sub_ = nh.subscribe("/optimal_path", 1,
            &PerformanceMonitorNode::optimalPathCallback, this);
        aruco_info_sub_ = nh.subscribe("/aruco_info", 1,
            &PerformanceMonitorNode::arucoInfoCallback, this);
        instruction_sub_ = nh.subscribe("/instruction", 1,
            &PerformanceMonitorNode::instructionCallback, this);
        
        // Timer for performance monitoring
        monitor_timer_ = nh.createTimer(ros::Duration(1.0 / monitoring_frequency_),
            &PerformanceMonitorNode::monitorCallback, this);
        
        // Initialize state
        task_active_ = false;
        task_start_time_ = ros::Time::now();
        total_distance_traveled_ = 0.0;
        optimal_path_length_ = 0.0;
        marker_detected_ = false;
        marker_distance_ = std::numeric_limits<double>::max();
        
        last_position_.x = 0.0;
        last_position_.y = 0.0;
        last_position_.z = 0.0;
        
        // Performance history
        control_frequency_history_.resize(performance_history_size_);
        processing_delay_history_.resize(performance_history_size_);
        
        ROS_INFO("Performance Monitor Node initialized");
    }

private:
    ros::Publisher performance_pub_;
    ros::Publisher alert_pub_;
    
    ros::Subscriber odometry_sub_;
    ros::Subscriber vln_status_sub_;
    ros::Subscriber cmd_vel_sub_;
    ros::Subscriber optimal_path_sub_;
    ros::Subscriber aruco_info_sub_;
    ros::Subscriber instruction_sub_;
    
    ros::Timer monitor_timer_;
    
    // Parameters
    bool competition_mode_;
    double timeout_duration_;
    double marker_completion_radius_;
    double monitoring_frequency_;
    int performance_history_size_;
    
    // State tracking
    bool task_active_;
    ros::Time task_start_time_;
    ros::Time last_cmd_vel_time_;
    ros::Time last_processing_time_;
    
    geometry_msgs::Point current_position_;
    geometry_msgs::Point last_position_;
    geometry_msgs::Point start_position_;
    
    double total_distance_traveled_;
    double optimal_path_length_;
    
    bool marker_detected_;
    geometry_msgs::Point marker_position_;
    double marker_distance_;
    
    // Performance metrics
    std::deque<double> control_frequency_history_;
    std::deque<double> processing_delay_history_;
    std::chrono::high_resolution_clock::time_point last_update_time_;
    
    std::string current_bottleneck_;
    
    void odometryCallback(const nav_msgs::Odometry::ConstPtr& msg) {
        // Update current position
        geometry_msgs::Point new_position = msg->pose.pose.position;
        
        if (task_active_) {
            // Calculate distance traveled
            double dx = new_position.x - last_position_.x;
            double dy = new_position.y - last_position_.y;
            double distance_increment = std::sqrt(dx*dx + dy*dy);
            
            if (distance_increment < 5.0) { // Sanity check for teleportation
                total_distance_traveled_ += distance_increment;
            }
            
            // Update marker distance if marker detected
            if (marker_detected_) {
                double dx_marker = marker_position_.x - new_position.x;
                double dy_marker = marker_position_.y - new_position.y;
                marker_distance_ = std::sqrt(dx_marker*dx_marker + dy_marker*dy_marker);
            }
        }
        
        last_position_ = current_position_;
        current_position_ = new_position;
        
        // Update processing time tracking
        auto now = std::chrono::high_resolution_clock::now();
        if (last_update_time_.time_since_epoch().count() > 0) {
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(now - last_update_time_);
            double delay = duration.count() / 1000000.0; // Convert to seconds
            updatePerformanceHistory(processing_delay_history_, delay);
        }
        last_update_time_ = now;
    }
    
    void vlnStatusCallback(const magv_vln_msgs::VehicleStatus::ConstPtr& msg) {
        bool was_active = task_active_;
        
        // Check if task state changed
        if (msg->state == magv_vln_msgs::VehicleStatus::STATE_NAVIGATION ||
            msg->state == magv_vln_msgs::VehicleStatus::STATE_INITIALIZING) {
            
            if (!was_active) {
                // Task started
                task_active_ = true;
                task_start_time_ = ros::Time::now();
                start_position_ = msg->current_position;
                total_distance_traveled_ = 0.0;
                marker_detected_ = false;
                
                ROS_INFO("Performance monitoring started");
            }
        } else {
            if (was_active) {
                // Task ended
                task_active_ = false;
                publishFinalPerformanceReport();
                ROS_INFO("Performance monitoring stopped");
            }
        }
    }
    
    void cmdVelCallback(const geometry_msgs::Twist::ConstPtr& msg) {
        last_cmd_vel_time_ = ros::Time::now();
        
        // Calculate control frequency
        static ros::Time last_cmd_time = ros::Time::now();
        double dt = (last_cmd_vel_time_ - last_cmd_time).toSec();
        if (dt > 0 && dt < 1.0) { // Reasonable time difference
            double frequency = 1.0 / dt;
            updatePerformanceHistory(control_frequency_history_, frequency);
        }
        last_cmd_time = last_cmd_vel_time_;
    }
    
    void optimalPathCallback(const magv_vln_msgs::OptimalPath::ConstPtr& msg) {
        optimal_path_length_ = msg->optimal_length;
        last_processing_time_ = ros::Time::now();
    }
    
    void arucoInfoCallback(const aruco_detector::ArucoInfo::ConstPtr& msg) {
        if (!msg->markers.empty()) {
            marker_detected_ = true;
            marker_position_ = msg->markers[0].pose.position;
            
            // Update marker distance
            double dx = marker_position_.x - current_position_.x;
            double dy = marker_position_.y - current_position_.y;
            marker_distance_ = std::sqrt(dx*dx + dy*dy);
        } else {
            marker_detected_ = false;
            marker_distance_ = std::numeric_limits<double>::max();
        }
    }
    
    void instructionCallback(const std_msgs::String::ConstPtr& msg) {
        // New instruction received - reset monitoring
        if (!task_active_) {
            task_start_time_ = ros::Time::now();
            start_position_ = current_position_;
            total_distance_traveled_ = 0.0;
        }
    }
    
    void monitorCallback(const ros::TimerEvent& event) {
        if (!task_active_) {
            return;
        }
        
        // Calculate current metrics
        magv_vln_msgs::PerformanceMetrics metrics;
        calculateCurrentMetrics(metrics);
        
        // Publish metrics
        performance_pub_.publish(metrics);
        
        // Check for performance alerts
        checkPerformanceAlerts(metrics);
    }
    
    void calculateCurrentMetrics(magv_vln_msgs::PerformanceMetrics& metrics) {
        metrics.header.stamp = ros::Time::now();
        metrics.header.frame_id = "map";
        
        // Time metrics
        double elapsed_time = (ros::Time::now() - task_start_time_).toSec();
        metrics.time_elapsed = elapsed_time;
        metrics.time_remaining = timeout_duration_ - elapsed_time;
        
        // Distance metrics
        metrics.distance_traveled = total_distance_traveled_;
        metrics.optimal_distance = optimal_path_length_;
        
        if (metrics.optimal_distance > 0) {
            metrics.path_efficiency = metrics.optimal_distance / metrics.distance_traveled;
        } else {
            metrics.path_efficiency = 1.0;
        }
        
        // SPL calculation (simplified)
        if (marker_detected_ && marker_distance_ <= marker_completion_radius_) {
            metrics.current_spl_score = metrics.path_efficiency;
        } else {
            // Estimate SPL based on current progress
            double progress_factor = 1.0;
            if (marker_detected_) {
                double total_distance_to_goal = total_distance_traveled_ + marker_distance_;
                progress_factor = total_distance_traveled_ / total_distance_to_goal;
            }
            metrics.current_spl_score = metrics.path_efficiency * progress_factor;
        }
        
        // Completion probability based on time remaining and current progress
        if (metrics.time_remaining > 0) {
            double time_factor = std::min(1.0, metrics.time_remaining / 20.0); // 20s buffer
            double marker_factor = marker_detected_ ? 0.8 : 0.5; // Higher if marker detected
            double distance_factor = marker_detected_ ? 
                std::max(0.1, 1.0 - marker_distance_ / 10.0) : 0.3; // Higher if closer to marker
            
            metrics.completion_probability = time_factor * marker_factor * distance_factor;
        } else {
            metrics.completion_probability = 0.0;
        }
        
        // System performance
        metrics.processing_delay = calculateAverageFromHistory(processing_delay_history_);
        metrics.control_frequency = calculateAverageFromHistory(control_frequency_history_);
        
        // Identify bottleneck
        identifyBottleneck(metrics);
        metrics.bottleneck_component = current_bottleneck_;
        
        // Competition specific
        metrics.marker_detected = marker_detected_;
        metrics.marker_distance = marker_distance_;
        metrics.within_completion_zone = marker_detected_ && marker_distance_ <= marker_completion_radius_;
    }
    
    void identifyBottleneck(const magv_vln_msgs::PerformanceMetrics& metrics) {
        current_bottleneck_ = "none";
        
        // Check processing delay
        if (metrics.processing_delay > 0.1) { // 100ms delay threshold
            current_bottleneck_ = "processing_delay";
            return;
        }
        
        // Check control frequency
        if (metrics.control_frequency < 15.0) { // Below 15Hz
            current_bottleneck_ = "control_frequency";
            return;
        }
        
        // Check if we're making progress
        static double last_distance_check = total_distance_traveled_;
        static ros::Time last_progress_time = ros::Time::now();
        
        double time_since_check = (ros::Time::now() - last_progress_time).toSec();
        if (time_since_check > 5.0) { // Check every 5 seconds
            double distance_progress = total_distance_traveled_ - last_distance_check;
            if (distance_progress < 0.5) { // Less than 50cm in 5 seconds
                current_bottleneck_ = "navigation_stuck";
            }
            last_distance_check = total_distance_traveled_;
            last_progress_time = ros::Time::now();
        }
        
        // Check time pressure
        if (metrics.time_remaining < 20.0 && !metrics.marker_detected) {
            current_bottleneck_ = "time_pressure";
        }
    }
    
    void checkPerformanceAlerts(const magv_vln_msgs::PerformanceMetrics& metrics) {
        std::vector<std::string> alerts;
        
        // Critical alerts
        if (metrics.time_remaining < 10.0) {
            alerts.push_back("CRITICAL: Less than 10 seconds remaining!");
        }
        
        if (metrics.completion_probability < 0.3) {
            alerts.push_back("WARNING: Low completion probability (" + 
                           std::to_string((int)(metrics.completion_probability * 100)) + "%)");
        }
        
        if (metrics.control_frequency < 10.0) {
            alerts.push_back("WARNING: Low control frequency (" + 
                           std::to_string(metrics.control_frequency) + " Hz)");
        }
        
        if (metrics.processing_delay > 0.2) {
            alerts.push_back("WARNING: High processing delay (" + 
                           std::to_string((int)(metrics.processing_delay * 1000)) + " ms)");
        }
        
        if (current_bottleneck_ == "navigation_stuck") {
            alerts.push_back("WARNING: Navigation appears stuck");
        }
        
        // Publish alerts
        if (!alerts.empty()) {
            Json::Value alert_json;
            alert_json["timestamp"] = ros::Time::now().toSec();
            alert_json["alerts"] = Json::Value(Json::arrayValue);
            
            for (const auto& alert : alerts) {
                alert_json["alerts"].append(alert);
                ROS_WARN("%s", alert.c_str());
            }
            
            std_msgs::String alert_msg;
            Json::StreamWriterBuilder builder;
            alert_msg.data = Json::writeString(builder, alert_json);
            alert_pub_.publish(alert_msg);
        }
    }
    
    void publishFinalPerformanceReport() {
        magv_vln_msgs::PerformanceMetrics final_metrics;
        calculateCurrentMetrics(final_metrics);
        
        ROS_INFO("=== FINAL PERFORMANCE REPORT ===");
        ROS_INFO("Total time: %.1f seconds", final_metrics.time_elapsed);
        ROS_INFO("Distance traveled: %.2f meters", final_metrics.distance_traveled);
        ROS_INFO("Path efficiency: %.3f", final_metrics.path_efficiency);
        ROS_INFO("Final SPL score: %.3f", final_metrics.current_spl_score);
        ROS_INFO("Marker detected: %s", final_metrics.marker_detected ? "Yes" : "No");
        if (final_metrics.marker_detected) {
            ROS_INFO("Final marker distance: %.2f meters", final_metrics.marker_distance);
            ROS_INFO("Task completed: %s", final_metrics.within_completion_zone ? "Yes" : "No");
        }
        ROS_INFO("Average control frequency: %.1f Hz", final_metrics.control_frequency);
        ROS_INFO("Average processing delay: %.3f seconds", final_metrics.processing_delay);
        ROS_INFO("================================");
        
        // Publish final metrics with special flag
        final_metrics.header.frame_id = "final_report";
        performance_pub_.publish(final_metrics);
    }
    
    void updatePerformanceHistory(std::deque<double>& history, double value) {
        history.push_back(value);
        if (history.size() > performance_history_size_) {
            history.pop_front();
        }
    }
    
    double calculateAverageFromHistory(const std::deque<double>& history) {
        if (history.empty()) return 0.0;
        
        double sum = 0.0;
        for (double value : history) {
            sum += value;
        }
        return sum / history.size();
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "performance_monitor_node");
    
    try {
        PerformanceMonitorNode node;
        ros::spin();
    } catch (const std::exception& e) {
        ROS_ERROR("Performance monitor node error: %s", e.what());
        return 1;
    }
    
    return 0;
}
