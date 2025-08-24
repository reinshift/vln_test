#include <ros/ros.h>
#include <std_msgs/String.h>
#include <magv_vln_msgs/TaskOptimization.h>
#include <magv_vln_msgs/VehicleStatus.h>
#include <geometry_msgs/Point.h>
#include <nav_msgs/Odometry.h>
#include <json/json.h>
#include <vector>
#include <algorithm>
#include <unordered_map>

struct SubTask {
    std::string id;
    std::string action;
    std::string direction;
    std::string goal;
    geometry_msgs::Point estimated_position;
    double estimated_time;
    double estimated_distance;
    int priority;
};

class TaskOptimizerNode {
public:
    TaskOptimizerNode() {
        ros::NodeHandle nh;
        ros::NodeHandle pnh("~");
        
        // Parameters
        pnh.param("optimization_enabled", optimization_enabled_, true);
        pnh.param("max_planning_time", max_planning_time_, 5.0);
        pnh.param("timeout_buffer", timeout_buffer_, 10.0); // Safety buffer before timeout
        
        // Publishers
        optimized_tasks_pub_ = nh.advertise<std_msgs::String>("/optimized_subtasks", 1);
        optimization_pub_ = nh.advertise<magv_vln_msgs::TaskOptimization>("/task_optimization", 1);
        
        // Subscribers
        subtasks_sub_ = nh.subscribe("/subtasks", 1, &TaskOptimizerNode::subtasksCallback, this);
        vln_status_sub_ = nh.subscribe("/vln_status", 1, &TaskOptimizerNode::vlnStatusCallback, this);
        odometry_sub_ = nh.subscribe("/magv/odometry/gt", 1, &TaskOptimizerNode::odometryCallback, this);
        
        // Initialize
        current_position_.x = 0.0;
        current_position_.y = 0.0;
        current_position_.z = 0.0;
        
        ROS_INFO("Task Optimizer Node initialized");
    }

private:
    ros::Publisher optimized_tasks_pub_;
    ros::Publisher optimization_pub_;
    
    ros::Subscriber subtasks_sub_;
    ros::Subscriber vln_status_sub_;
    ros::Subscriber odometry_sub_;
    
    bool optimization_enabled_;
    double max_planning_time_;
    double timeout_buffer_;
    
    geometry_msgs::Point current_position_;
    std::vector<SubTask> original_tasks_;
    std::vector<SubTask> optimized_tasks_;
    
    void subtasksCallback(const std_msgs::String::ConstPtr& msg) {
        if (!optimization_enabled_) {
            // If optimization disabled, pass through original tasks
            optimized_tasks_pub_.publish(*msg);
            return;
        }
        
        // Parse subtasks
        if (parseSubtasks(msg->data)) {
            // Optimize task sequence
            optimizeTaskSequence();
            
            // Publish optimized tasks
            publishOptimizedTasks();
            
            // Publish optimization metrics
            publishOptimizationMetrics();
        } else {
            ROS_WARN("Failed to parse subtasks, using original order");
            optimized_tasks_pub_.publish(*msg);
        }
    }
    
    void vlnStatusCallback(const magv_vln_msgs::VehicleStatus::ConstPtr& msg) {
        current_position_ = msg->current_position;
    }
    
    void odometryCallback(const nav_msgs::Odometry::ConstPtr& msg) {
        current_position_ = msg->pose.pose.position;
    }
    
    bool parseSubtasks(const std::string& subtasks_json) {
        Json::Value root;
        Json::Reader reader;
        
        if (!reader.parse(subtasks_json, root)) {
            ROS_ERROR("Failed to parse subtasks JSON");
            return false;
        }
        
        original_tasks_.clear();
        
        if (root.isArray()) {
            // Handle array format
            for (const auto& task_json : root) {
                SubTask task;
                parseTaskFromJson(task_json, task);
                original_tasks_.push_back(task);
            }
        } else if (root.isObject()) {
            // Handle object format with numbered keys
            for (const auto& key : root.getMemberNames()) {
                if (key.find("subtask") != std::string::npos) {
                    SubTask task;
                    task.id = key;
                    
                    if (root[key].isString()) {
                        task.direction = root[key].asString();
                        task.action = "navigate";
                    } else if (root[key].isObject()) {
                        parseTaskFromJson(root[key], task);
                    }
                    
                    original_tasks_.push_back(task);
                }
            }
        }
        
        // Estimate positions and costs for each task
        estimateTaskCosts();
        
        return !original_tasks_.empty();
    }
    
    void parseTaskFromJson(const Json::Value& task_json, SubTask& task) {
        if (task_json.isMember("action")) {
            task.action = task_json["action"].asString();
        }
        if (task_json.isMember("direction")) {
            task.direction = task_json["direction"].asString();
        }
        if (task_json.isMember("goal")) {
            task.goal = task_json["goal"].asString();
        }
        if (task_json.isMember("priority")) {
            task.priority = task_json["priority"].asInt();
        } else {
            task.priority = 1; // Default priority
        }
    }
    
    void estimateTaskCosts() {
        geometry_msgs::Point current_pos = current_position_;
        
        for (auto& task : original_tasks_) {
            // Estimate position based on direction and goal
            estimateTaskPosition(task, current_pos);
            
            // Estimate time and distance
            double distance = calculateDistance(current_pos, task.estimated_position);
            task.estimated_distance = distance;
            task.estimated_time = estimateExecutionTime(task, distance);
            
            // Update current position for next task
            current_pos = task.estimated_position;
        }
    }
    
    void estimateTaskPosition(SubTask& task, const geometry_msgs::Point& current_pos) {
        // Simple position estimation based on direction
        double step_size = 3.0; // Estimated 3 meters per navigation step
        
        task.estimated_position = current_pos;
        
        if (task.direction == "forward") {
            task.estimated_position.x += step_size;
        } else if (task.direction == "backward") {
            task.estimated_position.x -= step_size;
        } else if (task.direction == "left") {
            task.estimated_position.y += step_size;
        } else if (task.direction == "right") {
            task.estimated_position.y -= step_size;
        }
        
        // Add some randomness for goals
        if (!task.goal.empty() && task.goal != "null") {
            task.estimated_position.x += (rand() % 200 - 100) / 100.0; // ±1m variation
            task.estimated_position.y += (rand() % 200 - 100) / 100.0;
        }
    }
    
    double estimateExecutionTime(const SubTask& task, double distance) {
        double base_time = 0.0;
        
        // Base time for different actions
        if (task.action == "navigate") {
            base_time = 15.0; // Base navigation time
            base_time += distance * 2.0; // 2 seconds per meter (conservative)
        } else if (task.action == "turn") {
            base_time = 5.0; // Turn time
        } else if (task.action == "search") {
            base_time = 20.0; // Search time
        }
        
        // Add scanning time if goal is specified
        if (!task.goal.empty() && task.goal != "null") {
            base_time += 10.0; // Extra time for goal detection
        }
        
        return base_time;
    }
    
    void optimizeTaskSequence() {
        optimized_tasks_ = original_tasks_;
        
        if (optimized_tasks_.size() <= 1) {
            return; // No optimization needed
        }
        
        // Simple optimization strategies:
        // 1. Group similar directional movements
        // 2. Prioritize tasks with higher priority
        // 3. Minimize total path length
        
        // Strategy 1: Sort by priority first
        std::stable_sort(optimized_tasks_.begin(), optimized_tasks_.end(),
                        [](const SubTask& a, const SubTask& b) {
                            return a.priority > b.priority;
                        });
        
        // Strategy 2: Apply nearest neighbor heuristic for same priority tasks
        optimizeByNearestNeighbor();
        
        // Strategy 3: Check for opportunity to combine tasks
        combineCompatibleTasks();
    }
    
    void optimizeByNearestNeighbor() {
        if (optimized_tasks_.size() <= 2) return;
        
        std::vector<SubTask> reordered;
        std::vector<bool> visited(optimized_tasks_.size(), false);
        
        geometry_msgs::Point current_pos = current_position_;
        reordered.push_back(optimized_tasks_[0]);
        visited[0] = true;
        current_pos = optimized_tasks_[0].estimated_position;
        
        for (size_t i = 1; i < optimized_tasks_.size(); ++i) {
            int next_idx = -1;
            double min_distance = std::numeric_limits<double>::max();
            
            for (size_t j = 1; j < optimized_tasks_.size(); ++j) {
                if (!visited[j]) {
                    double dist = calculateDistance(current_pos, optimized_tasks_[j].estimated_position);
                    if (dist < min_distance) {
                        min_distance = dist;
                        next_idx = j;
                    }
                }
            }
            
            if (next_idx >= 0) {
                reordered.push_back(optimized_tasks_[next_idx]);
                visited[next_idx] = true;
                current_pos = optimized_tasks_[next_idx].estimated_position;
            }
        }
        
        if (reordered.size() == optimized_tasks_.size()) {
            optimized_tasks_ = reordered;
        }
    }
    
    void combineCompatibleTasks() {
        // Look for consecutive tasks that can be combined
        std::vector<SubTask> combined;
        
        for (size_t i = 0; i < optimized_tasks_.size(); ++i) {
            SubTask current_task = optimized_tasks_[i];
            
            // Look ahead for combinable tasks
            while (i + 1 < optimized_tasks_.size() && 
                   canCombineTasks(current_task, optimized_tasks_[i + 1])) {
                i++;
                // Combine the tasks (simple approach: extend the direction)
                current_task.direction += "_" + optimized_tasks_[i].direction;
                current_task.estimated_position = optimized_tasks_[i].estimated_position;
                current_task.estimated_time += optimized_tasks_[i].estimated_time * 0.8; // 20% efficiency gain
                current_task.estimated_distance += optimized_tasks_[i].estimated_distance;
            }
            
            combined.push_back(current_task);
        }
        
        if (combined.size() < optimized_tasks_.size()) {
            optimized_tasks_ = combined;
            ROS_INFO("Combined %d tasks into %d optimized tasks", 
                     (int)original_tasks_.size(), (int)optimized_tasks_.size());
        }
    }
    
    bool canCombineTasks(const SubTask& task1, const SubTask& task2) {
        // Simple combination rules
        if (task1.action != task2.action) return false;
        if (task1.action != "navigate") return false;
        if (!task1.goal.empty() && task1.goal != "null") return false; // Don't combine if first has specific goal
        
        // Check if directions are compatible (same or sequential)
        return (task1.direction == task2.direction) ||
               (task1.direction == "forward" && task2.direction == "forward");
    }
    
    void publishOptimizedTasks() {
        Json::Value optimized_json(Json::arrayValue);
        
        for (size_t i = 0; i < optimized_tasks_.size(); ++i) {
            Json::Value task_json;
            task_json["subtask_" + std::to_string(i+1)] = optimized_tasks_[i].direction;
            task_json["action"] = optimized_tasks_[i].action;
            if (!optimized_tasks_[i].goal.empty()) {
                task_json["goal"] = optimized_tasks_[i].goal;
            }
            task_json["estimated_time"] = optimized_tasks_[i].estimated_time;
            task_json["priority"] = optimized_tasks_[i].priority;
            
            optimized_json.append(task_json);
        }
        
        std_msgs::String msg;
        Json::StreamWriterBuilder builder;
        msg.data = Json::writeString(builder, optimized_json);
        optimized_tasks_pub_.publish(msg);
        
        ROS_INFO("Published %d optimized subtasks", (int)optimized_tasks_.size());
    }
    
    void publishOptimizationMetrics() {
        magv_vln_msgs::TaskOptimization metrics;
        metrics.header.stamp = ros::Time::now();
        metrics.header.frame_id = "map";
        
        // Calculate total estimates
        double total_time = 0.0;
        double total_distance = 0.0;
        
        for (const auto& task : optimized_tasks_) {
            total_time += task.estimated_time;
            total_distance += task.estimated_distance;
        }
        
        metrics.estimated_completion_time = total_time;
        metrics.total_path_length = total_distance;
        
        // Calculate efficiency score
        double original_time = 0.0;
        double original_distance = 0.0;
        for (const auto& task : original_tasks_) {
            original_time += task.estimated_time;
            original_distance += task.estimated_distance;
        }
        
        metrics.efficiency_score = original_time > 0 ? total_time / original_time : 1.0;
        
        // Predict SPL score (simplified)
        geometry_msgs::Point start = current_position_;
        geometry_msgs::Point end = optimized_tasks_.empty() ? start : optimized_tasks_.back().estimated_position;
        double optimal_distance = calculateDistance(start, end);
        metrics.spl_prediction = optimal_distance > 0 ? optimal_distance / total_distance : 1.0;
        
        // Timeout probability
        double timeout_threshold = 100.0 - timeout_buffer_; // Safety buffer
        metrics.timeout_probability = total_time > timeout_threshold ? 
                                     (total_time - timeout_threshold) / timeout_threshold : 0.0;
        metrics.timeout_probability = std::min(1.0, std::max(0.0, metrics.timeout_probability));
        
        // Optimization reasons
        if (optimized_tasks_.size() < original_tasks_.size()) {
            metrics.optimization_reasons.push_back("Combined compatible tasks");
        }
        if (optimized_tasks_.size() > 1) {
            metrics.optimization_reasons.push_back("Optimized task sequence");
        }
        
        metrics.requires_replanning = metrics.timeout_probability > 0.7;
        
        optimization_pub_.publish(metrics);
        
        ROS_INFO("Task optimization - Time: %.1fs, Distance: %.1fm, SPL: %.3f, Timeout risk: %.1f%%",
                 total_time, total_distance, metrics.spl_prediction, 
                 metrics.timeout_probability * 100.0);
    }
    
    double calculateDistance(const geometry_msgs::Point& p1, const geometry_msgs::Point& p2) {
        double dx = p2.x - p1.x;
        double dy = p2.y - p1.y;
        return std::sqrt(dx*dx + dy*dy);
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "task_optimizer_node");
    
    try {
        TaskOptimizerNode node;
        ros::spin();
    } catch (const std::exception& e) {
        ROS_ERROR("Task optimizer node error: %s", e.what());
        return 1;
    }
    
    return 0;
}
