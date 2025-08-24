#include "optimal_planner/astar_planner.h"
#include <algorithm>
#include <cmath>
#include <tf2/utils.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

namespace optimal_planner {

AStarPlanner::AStarPlanner() 
    : inflation_radius_(0.4)  // Default robot radius + safety margin
    , heuristic_weight_(1.2)  // Slightly weighted A* for faster planning
    , max_iterations_(10000)  // Maximum iterations to prevent infinite loops
{
}

AStarPlanner::~AStarPlanner() {
}

bool AStarPlanner::planPath(const nav_msgs::OccupancyGrid& occupancy_grid,
                           const geometry_msgs::Point& start,
                           const geometry_msgs::Point& goal,
                           magv_vln_msgs::OptimalPath& path) {
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Clear previous path
    path.waypoints.clear();
    
    // Inflate obstacles
    inflateObstacles(occupancy_grid);
    
    // Convert world coordinates to grid coordinates
    int start_x, start_y, goal_x, goal_y;
    if (!worldToGrid(start, inflated_grid_, start_x, start_y) ||
        !worldToGrid(goal, inflated_grid_, goal_x, goal_y)) {
        ROS_ERROR("Start or goal position is outside the grid");
        return false;
    }
    
    // Check if start and goal are valid
    if (!isValidCell(start_x, start_y, inflated_grid_) ||
        !isValidCell(goal_x, goal_y, inflated_grid_)) {
        ROS_ERROR("Start or goal position is in occupied space");
        return false;
    }
    
    // A* algorithm implementation
    std::priority_queue<Node, std::vector<Node>, std::greater<Node>> open_set;
    std::unordered_map<int, Node> all_nodes;
    std::unordered_map<int, bool> closed_set;
    
    // Create start node
    int start_id = start_y * inflated_grid_.info.width + start_x;
    Node start_node(start_x, start_y, 0.0, 
                    calculateHeuristic(start_x, start_y, goal_x, goal_y));
    
    open_set.push(start_node);
    all_nodes[start_id] = start_node;
    
    int iterations = 0;
    bool path_found = false;
    Node* goal_node = nullptr;
    
    while (!open_set.empty() && iterations < max_iterations_) {
        iterations++;
        
        // Get node with lowest f_cost
        Node current = open_set.top();
        open_set.pop();
        
        int current_id = current.y * inflated_grid_.info.width + current.x;
        
        // Skip if already processed
        if (closed_set.find(current_id) != closed_set.end()) {
            continue;
        }
        
        // Mark as processed
        closed_set[current_id] = true;
        
        // Check if goal reached
        if (current.x == goal_x && current.y == goal_y) {
            path_found = true;
            goal_node = &all_nodes[current_id];
            break;
        }
        
        // Explore neighbors
        auto neighbors = getNeighbors(current.x, current.y);
        for (const auto& neighbor : neighbors) {
            int nx = neighbor.first;
            int ny = neighbor.second;
            int neighbor_id = ny * inflated_grid_.info.width + nx;
            
            // Skip if invalid or already processed
            if (!isValidCell(nx, ny, inflated_grid_) ||
                closed_set.find(neighbor_id) != closed_set.end()) {
                continue;
            }
            
            // Calculate costs
            double dx = nx - current.x;
            double dy = ny - current.y;
            double move_cost = std::sqrt(dx*dx + dy*dy) * inflated_grid_.info.resolution;
            double tentative_g = current.g_cost + move_cost;
            
            // Check if this path to neighbor is better
            auto existing = all_nodes.find(neighbor_id);
            if (existing == all_nodes.end() || tentative_g < existing->second.g_cost) {
                double h_cost = heuristic_weight_ * calculateHeuristic(nx, ny, goal_x, goal_y);
                Node neighbor_node(nx, ny, tentative_g, h_cost, &all_nodes[current_id]);
                
                all_nodes[neighbor_id] = neighbor_node;
                open_set.push(neighbor_node);
            }
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    double computation_time = duration.count() / 1000000.0; // Convert to seconds
    
    if (path_found) {
        reconstructPath(goal_node, inflated_grid_, path);
        calculatePathMetrics(path, start, goal, computation_time);
        ROS_INFO("A* path found with %d waypoints in %.3f seconds", 
                 (int)path.waypoints.size(), computation_time);
        return true;
    } else {
        ROS_WARN("A* failed to find path after %d iterations", iterations);
        return false;
    }
}

bool AStarPlanner::worldToGrid(const geometry_msgs::Point& world_point,
                              const nav_msgs::OccupancyGrid& grid,
                              int& grid_x, int& grid_y) {
    
    double resolution = grid.info.resolution;
    double origin_x = grid.info.origin.position.x;
    double origin_y = grid.info.origin.position.y;
    
    grid_x = static_cast<int>((world_point.x - origin_x) / resolution);
    grid_y = static_cast<int>((world_point.y - origin_y) / resolution);
    
    return (grid_x >= 0 && grid_x < grid.info.width &&
            grid_y >= 0 && grid_y < grid.info.height);
}

geometry_msgs::Point AStarPlanner::gridToWorld(int grid_x, int grid_y,
                                               const nav_msgs::OccupancyGrid& grid) {
    geometry_msgs::Point world_point;
    world_point.x = grid.info.origin.position.x + (grid_x + 0.5) * grid.info.resolution;
    world_point.y = grid.info.origin.position.y + (grid_y + 0.5) * grid.info.resolution;
    world_point.z = 0.0;
    return world_point;
}

bool AStarPlanner::isValidCell(int x, int y, const nav_msgs::OccupancyGrid& grid) {
    if (x < 0 || x >= grid.info.width || y < 0 || y >= grid.info.height) {
        return false;
    }
    
    int index = y * grid.info.width + x;
    return grid.data[index] < 50; // Free space threshold
}

double AStarPlanner::calculateHeuristic(int x1, int y1, int x2, int y2) {
    double dx = x2 - x1;
    double dy = y2 - y1;
    return std::sqrt(dx*dx + dy*dy) * inflated_grid_.info.resolution;
}

void AStarPlanner::inflateObstacles(const nav_msgs::OccupancyGrid& original_grid) {
    inflated_grid_ = original_grid;
    
    int inflation_cells = static_cast<int>(inflation_radius_ / original_grid.info.resolution);
    
    // Create temporary grid for inflation
    std::vector<int8_t> temp_data = original_grid.data;
    
    for (int y = 0; y < original_grid.info.height; ++y) {
        for (int x = 0; x < original_grid.info.width; ++x) {
            int index = y * original_grid.info.width + x;
            
            if (original_grid.data[index] > 50) { // Occupied cell
                // Inflate around this cell
                for (int dy = -inflation_cells; dy <= inflation_cells; ++dy) {
                    for (int dx = -inflation_cells; dx <= inflation_cells; ++dx) {
                        int nx = x + dx;
                        int ny = y + dy;
                        
                        if (nx >= 0 && nx < original_grid.info.width &&
                            ny >= 0 && ny < original_grid.info.height) {
                            
                            double dist = std::sqrt(dx*dx + dy*dy) * original_grid.info.resolution;
                            if (dist <= inflation_radius_) {
                                int nindex = ny * original_grid.info.width + nx;
                                temp_data[nindex] = std::max(temp_data[nindex], static_cast<int8_t>(100));
                            }
                        }
                    }
                }
            }
        }
    }
    
    inflated_grid_.data = temp_data;
}

void AStarPlanner::reconstructPath(Node* goal_node,
                                  const nav_msgs::OccupancyGrid& grid,
                                  magv_vln_msgs::OptimalPath& path) {
    
    std::vector<Node*> node_path;
    Node* current = goal_node;
    
    // Trace back from goal to start
    while (current != nullptr) {
        node_path.push_back(current);
        current = current->parent;
    }
    
    // Reverse to get start-to-goal order
    std::reverse(node_path.begin(), node_path.end());
    
    // Convert to waypoints
    path.waypoints.clear();
    for (size_t i = 0; i < node_path.size(); ++i) {
        magv_vln_msgs::PathPoint waypoint;
        waypoint.header.frame_id = "map";
        waypoint.header.stamp = ros::Time::now();
        
        // Convert grid coordinates to world coordinates
        waypoint.position = gridToWorld(node_path[i]->x, node_path[i]->y, grid);
        
        // Set orientation to face next waypoint
        if (i < node_path.size() - 1) {
            geometry_msgs::Point next_pos = gridToWorld(node_path[i+1]->x, node_path[i+1]->y, grid);
            double yaw = std::atan2(next_pos.y - waypoint.position.y,
                                   next_pos.x - waypoint.position.x);
            tf2::Quaternion quat;
            quat.setRPY(0, 0, yaw);
            waypoint.orientation = tf2::toMsg(quat);
        } else {
            // Last waypoint keeps previous orientation
            if (!path.waypoints.empty()) {
                waypoint.orientation = path.waypoints.back().orientation;
            } else {
                tf2::Quaternion quat;
                quat.setRPY(0, 0, 0);
                waypoint.orientation = tf2::toMsg(quat);
            }
        }
        
        waypoint.is_final_goal = (i == node_path.size() - 1);
        waypoint.position_tolerance = 0.2; // 20cm tolerance
        
        path.waypoints.push_back(waypoint);
    }
}

std::vector<std::pair<int, int>> AStarPlanner::getNeighbors(int x, int y) {
    std::vector<std::pair<int, int>> neighbors;
    
    // 8-connected neighborhood
    for (int dx = -1; dx <= 1; ++dx) {
        for (int dy = -1; dy <= 1; ++dy) {
            if (dx == 0 && dy == 0) continue; // Skip center cell
            neighbors.push_back({x + dx, y + dy});
        }
    }
    
    return neighbors;
}

void AStarPlanner::calculatePathMetrics(magv_vln_msgs::OptimalPath& path,
                                       const geometry_msgs::Point& start,
                                       const geometry_msgs::Point& goal,
                                       double computation_time) {
    
    path.header.frame_id = "map";
    path.header.stamp = ros::Time::now();
    path.planning_method = "astar";
    path.computation_time = computation_time;
    
    // Calculate total path length
    path.total_length = 0.0;
    for (size_t i = 1; i < path.waypoints.size(); ++i) {
        double dx = path.waypoints[i].position.x - path.waypoints[i-1].position.x;
        double dy = path.waypoints[i].position.y - path.waypoints[i-1].position.y;
        path.total_length += std::sqrt(dx*dx + dy*dy);
    }
    
    // Calculate optimal (straight-line) distance
    double dx = goal.x - start.x;
    double dy = goal.y - start.y;
    path.optimal_length = std::sqrt(dx*dx + dy*dy);
    
    // Calculate efficiency and SPL score
    if (path.optimal_length > 0) {
        path.efficiency = path.optimal_length / path.total_length;
        path.spl_score = path.efficiency; // Simplified SPL calculation
    } else {
        path.efficiency = 1.0;
        path.spl_score = 1.0;
    }
    
    // Check obstacle clearance
    path.obstacle_free = true; // Simplified - would need proper collision checking
    path.clearance_min = inflation_radius_; // Conservative estimate
}

} // namespace optimal_planner
