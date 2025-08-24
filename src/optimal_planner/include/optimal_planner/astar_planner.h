#ifndef OPTIMAL_PLANNER_ASTAR_PLANNER_H
#define OPTIMAL_PLANNER_ASTAR_PLANNER_H

#include <ros/ros.h>
#include <nav_msgs/OccupancyGrid.h>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/PoseStamped.h>
#include <magv_vln_msgs/OptimalPath.h>
#include <vector>
#include <queue>
#include <unordered_map>
#include <chrono>

namespace optimal_planner {

struct Node {
    int x, y;
    double g_cost, h_cost, f_cost;
    Node* parent;
    
    Node(int x = 0, int y = 0, double g = 0, double h = 0, Node* p = nullptr)
        : x(x), y(y), g_cost(g), h_cost(h), f_cost(g + h), parent(p) {}
    
    bool operator>(const Node& other) const {
        return f_cost > other.f_cost;
    }
};

class AStarPlanner {
public:
    AStarPlanner();
    ~AStarPlanner();
    
    /**
     * @brief Plan optimal path using A* algorithm
     * @param occupancy_grid The occupancy grid map
     * @param start Start position in world coordinates
     * @param goal Goal position in world coordinates
     * @param path Output optimal path
     * @return true if path found, false otherwise
     */
    bool planPath(const nav_msgs::OccupancyGrid& occupancy_grid,
                  const geometry_msgs::Point& start,
                  const geometry_msgs::Point& goal,
                  magv_vln_msgs::OptimalPath& path);
    
    /**
     * @brief Set planning parameters
     */
    void setInflationRadius(double radius) { inflation_radius_ = radius; }
    void setHeuristicWeight(double weight) { heuristic_weight_ = weight; }
    void setMaxIterations(int max_iter) { max_iterations_ = max_iter; }
    
private:
    // Planning parameters
    double inflation_radius_;      // Obstacle inflation radius (meters)
    double heuristic_weight_;      // A* heuristic weight
    int max_iterations_;           // Maximum planning iterations
    
    // Grid processing
    nav_msgs::OccupancyGrid inflated_grid_;
    
    /**
     * @brief Convert world coordinates to grid coordinates
     */
    bool worldToGrid(const geometry_msgs::Point& world_point, 
                     const nav_msgs::OccupancyGrid& grid,
                     int& grid_x, int& grid_y);
    
    /**
     * @brief Convert grid coordinates to world coordinates
     */
    geometry_msgs::Point gridToWorld(int grid_x, int grid_y,
                                    const nav_msgs::OccupancyGrid& grid);
    
    /**
     * @brief Check if grid cell is valid and free
     */
    bool isValidCell(int x, int y, const nav_msgs::OccupancyGrid& grid);
    
    /**
     * @brief Calculate heuristic distance (Euclidean)
     */
    double calculateHeuristic(int x1, int y1, int x2, int y2);
    
    /**
     * @brief Inflate obstacles in occupancy grid
     */
    void inflateObstacles(const nav_msgs::OccupancyGrid& original_grid);
    
    /**
     * @brief Reconstruct path from goal to start
     */
    void reconstructPath(Node* goal_node, 
                        const nav_msgs::OccupancyGrid& grid,
                        magv_vln_msgs::OptimalPath& path);
    
    /**
     * @brief Get neighboring cells
     */
    std::vector<std::pair<int, int>> getNeighbors(int x, int y);
    
    /**
     * @brief Calculate path statistics
     */
    void calculatePathMetrics(magv_vln_msgs::OptimalPath& path,
                             const geometry_msgs::Point& start,
                             const geometry_msgs::Point& goal,
                             double computation_time);
};

} // namespace optimal_planner

#endif // OPTIMAL_PLANNER_ASTAR_PLANNER_H
