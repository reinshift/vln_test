# Optimal Path Message for improved path planning
# Represents a complete path with optimization metrics

# Header for timestamp
std_msgs/Header header

# Array of path points
PathPoint[] waypoints

# Path metrics for SPL optimization
float32 total_length      # Total path length in meters
float32 optimal_length    # Theoretical shortest path length
float32 spl_score        # Success weighted by Path Length score
float32 efficiency      # Path efficiency ratio

# Path generation method
string planning_method   # "astar", "rrt_star", "value_map", "hybrid"
float32 computation_time # Time taken to compute path (seconds)

# Safety metrics
bool obstacle_free      # Whether path is clear of static obstacles
float32 clearance_min   # Minimum clearance to obstacles (meters)
