# Path Point Message for VLN
# Represents a single waypoint in a navigation path

# Header for timestamp
std_msgs/Header header

# Position in world coordinates
geometry_msgs/Point position

# Orientation at this waypoint
geometry_msgs/Quaternion orientation

# Whether this is the final destination
bool is_final_goal

# Associated value/cost for reaching this point
float32 value

# Tolerance for reaching this waypoint
float32 position_tolerance
float32 orientation_tolerance
