# SubTask Message for VLN
# Represents a single navigation subtask

# Header for timestamp
std_msgs/Header header

# Subtask ID/index
int32 subtask_id

# Direction for this subtask
string direction  # forward, backward, left, right

# Target object/goal for this subtask (can be empty)
string goal

# Estimated completion status
bool completed

# Additional parameters
float64 distance_threshold  # Distance threshold for completion
float64 angle_threshold     # Angle threshold for turns
