# Vehicle Status Message
# Defines the current operational state of the vehicle

# Header for timestamp and frame information
std_msgs/Header header

# Current vehicle state
uint8 state

# State constants
uint8 STATE_INITIALIZING = 0
uint8 STATE_EXPLORATION = 1
uint8 STATE_NAVIGATION = 2
uint8 STATE_IDLE = 3
uint8 STATE_ERROR = 4
uint8 STATE_EMERGENCY_STOP = 5

# State description (human-readable)
string state_description

# Additional status information
bool is_moving
bool has_goal
bool sensors_ready
bool navigation_ready

# Current position (if available)
geometry_msgs/Point current_position

# Current goal (if any)
geometry_msgs/Point target_position

# Timestamp of last state change
time last_state_change

# Additional diagnostic information
string diagnostic_info

# Current subtask information (for VLN)
string current_subtask_json
int32 current_subtask_index
int32 total_subtasks

# Task completion status
bool task_completed
string completion_message
