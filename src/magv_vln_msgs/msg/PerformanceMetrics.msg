# Performance Monitor Message
# Real-time performance monitoring for competition

# Header for timestamp
std_msgs/Header header

# Real-time metrics
float32 current_spl_score         # Current SPL score
float32 time_elapsed             # Time elapsed since task start (seconds)
float32 time_remaining           # Estimated time remaining (seconds)
float32 completion_probability   # Probability of successful completion

# Path efficiency metrics
float32 distance_traveled        # Distance traveled so far (meters)
float32 optimal_distance         # Optimal distance to current point
float32 path_efficiency         # Current path efficiency ratio

# System performance
float32 processing_delay         # Average processing delay (seconds)
float32 control_frequency       # Actual control loop frequency (Hz)
string bottleneck_component     # Component causing performance bottleneck

# Competition specific
bool marker_detected            # Whether target marker is detected
float32 marker_distance         # Distance to detected marker (meters)
bool within_completion_zone     # Whether robot is within 2m of marker center
