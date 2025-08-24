# Task Optimizer Message
# Used for optimizing task execution sequences

# Header for timestamp
std_msgs/Header header

# Task optimization metrics
float32 estimated_completion_time   # Estimated time to complete all tasks (seconds)
float32 total_path_length          # Total estimated path length
float32 efficiency_score           # Overall efficiency score (0-1)

# Task sequence optimization
string[] optimized_sequence        # Optimized task execution order
string[] optimization_reasons      # Reasons for each optimization

# Performance predictions
float32 spl_prediction            # Predicted SPL score
float32 timeout_probability       # Probability of timeout (0-1)
bool requires_replanning          # Whether replanning is needed
