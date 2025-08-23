# Represents a single object detected in the environment.

std_msgs/Header header

# Class label of the detected object (e.g., 'tree', 'bench').
string label

# Position of the object in the world frame.
geometry_msgs/Point position

# Confidence score of the detection.
float32 confidence

