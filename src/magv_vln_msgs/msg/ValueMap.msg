# Value Map Message for VLN
# Represents a value map where each grid cell has an associated value

# Header for timestamp and frame information
std_msgs/Header header

# Map metadata (similar to OccupancyGrid)
nav_msgs/MapMetaData info

# Value data for each cell (float values instead of occupancy probabilities)
# The values are stored in row-major order, starting with (0,0)
# Values represent the desirability/cost of each cell for navigation
float32[] data
