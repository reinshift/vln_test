# Describes a single 2D object detection.

std_msgs/Header header

# The ID of the result, equivalent to the class ID in the model.
string id

# The label of the detected object.
string label

# The confidence score of the detection.
float64 score

# The 2D bounding box of the detected object.
magv_vln_msgs/BoundingBox2D bbox

