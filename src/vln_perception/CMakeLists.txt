cmake_minimum_required(VERSION 3.0.2)
project(vln_perception)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  rospy
  std_msgs
  vln_msgs
)

catkin_python_setup()

catkin_package(
  CATKIN_DEPENDS geometry_msgs rospy std_msgs vln_msgs
)

catkin_install_python(PROGRAMS
  scripts/aruco_node.py
  scripts/instruction_gateway_node.py
  scripts/landmark_grounding_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

