cmake_minimum_required(VERSION 3.0.2)
project(vln_runtime)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  nav_msgs
  rospy
  sensor_msgs
  std_msgs
  vln_msgs
)

catkin_python_setup()

catkin_package(
  CATKIN_DEPENDS geometry_msgs nav_msgs rospy sensor_msgs std_msgs vln_msgs
)

catkin_install_python(PROGRAMS
  scripts/controller_node.py
  scripts/mission_manager_node.py
  scripts/planner_node.py
  scripts/safety_supervisor_node.py
  scripts/semantic_map_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

