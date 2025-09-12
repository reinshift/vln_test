cmake_minimum_required(VERSION 3.0.2)
project(vln_mock)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  geometry_msgs
  nav_msgs
  sensor_msgs
  tf
  tf2_ros
  magv_vln_msgs
)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
)

# Create src directory if it doesn't exist
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src)

# Install Python scripts
catkin_install_python(PROGRAMS
  scripts/core_node.py
  scripts/controller.py
  scripts/vehicle_status_manager.py
  scripts/instruction_latch_bridge.py
  scripts/odom_tf_broadcaster.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# Install launch files
install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
