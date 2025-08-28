cmake_minimum_required(VERSION 3.0.2)
project(vln_mock)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  geometry_msgs
  nav_msgs
  actionlib
  magv_vln_msgs
  sensor_msgs
  pcl_conversions
  pcl_ros
)

find_package(PCL REQUIRED)

catkin_package(
)

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)

link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# Create src directory if it doesn't exist
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src)

# Python scripts are installed automatically by catkin

catkin_install_python(PROGRAMS
  scripts/core_node.py
  scripts/controller.py
  scripts/vehicle_status_manager.py
  scripts/ground_dino_simulator.py
  scripts/test_image_publisher.py
  scripts/test_instruction_publisher.py
  scripts/test_occupancy_grid_publisher.py
  scripts/test_odom_publisher.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)



# Install launch files
install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
