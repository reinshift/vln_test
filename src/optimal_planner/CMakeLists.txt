cmake_minimum_required(VERSION 3.0.2)
project(optimal_planner)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimization flags for competition performance
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -DNDEBUG")

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  geometry_msgs
  nav_msgs
  magv_vln_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
  cv_bridge
  image_transport
  sensor_msgs
  pcl_conversions
  pcl_ros
)

# Find required packages
find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES optimal_planner_core
  CATKIN_DEPENDS roscpp rospy std_msgs geometry_msgs nav_msgs magv_vln_msgs
  DEPENDS PCL OpenCV Eigen3
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)

link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# Core planning library
add_library(optimal_planner_core
  src/astar_planner.cpp
  src/rrt_star_planner.cpp
  src/path_optimizer.cpp
  src/dynamic_window_approach.cpp
  src/competition_planner.cpp
)

target_link_libraries(optimal_planner_core
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${OpenCV_LIBRARIES}
)

# Main planning node
add_executable(optimal_planner_node
  src/optimal_planner_node.cpp
)

target_link_libraries(optimal_planner_node
  optimal_planner_core
  ${catkin_LIBRARIES}
)

# Task optimizer node
add_executable(task_optimizer_node
  src/task_optimizer_node.cpp
)

target_link_libraries(task_optimizer_node
  optimal_planner_core
  ${catkin_LIBRARIES}
)

# Performance monitor node
add_executable(performance_monitor_node
  src/performance_monitor_node.cpp
)

target_link_libraries(performance_monitor_node
  ${catkin_LIBRARIES}
)
