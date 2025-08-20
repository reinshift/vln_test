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

# Add executable for vehicle status manager
add_executable(vehicle_status_manager src/vehicle_status_manager.cpp)
target_link_libraries(vehicle_status_manager ${catkin_LIBRARIES} ${PCL_LIBRARIES})
add_dependencies(vehicle_status_manager ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

catkin_install_python(PROGRAMS
  scripts/vln_demo_mock.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# Install C++ executables
install(TARGETS vehicle_status_manager
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
