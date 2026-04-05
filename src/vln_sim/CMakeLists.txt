cmake_minimum_required(VERSION 3.0.2)
project(vln_sim)

find_package(catkin REQUIRED COMPONENTS
  gazebo_msgs
  geometry_msgs
  nav_msgs
  rospy
  sensor_msgs
  std_msgs
  vln_msgs
)

catkin_python_setup()

catkin_package(
  CATKIN_DEPENDS gazebo_msgs geometry_msgs nav_msgs rospy sensor_msgs std_msgs vln_msgs
)

catkin_install_python(PROGRAMS
  scripts/episode_reset_service_node.py
  scripts/mock_uav_bridge_node.py
  scripts/px4_uav_bridge_node.py
  scripts/spawn_vehicle_node.py
  scripts/world_profile_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
