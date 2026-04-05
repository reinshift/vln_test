cmake_minimum_required(VERSION 3.0.2)
project(vln_msgs)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  message_generation
  nav_msgs
  std_msgs
)

add_message_files(FILES
  EpisodeReport.msg
  LandmarkObservation.msg
  Mission.msg
  MissionStep.msg
  RuntimeState.msg
  SafetyEvent.msg
  SemanticLandmark.msg
  Trajectory.msg
)

add_service_files(FILES
  ResetEpisode.srv
)

generate_messages(DEPENDENCIES
  geometry_msgs
  nav_msgs
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS geometry_msgs message_runtime nav_msgs std_msgs
)

