cmake_minimum_required(VERSION 3.0.2)
project(magv_vln_msgs)

find_package(
  catkin REQUIRED COMPONENTS 
    message_generation 
    std_msgs 
    geometry_msgs
    nav_msgs
    actionlib_msgs
)

add_message_files(
  DIRECTORY msg
    FILES
      PositionCommand.msg
      VehicleStatus.msg
      SubTask.msg
      ValueMap.msg
      PathPoint.msg
      DetectedObject.msg
      DetectedObjectArray.msg
      BoundingBox2D.msg
      Detection2D.msg
      Detection2DArray.msg
      OptimalPath.msg
      TaskOptimization.msg
      PerformanceMetrics.msg
)

generate_messages(
  DEPENDENCIES 
    std_msgs 
    geometry_msgs 
    nav_msgs
)

catkin_package(
  CATKIN_DEPENDS 
    message_runtime 
    std_msgs 
    geometry_msgs
    nav_msgs
)
