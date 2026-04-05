cmake_minimum_required(VERSION 3.0.2)
project(vln_eval)

find_package(catkin REQUIRED)

catkin_python_setup()

catkin_package()

catkin_install_python(PROGRAMS
  scripts/run_px4_episode.py
  scripts/run_px4_eval.py
  scripts/run_uav_eval.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
