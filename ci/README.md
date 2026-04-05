# CI Notes

- `pr.yml`
  - Provisions an external PX4 checkout, installs MAVROS/Gazebo ROS dependencies, then runs build, tests, and PX4 sim smoke.
- `nightly.yml`
  - Runs the same PX4-backed baseline plus full PX4 episode regression.
