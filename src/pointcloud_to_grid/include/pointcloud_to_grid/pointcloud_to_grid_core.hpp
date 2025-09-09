#pragma once
#include <ros/ros.h>
#include <nav_msgs/OccupancyGrid.h>
#include <algorithm>
#include <cmath>
class PointXY{
public: 
  int x;
  int y;
};

class PointXYZI{
public: 
  double x;
  double y;
  double z;
  double intensity;
};

class GridMap{
  public:
    double position_x;
    double position_y;
    double cell_size;
    double length_x;
    double length_y;
    std::string cloud_in_topic;
    std::string frame_out;
    std::string mapi_topic_name;
    std::string maph_topic_name;
    double topleft_x;
    double topleft_y;
    double bottomright_x;
    double bottomright_y;
    int cell_num_x;
    int cell_num_y;
    double intensity_factor;
    double height_factor;

    void initGrid(nav_msgs::OccupancyGridPtr grid) {
      // Fallbacks to ensure valid metadata
      grid->header.frame_id = frame_out.empty() ? std::string("map") : frame_out;
      grid->info.resolution = (cell_size > 0.0) ? cell_size : 0.5;
      grid->info.width = std::max(1, cell_num_x);
      grid->info.height = std::max(1, cell_num_y);
      // Set origin to center the map around the robot position
      grid->info.origin.position.x = position_x - length_x / 2.0;
      grid->info.origin.position.y = position_y - length_y / 2.0;
      grid->info.origin.position.z = 0;
      grid->info.origin.orientation.w = 1;
      grid->data.assign(static_cast<size_t>(grid->info.width) * static_cast<size_t>(grid->info.height), -1);
    }

    void paramRefresh(){
      if (cell_size <= 0.0) {
        ROS_WARN_THROTTLE(5.0, "cell_size (%.3f) invalid, resetting to 0.5 m", cell_size);
        cell_size = 0.5;
      }
      topleft_x = position_x - length_x / 2.0;
      bottomright_x = position_x + length_x / 2.0;
      topleft_y = position_y + length_y / 2.0;
      bottomright_y = position_y - length_y / 2.0;
      cell_num_x = std::max(1, static_cast<int>(std::round(length_x / cell_size)));
      cell_num_y = std::max(1, static_cast<int>(std::round(length_y / cell_size)));
      ROS_INFO_STREAM("Grid Cells: " << cell_num_x << "x" << cell_num_y
                      << ", res= " << cell_size << " m/cell, frame= " << (frame_out.empty()?"map":frame_out)
                      << ", Subscribed to: " << cloud_in_topic);
    }

    // number of cells
    int getSize(){
      return cell_num_x * cell_num_y;
    }
    
    // number of cells
    int getSizeX(){
      return cell_num_x;
    }

    // number of cells
    int getSizeY(){
      return cell_num_y;
    }

    // length [m] meters
    double getLengthX(){
      return length_x;
    }

    // length [m] meters
    double getLengthY(){
      return length_y;
    }

    // resolution [m/cell] size of a single cell
    double getResolution(){
      return cell_size;
    }
};