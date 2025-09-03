#include <ros/ros.h>
#include <nav_msgs/OccupancyGrid.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/conversions.h>
#include <pcl_ros/point_cloud.h>
#include <pointcloud_to_grid/pointcloud_to_grid_core.hpp>
#include <pointcloud_to_grid/MyParamsConfig.h>
#include <dynamic_reconfigure/server.h>
 

// Global variables
nav_msgs::OccupancyGridPtr intensity_grid(new nav_msgs::OccupancyGrid);
nav_msgs::OccupancyGridPtr height_grid(new nav_msgs::OccupancyGrid);
GridMap grid_map;
ros::Publisher pub_igrid, pub_hgrid;
ros::Subscriber sub_pc2;
ros::NodeHandle* nh_ptr = nullptr;
std::string last_cloud_topic = "";
std::string last_igrid_topic = "";
std::string last_hgrid_topic = "";

// Forward declaration (subscribe to sensor_msgs::PointCloud2)
void pointcloudCallback(const sensor_msgs::PointCloud2::ConstPtr &msg);

PointXY getIndex(double x, double y){
  PointXY ret;
  ret.x = static_cast<int>((x - grid_map.topleft_x) / grid_map.cell_size);
  ret.y = static_cast<int>((grid_map.topleft_y - y) / grid_map.cell_size);
  return ret;
}

void paramsCallback(my_dyn_rec::MyParamsConfig &config, uint32_t level)
{
  grid_map.cell_size = config.cell_size;
  grid_map.position_x = config.position_x;
  grid_map.position_y = config.position_y;
  grid_map.length_x = config.length_x;
  grid_map.length_y = config.length_y;
  grid_map.cloud_in_topic = config.cloud_in_topic;
  grid_map.intensity_factor = config.intensity_factor;
  grid_map.height_factor = config.height_factor;
  grid_map.frame_out = config.frame_out;
  grid_map.mapi_topic_name = config.mapi_topic_name;
  grid_map.maph_topic_name = config.maph_topic_name;

  if (nh_ptr) {
    if (last_cloud_topic != grid_map.cloud_in_topic) {
      sub_pc2 = nh_ptr->subscribe<sensor_msgs::PointCloud2>(grid_map.cloud_in_topic, 1, pointcloudCallback);
      last_cloud_topic = grid_map.cloud_in_topic;
      ROS_INFO("Subscribing to new pointcloud topic: %s", last_cloud_topic.c_str());
    }
    if (last_igrid_topic != grid_map.mapi_topic_name) {
      pub_igrid = nh_ptr->advertise<nav_msgs::OccupancyGrid>(grid_map.mapi_topic_name, 1);
      last_igrid_topic = grid_map.mapi_topic_name;
      ROS_INFO("Advertising new intensity grid topic: %s", last_igrid_topic.c_str());
    }
    if (last_hgrid_topic != grid_map.maph_topic_name) {
      pub_hgrid = nh_ptr->advertise<nav_msgs::OccupancyGrid>(grid_map.maph_topic_name, 1);
      last_hgrid_topic = grid_map.maph_topic_name;
      ROS_INFO("Advertising new height grid topic: %s", last_hgrid_topic.c_str());
    }
  }

  // First compute grid dimensions from parameters, then initialize grids
  grid_map.paramRefresh();
  grid_map.initGrid(height_grid);
  grid_map.paramRefresh();
}


void pointcloudCallback(const sensor_msgs::PointCloud2::ConstPtr &msg)
{
  // Ensure grids have correct size before use
  const size_t expected = static_cast<size_t>(grid_map.getSize());
  if (intensity_grid->data.size() != expected || height_grid->data.size() != expected) {
    try {
      intensity_grid->data.assign(expected, -1);
      height_grid->data.assign(expected, -1);
    } catch (...) {
      ROS_ERROR("Failed to resize grids, skipping frame");
      return;
    }
  }

  // Clear grid data
  std::fill(intensity_grid->data.begin(), intensity_grid->data.end(), -1);
  std::fill(height_grid->data.begin(), height_grid->data.end(), -1);

  // Iterate PointCloud2 directly to avoid PCL runtime issues
  try {
    sensor_msgs::PointCloud2ConstIterator<float> iter_x(*msg, "x");
    sensor_msgs::PointCloud2ConstIterator<float> iter_y(*msg, "y");
    sensor_msgs::PointCloud2ConstIterator<float> iter_z(*msg, "z");
    // Intensity may be absent in some datasets; guard accordingly
    bool has_intensity = false;
    for (const auto &f : msg->fields) {
      if (f.name == "intensity") { has_intensity = true; break; }
    }
    sensor_msgs::PointCloud2ConstIterator<float> iter_i(*msg, has_intensity ? "intensity" : "x");

    for (; iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z, ++iter_i) {
      const float x = *iter_x;
      const float y = *iter_y;
      const float z = *iter_z;
      const float inten = has_intensity ? *iter_i : 0.0f;

      // Enhanced noise filtering
      if (std::fabs(x) <= 0.01f) continue; // filter near-origin
      if (std::isnan(x) || std::isnan(y) || std::isnan(z)) continue; // filter NaN values
      if (std::fabs(z) > 3.0f) continue; // filter extreme height values
      if (std::sqrt(x*x + y*y) > 20.0f) continue; // filter points too far away
      
      if (x <= grid_map.topleft_x || x >= grid_map.bottomright_x) continue;
      if (y <= grid_map.bottomright_y || y >= grid_map.topleft_y) continue;

      PointXY cell = getIndex(x, y);
      if (cell.x < 0 || cell.x >= grid_map.cell_num_x || cell.y < 0 || cell.y >= grid_map.cell_num_y) continue;

      const size_t index = static_cast<size_t>(cell.y) * static_cast<size_t>(grid_map.cell_num_x) + static_cast<size_t>(cell.x);
      if (index >= intensity_grid->data.size()) continue; // extra safety

      // Clamp values to int8 range
      const double intensity_val = static_cast<double>(inten) * grid_map.intensity_factor;
      const double height_val = static_cast<double>(z) * grid_map.height_factor;
      intensity_grid->data[index] = static_cast<signed char>(std::max(-128.0, std::min(127.0, intensity_val)));
      height_grid->data[index] = static_cast<signed char>(std::max(-128.0, std::min(127.0, height_val)));
    }
  } catch (const std::exception &e) {
    ROS_ERROR("Exception while parsing PointCloud2: %s", e.what());
  } catch (...) {
    ROS_ERROR("Unknown error while parsing PointCloud2");
  }

  ros::Time now = ros::Time::now();
  height_grid->header.stamp = now;
  intensity_grid->info.map_load_time = now;
  height_grid->info.map_load_time = now;
  pub_igrid.publish(intensity_grid);
  pub_hgrid.publish(height_grid);
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "pointcloud_to_grid_node");
  ros::NodeHandle nh;
  nh_ptr = &nh;

  dynamic_reconfigure::Server<my_dyn_rec::MyParamsConfig> server;
  dynamic_reconfigure::Server<my_dyn_rec::MyParamsConfig>::CallbackType f;
  f = boost::bind(&paramsCallback, _1, _2);

  // The dynamic reconfigure server will call the callback initially, which will set up
  // the publishers and subscriber with the correct topic names.
  server.setCallback(f);

  ros::spin();

  nh_ptr = nullptr;
  return 0;
}