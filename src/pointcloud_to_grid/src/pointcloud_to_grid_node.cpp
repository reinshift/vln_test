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

// TF2 for frame transformations
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2/transform_datatypes.h>
#include <tf2/convert.h>
#include <tf2/LinearMath/Transform.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <geometry_msgs/TransformStamped.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.h>
#include <memory>


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
// TF2
std::unique_ptr<tf2_ros::Buffer> g_tf_buffer;
std::unique_ptr<tf2_ros::TransformListener> g_tf_listener;
// Simple height filter params (static rosparams)
static double g_min_z = 0.05;  // meters
static double g_max_z = 1.5;   // meters

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
      pub_igrid = nh_ptr->advertise<nav_msgs::OccupancyGrid>(grid_map.mapi_topic_name, 1, true /* latch */);
      last_igrid_topic = grid_map.mapi_topic_name;
      ROS_INFO("Advertising new intensity grid topic: %s", last_igrid_topic.c_str());
    }
    if (last_hgrid_topic != grid_map.maph_topic_name) {
      pub_hgrid = nh_ptr->advertise<nav_msgs::OccupancyGrid>(grid_map.maph_topic_name, 1, true /* latch */);
      last_hgrid_topic = grid_map.maph_topic_name;
      ROS_INFO("Advertising new height grid topic: %s", last_hgrid_topic.c_str());
    }
  }

  // First compute grid dimensions from parameters, then initialize grids
  grid_map.paramRefresh();
  grid_map.initGrid(intensity_grid);
  grid_map.initGrid(height_grid);

  // Publish an initial empty grid with valid metadata so RViz subscribers see correct size immediately
  ros::Time now = ros::Time::now();
  intensity_grid->header.stamp = now;
  height_grid->header.stamp = now;
  intensity_grid->info.map_load_time = now;
  height_grid->info.map_load_time = now;
  if (pub_igrid) pub_igrid.publish(intensity_grid);
  if (pub_hgrid) pub_hgrid.publish(height_grid);
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

  // Prepare point cloud in target frame (frame_out, default 'map') using tf2::doTransform
  const std::string target_frame = grid_map.frame_out.empty() ? std::string("map") : grid_map.frame_out;
  const std::string src_frame = msg->header.frame_id;
  sensor_msgs::PointCloud2 cloud_tf;
  const sensor_msgs::PointCloud2* cloud_ptr = msg.get();
  if (!src_frame.empty() && src_frame != target_frame && g_tf_buffer) {
    try {
      geometry_msgs::TransformStamped tf_st;
      if (g_tf_buffer->canTransform(target_frame, src_frame, msg->header.stamp, ros::Duration(0.05))) {
        tf_st = g_tf_buffer->lookupTransform(target_frame, src_frame, msg->header.stamp, ros::Duration(0.05));
      } else if (g_tf_buffer->canTransform(target_frame, src_frame, ros::Time(0), ros::Duration(0.05))) {
        tf_st = g_tf_buffer->lookupTransform(target_frame, src_frame, ros::Time(0), ros::Duration(0.05));
      }
      if (!tf_st.header.frame_id.empty()) {
        tf2::doTransform(*msg, cloud_tf, tf_st);
        cloud_ptr = &cloud_tf;
      } else {
        ROS_WARN_THROTTLE(5.0, "No TF from %s to %s; using original cloud frame %s", src_frame.c_str(), target_frame.c_str(), src_frame.c_str());
      }
    } catch (const std::exception &e) {
      ROS_WARN_THROTTLE(5.0, "TF transform failed: %s; using original cloud", e.what());
    }
  }

  // Clear grid data
  std::fill(intensity_grid->data.begin(), intensity_grid->data.end(), -1);
  std::fill(height_grid->data.begin(), height_grid->data.end(), -1);

  // Iterate PointCloud2 (now in target frame if TF succeeded)
  try {
    sensor_msgs::PointCloud2ConstIterator<float> iter_x(*cloud_ptr, "x");
    sensor_msgs::PointCloud2ConstIterator<float> iter_y(*cloud_ptr, "y");
    sensor_msgs::PointCloud2ConstIterator<float> iter_z(*cloud_ptr, "z");
    // Intensity may be absent in some datasets; guard accordingly
    bool has_intensity = false;
    for (const auto &f : cloud_ptr->fields) {
      if (f.name == "intensity") { has_intensity = true; break; }
    }
    sensor_msgs::PointCloud2ConstIterator<float> iter_i(*cloud_ptr, has_intensity ? "intensity" : "x");

    for (; iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z, ++iter_i) {
      const float x = *iter_x;
      const float y = *iter_y;
      const float z = *iter_z;
      const float inten = has_intensity ? *iter_i : 0.0f;

      // Enhanced noise filtering
      if (std::fabs(x) <= 0.01f) continue; // filter near-origin
      if (std::isnan(x) || std::isnan(y) || std::isnan(z)) continue; // filter NaN values
      // Height filter from rosparams (default 0.05~1.5m) to suppress ground/overhead points
      if (z < static_cast<float>(g_min_z) || z > static_cast<float>(g_max_z)) continue;
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
  intensity_grid->header.stamp = now;
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
  // Initialize TF2 buffer/listener
  g_tf_buffer.reset(new tf2_ros::Buffer(ros::Duration(10.0)));
  g_tf_listener.reset(new tf2_ros::TransformListener(*g_tf_buffer));


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