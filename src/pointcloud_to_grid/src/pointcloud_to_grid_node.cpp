#include <ros/ros.h>
#include <nav_msgs/OccupancyGrid.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/conversions.h>
#include <pcl_ros/point_cloud.h>
#include <pointcloud_to_grid/pointcloud_to_grid_core.hpp>
#include <pointcloud_to_grid/MyParamsConfig.h>
#include <dynamic_reconfigure/server.h>
#include <std_msgs/String.h>
#include <sstream>

// Global variables
nav_msgs::OccupancyGridPtr intensity_grid(new nav_msgs::OccupancyGrid);
nav_msgs::OccupancyGridPtr height_grid(new nav_msgs::OccupancyGrid);
GridMap grid_map;
ros::Publisher pub_igrid, pub_hgrid;
ros::Publisher pub_debug;
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

  grid_map.initGrid(intensity_grid);
  grid_map.initGrid(height_grid);
  grid_map.paramRefresh();

  // Emit a one-shot config message on /grid_debug so rosbag can capture it
  if (nh_ptr) {
    std_msgs::String dbg;
    std::ostringstream oss;
    oss << "startup grid cfg: frame=" << grid_map.frame_out
        << " res=" << grid_map.getResolution()
        << " size=" << grid_map.getSizeX() << "x" << grid_map.getSizeY()
        << " origin=(" << grid_map.topleft_x << "," << grid_map.bottomright_y << ")"
        << " cloud_in=" << grid_map.cloud_in_topic
        << " mapi=" << grid_map.mapi_topic_name
        << " maph=" << grid_map.maph_topic_name;
    dbg.data = oss.str();
    if (pub_debug) pub_debug.publish(dbg);
  }
}


void pointcloudCallback(const sensor_msgs::PointCloud2::ConstPtr &msg)
{
  pcl::PointCloud<pcl::PointXYZI> out_cloud;
  // Convert ROS PointCloud2 to PCL point cloud
  pcl::fromROSMsg(*msg, out_cloud);
  // Clear grid data
  std::fill(intensity_grid->data.begin(), intensity_grid->data.end(), -1);
  std::fill(height_grid->data.begin(), height_grid->data.end(), -1);

  for (const auto& out_point : out_cloud)
  {
    if (std::fabs(out_point.x) > 0.01) { // Filter out points at origin
      if (out_point.x > grid_map.topleft_x && out_point.x < grid_map.bottomright_x &&
          out_point.y > grid_map.bottomright_y && out_point.y < grid_map.topleft_y)
      {
        PointXY cell = getIndex(out_point.x, out_point.y);
        if (cell.x >= 0 && cell.x < grid_map.cell_num_x && cell.y >= 0 && cell.y < grid_map.cell_num_y)
        {
          size_t index = cell.y * grid_map.cell_num_x + cell.x;

          // Clamp intensity value to signed char range
          double intensity_val = out_point.intensity * grid_map.intensity_factor;
          intensity_grid->data[index] = static_cast<signed char>(std::max(-128.0, std::min(127.0, intensity_val)));

          // Clamp height value to signed char range
          double height_val = out_point.z * grid_map.height_factor;
          height_grid->data[index] = static_cast<signed char>(std::max(-128.0, std::min(127.0, height_val)));
        }
      }
    }
  }

  ros::Time now = ros::Time::now();
  intensity_grid->header.stamp = now;
  height_grid->header.stamp = now;
  intensity_grid->info.map_load_time = now;
  height_grid->info.map_load_time = now;
  pub_igrid.publish(intensity_grid);
  pub_hgrid.publish(height_grid);
  // Publish debug summary on /grid_debug to be recorded by rosbag
  try {
    size_t total = static_cast<size_t>(grid_map.getSize());
    size_t filled_i = 0, filled_h = 0;
    for (size_t i = 0; i < intensity_grid->data.size(); ++i) {
      if (intensity_grid->data[i] != -1) ++filled_i;
      if (height_grid->data[i] != -1) ++filled_h;
    }
    char buf[256];
    std::snprintf(buf, sizeof(buf),
                  "pc2_points=%zu grid=%dx%d res=%.3f fill_i=%zu/%zu fill_h=%zu/%zu",
                  out_cloud.size(), grid_map.getSizeX(), grid_map.getSizeY(),
                  grid_map.getResolution(), filled_i, total, filled_h, total);
    std_msgs::String dbg;
    dbg.data = buf;
    if (pub_debug) pub_debug.publish(dbg);
  } catch (...) {}
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "pointcloud_to_grid_node");
  ros::NodeHandle nh;
  nh_ptr = &nh;
  // Advertise debug topic (String) as latched so the last message is delivered to late subscribers/rosbag
  ros::AdvertiseOptions ao = ros::AdvertiseOptions::create<std_msgs::String>(
      "/grid_debug", 10,
      ros::SubscriberStatusCallback(), ros::SubscriberStatusCallback(),
      ros::VoidConstPtr());
  ao.latch = true;
  pub_debug = nh.advertise(ao);

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