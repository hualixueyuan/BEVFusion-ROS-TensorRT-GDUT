#ifndef bevfusion_ros_h
#define bevfusion_ros_h


#include "bevfusion_plugin.hpp"

#include <bevfusion/GdutSegThresholdsConfig.h>
#include <dynamic_reconfigure/server.h>
#include <ros/ros.h>
#include <array>
// message_filters消息同步器
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h> // 时间接近

// 图像
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>

// 点云
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h> 
#include <pcl_conversions/pcl_conversions.h> 
#include <sensor_msgs/PointCloud2.h>


class RosNode 
{ 
  std::string model_name_, precition_;
  ros::NodeHandle n_;
  ros::NodeHandle private_n_;
  ros::Publisher pub_img_;
  bool seg_mode_ = false;
  bool enable_seg_dynamic_reconfigure_ = false;

  float det_confidence_threshold_ = 0.12f;
  std::array<float, 6> seg_thresholds_ = {{0.5f, 0.4f, 0.4f, 0.45f, 0.4f, 0.55f}};
  std::string output_topic_;
  std::string calib_config_path_;

  std::string topic_cloud_;
  std::string topic_img_f_, topic_img_fl_, topic_img_fr_;
  std::string topic_img_b_, topic_img_bl_, topic_img_br_;


  message_filters::Subscriber<sensor_msgs::PointCloud2> sub_cloud_; 
  message_filters::Subscriber<sensor_msgs::Image> sub_f_img_; 
  message_filters::Subscriber<sensor_msgs::Image> sub_fl_img_; 
  message_filters::Subscriber<sensor_msgs::Image> sub_fr_img_; 
  message_filters::Subscriber<sensor_msgs::Image> sub_b_img_; 
  message_filters::Subscriber<sensor_msgs::Image> sub_bl_img_; 
  message_filters::Subscriber<sensor_msgs::Image> sub_br_img_; 

  typedef message_filters::sync_policies::ApproximateTime<
    sensor_msgs::PointCloud2, 
    sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::Image,
    sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::Image> MySyncPolicy;
  
  typedef message_filters::Synchronizer<MySyncPolicy> Sync;
	std::shared_ptr<Sync> sync_;

  std::shared_ptr<BEVFusionNode> bevfusion_node_;
  using SegThresholdServer = dynamic_reconfigure::Server<bevfusion::GdutSegThresholdsConfig>;
  std::shared_ptr<SegThresholdServer> seg_threshold_server_;

  
 public:
  RosNode(const std::string model_name, const std::string  precision);
  ~RosNode(){};
  void getParameters();
  void segThresholdReconfigureCallback(bevfusion::GdutSegThresholdsConfig& config,
                                       uint32_t level);
  void callback(const sensor_msgs::PointCloud2ConstPtr& msg_cloud, 
    const sensor_msgs::ImageConstPtr& msg_f_img,
    const sensor_msgs::ImageConstPtr& msg_fl_img,
    const sensor_msgs::ImageConstPtr& msg_fr_img,
    const sensor_msgs::ImageConstPtr& msg_b_img,
    const sensor_msgs::ImageConstPtr& msg_bl_img,
    const sensor_msgs::ImageConstPtr& msg_br_img);
};

#endif
