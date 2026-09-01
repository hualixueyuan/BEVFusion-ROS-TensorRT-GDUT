#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION  // 预处理器宏定义

#include "bevfusion_ros.hpp"

#include <algorithm>

RosNode::RosNode(const std::string model_name, const std::string  precision)
  : model_name_(model_name), precition_(precision), private_n_("~"), seg_mode_(model_name == "seg")
{ 
  getParameters();
  bevfusion_node_.reset(new BEVFusionNode(model_name_, precition_, det_confidence_threshold_,
                                          seg_thresholds_, calib_config_path_));
  if (seg_mode_ && enable_seg_dynamic_reconfigure_) {
    seg_threshold_server_.reset(new SegThresholdServer(private_n_));
    SegThresholdServer::CallbackType callback =
        boost::bind(&RosNode::segThresholdReconfigureCallback, this, _1, _2);
    seg_threshold_server_->setCallback(callback);
  }
  pub_img_ = n_.advertise<sensor_msgs::Image>(output_topic_, 10);
  sub_cloud_.subscribe(n_, topic_cloud_, 10);
  sub_f_img_.subscribe(n_, topic_img_f_, 10);
  sub_b_img_.subscribe(n_, topic_img_b_, 10);

  sub_fl_img_.subscribe(n_,topic_img_fl_, 10);
  sub_fr_img_.subscribe(n_,topic_img_fr_, 10);
  
  sub_bl_img_.subscribe(n_,topic_img_bl_, 10);
  sub_br_img_.subscribe(n_,topic_img_br_, 10);
  
  sync_ = std::make_shared<Sync>( MySyncPolicy(10), sub_cloud_, 
    sub_f_img_, sub_fl_img_, sub_fr_img_,
    sub_b_img_ ,sub_bl_img_, sub_br_img_); 
  
  sync_->registerCallback(boost::bind(&RosNode::callback,this, _1, _2,_3, _4, _5, _6,_7)); // 绑定回调函数
  
  }

void RosNode::getParameters()
{
  private_n_.param<std::string>("topic_cloud", topic_cloud_, "/lidar_top");
  
  private_n_.param<std::string>("topic_img_f", topic_img_f_, "/cam_front/raw");
  private_n_.param<std::string>("topic_img_b", topic_img_b_, "/cam_back/raw");
  
  private_n_.param<std::string>("topic_img_fl", topic_img_fl_, "/cam_front_left/raw");
  private_n_.param<std::string>("topic_img_fr", topic_img_fr_, "/cam_front_right/raw");
  
  private_n_.param<std::string>("topic_img_bl", topic_img_bl_, "/cam_back_left/raw");
  private_n_.param<std::string>("topic_img_br", topic_img_br_, "/cam_back_right/raw");

  output_topic_ = seg_mode_ ? "/bevfusion/seg_image" : "/bevfusion/det_image";
  private_n_.param<std::string>("output_topic", output_topic_, output_topic_);

  calib_config_path_ = pkg_path + "/configs";
  private_n_.param<std::string>("calib_config_path", calib_config_path_, calib_config_path_);
  private_n_.param("enable_seg_dynamic_reconfigure", enable_seg_dynamic_reconfigure_, false);

  double threshold = det_confidence_threshold_;
  private_n_.param("det_confidence_threshold", threshold, threshold);
  det_confidence_threshold_ = static_cast<float>(std::max(0.0, std::min(1.0, threshold)));

  static const std::array<const char*, 6> names = {{
    "seg_drivable_threshold", "seg_ped_crossing_threshold", "seg_walkway_threshold",
    "seg_stop_line_threshold", "seg_carpark_threshold", "seg_divider_threshold"
  }};
  for (size_t i = 0; i < names.size(); ++i) {
    threshold = seg_thresholds_[i];
    private_n_.param(names[i], threshold, threshold);
    seg_thresholds_[i] = static_cast<float>(std::max(0.0, std::min(1.0, threshold)));
  }

  ROS_INFO_STREAM("BEVFusion " << model_name_ << " output: " << output_topic_);
  ROS_INFO_STREAM("Calibration config path: " << calib_config_path_);
  if (seg_mode_) {
    ROS_INFO_STREAM("Seg thresholds: drivable=" << seg_thresholds_[0]
                    << ", ped_crossing=" << seg_thresholds_[1]
                    << ", walkway=" << seg_thresholds_[2]
                    << ", stop_line=" << seg_thresholds_[3]
                    << ", carpark=" << seg_thresholds_[4]
                    << ", divider=" << seg_thresholds_[5]);
  } else {
    ROS_INFO_STREAM("Detection confidence threshold: " << det_confidence_threshold_);
  }
}

void RosNode::segThresholdReconfigureCallback(
    bevfusion::GdutSegThresholdsConfig& config, uint32_t)
{
  seg_thresholds_[0] = static_cast<float>(config.seg_drivable_threshold);
  seg_thresholds_[2] = static_cast<float>(config.seg_walkway_threshold);
  seg_thresholds_[5] = static_cast<float>(config.seg_divider_threshold);
  if (bevfusion_node_) {
    bevfusion_node_->setSegThresholds(
        seg_thresholds_[0], seg_thresholds_[2], seg_thresholds_[5]);
  }
  ROS_INFO_STREAM("Live SEG thresholds updated: drivable=" << seg_thresholds_[0]
                  << ", walkway=" << seg_thresholds_[2]
                  << ", divider=" << seg_thresholds_[5]);
}


void RosNode::callback(const sensor_msgs::PointCloud2ConstPtr& msg_cloud, 
  const sensor_msgs::ImageConstPtr& msg_f_img,
  const sensor_msgs::ImageConstPtr& msg_fl_img,
  const sensor_msgs::ImageConstPtr& msg_fr_img,
  const sensor_msgs::ImageConstPtr& msg_b_img,
  const sensor_msgs::ImageConstPtr& msg_bl_img,
  const sensor_msgs::ImageConstPtr& msg_br_img)
{
  
  cv::Mat f_img, fl_img, fr_img, b_img, bl_img, br_img;
  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_ptr(new pcl::PointCloud<pcl::PointXYZI>());
  
  pcl::fromROSMsg(*msg_cloud, *cloud_ptr);
  f_img  = cv_bridge::toCvShare(msg_f_img , "bgr8")->image;
  fl_img = cv_bridge::toCvShare(msg_fl_img, "bgr8")->image;
  fr_img = cv_bridge::toCvShare(msg_fr_img, "bgr8")->image;
  b_img  = cv_bridge::toCvShare(msg_b_img , "bgr8")->image;
  bl_img = cv_bridge::toCvShare(msg_bl_img, "bgr8")->image;
  br_img = cv_bridge::toCvShare(msg_br_img, "bgr8")->image;
  
  // 这里可能是ros包中前向左右图搞反了, 所以交换fr_img, fl_img的位置
  std::vector<unsigned char *> images = load_images(f_img, fr_img, fl_img, b_img, bl_img, br_img);
  
  // printf("size: %ld \n", cloud_ptr->points.size());
  
  int lidar_num = cloud_ptr->points.size();
  float lidar_arr[lidar_num * 5];
  for(size_t i = 0; i < cloud_ptr->points.size(); ++i )
  {
    long index = i * 5;
    lidar_arr[index]     = cloud_ptr->points[i].x;
    lidar_arr[index + 1] = cloud_ptr->points[i].y;
    lidar_arr[index + 2] = cloud_ptr->points[i].z;
    lidar_arr[index + 3] = cloud_ptr->points[i].intensity;
    // lidar_arr[index + 4] = cloud->points[i].time;
    lidar_arr[index + 4] = 0;
  }
  bevfusion_node_->Inference(images, lidar_arr, cloud_ptr->points.size());

  if (seg_mode_) {
    cv::Mat seg_img;
    if (bevfusion_node_->getLastSegImage(seg_img)) {
      auto msg_seg = cv_bridge::CvImage(msg_cloud->header, "bgr8", seg_img).toImageMsg();
      pub_img_.publish(msg_seg);
    }
    free_images(images);
    return;
  }

  cv::Mat img = cv::imread((pkg_path + "/configs/cuda-bevfusion.jpg").c_str());
  cv::resize(img, img, cv::Size(img.size().width /2, img.size().height /2));
  sensor_msgs::Image::Ptr msg_img_new; 
  msg_img_new = cv_bridge::CvImage(std_msgs::Header(), "bgr8", img).toImageMsg();
  pub_img_.publish(msg_img_new);
  free_images(images);
}
