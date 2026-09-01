#include "bevfusion_ros.hpp"

int main(int argc, char** argv) 
{  
  ros::init(argc, argv, "bevfusion_node");
  ros::NodeHandle n;
  ros::NodeHandle private_n("~");
  std::string model_name;
  std::string  precision; 
    
  if (!private_n.getParam("model_name", model_name))
    n.param<std::string>("model_name", model_name, "det");
  if (!private_n.getParam("precision", precision))
    n.param<std::string>("precision", precision, "fp16");
  
  std::cout << "\033[1;32m--model_name: " << model_name << "\033[0m" << std::endl;
  std::cout << "\033[1;32m--precision : " << precision << "\033[0m" << std::endl;

  auto bevfusion_node = std::make_shared<RosNode>(model_name, precision);
  ros::spin();
  return 0;
}
