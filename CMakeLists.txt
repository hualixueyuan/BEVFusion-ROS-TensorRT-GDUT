cmake_minimum_required(VERSION 3.16)
project(bevfusion)

add_definitions(-w)
# set(CMAKE_BUILD_TYPE "Release")   # pcl会报错 使用debug模式
# set(CMAKE_BUILD_TYPE "Debug")

add_compile_options(-W)
add_compile_options(-std=c++14)

set(arch ${CMAKE_HOST_SYSTEM_PROCESSOR})

set(CMAKE_CXX_FLAGS "-std=c++14 -O3 -g -DENABLE_TEXT_BACKEND_STB")

# set(CMAKE_CXX_FLAGS_RELEASE "-std=c++14 -Wextra -Wall -Wno-missing-field-initializers -Wno-deprecated-declarations -O3 -DENABLE_TEXT_BACKEND_STB")
# set(CMAKE_CXX_FLAGS_DEBUG   "-std=c++14 -O0 -g -DENABLE_TEXT_BACKEND_STB")

set(CUDA_NVCC_FLAGS_RELEASE "-Xcompiler -std=c++14,-Wextra,-Wall,-Wno-deprecated-declarations,-O3 -DENABLE_TEXT_BACKEND_STB")
set(CUDA_NVCC_FLAGS_DEBUG   "-std=c++14 -O0 -g -DENABLE_TEXT_BACKEND_STB")

if(DEFINED ENV{CUDASM} AND NOT "$ENV{CUDASM}" STREQUAL "")
  set(BEVFUSION_CUDA_ARCH "$ENV{CUDASM}")
else()
  set(BEVFUSION_CUDA_ARCH "86")
endif()
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
  -gencode arch=compute_${BEVFUSION_CUDA_ARCH},code=compute_${BEVFUSION_CUDA_ARCH}
)

# ----------------x86-----------------
# cuda
if(DEFINED ENV{CUDA_HOME} AND EXISTS "$ENV{CUDA_HOME}/bin/nvcc")
  set(CUDA_TOOLKIT_ROOT_DIR "$ENV{CUDA_HOME}")
elseif(DEFINED ENV{CONDA_PREFIX} AND EXISTS "$ENV{CONDA_PREFIX}/bin/nvcc")
  set(CUDA_TOOLKIT_ROOT_DIR "$ENV{CONDA_PREFIX}")
else()
  set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)
endif()
set(CUDA_INSTALL_TARGET_DIR targets/x86_64-linux)
set(CUDA_INCLUDE_DIRS ${CUDA_TOOLKIT_ROOT_DIR}/${CUDA_INSTALL_TARGET_DIR}/include)
set(CUDA_LIBS ${CUDA_TOOLKIT_ROOT_DIR}/${CUDA_INSTALL_TARGET_DIR}/lib)

# TENSORRT
if(DEFINED ENV{TensorRT_ROOT} AND EXISTS "$ENV{TensorRT_ROOT}/include/NvInfer.h")
  set(TensorRT_ROOT "$ENV{TensorRT_ROOT}")
elseif(DEFINED ENV{CONDA_PREFIX} AND EXISTS "$ENV{CONDA_PREFIX}/include/NvInfer.h")
  set(TensorRT_ROOT "$ENV{CONDA_PREFIX}")
else()
  set(TensorRT_ROOT /usr)
endif()
set(TensorRT_INCLUDE_DIRS ${TensorRT_ROOT}/include)
file(GLOB TensorRT_PYTHON_LIB_DIRS "${TensorRT_ROOT}/lib/python*/site-packages/tensorrt_libs")
if(TensorRT_PYTHON_LIB_DIRS)
  list(GET TensorRT_PYTHON_LIB_DIRS 0 TensorRT_LIBRARY_DIR)
else()
  set(TensorRT_LIBRARY_DIR "${TensorRT_ROOT}/lib")
endif()
if(EXISTS "${TensorRT_LIBRARY_DIR}/libnvinfer.so.10")
  set(NVINFER_LIBRARY "${TensorRT_LIBRARY_DIR}/libnvinfer.so.10")
  set(NVINFER_PLUGIN_LIBRARY "${TensorRT_LIBRARY_DIR}/libnvinfer_plugin.so.10")
else()
  find_library(NVINFER_LIBRARY NAMES nvinfer REQUIRED)
  find_library(NVINFER_PLUGIN_LIBRARY NAMES nvinfer_plugin REQUIRED)
endif()

# RTX 50-series GPUs require the CUDA 12.8 sparse-convolution build.  Keep the
# location configurable instead of replacing the legacy CUDA 11 binary that is
# bundled in this repository.
if(DEFINED ENV{SPCONV_ROOT} AND EXISTS "$ENV{SPCONV_ROOT}/libspconv/include/spconv/engine.hpp")
  set(SPCONV_ROOT "$ENV{SPCONV_ROOT}")
else()
  message(FATAL_ERROR "Set SPCONV_ROOT to a 3DSparseConvolution tree containing libspconv/include and libspconv/parser")
endif()
set(SPCONV_INCLUDE_DIR "${SPCONV_ROOT}/libspconv/include")
set(SPCONV_PARSER_DIR "${SPCONV_ROOT}/libspconv/parser")
set(SPCONV_LIBRARY "${SPCONV_ROOT}/libspconv/lib/x86_64_cuda12.8/libspconv.so")
if(NOT EXISTS "${SPCONV_LIBRARY}")
  message(FATAL_ERROR "CUDA 12.8 libspconv was not found: ${SPCONV_LIBRARY}")
endif()
# ------------------------------------

# ----------------orin-----------------
# # cuda
# set(CUDA_INCLUDE_DIRS /usr/local/cuda-11.4/include)
# set(CUDA_LIBS /usr/local/cuda-11.4/targets/aarch64-linux/lib)

# # tensorrt
# set(TensorRT_INCLUDE_DIRS /usr/include)
# set(TensorRT_LIBS /usr/lib/aarch64-linux-gnu)
# # ------------------------------------


find_package(CUDA REQUIRED)
find_package(Protobuf REQUIRED)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
# PCL's VTK auto-init definitions contain parentheses.  Legacy FindCUDA
# forwards them to nvcc without shell escaping, and CUDA 12 rejects them.
# This package does not instantiate VTK rendering modules, so they are not
# needed by either CUDA target.
remove_definitions(
  "-DvtkRenderingContext2D_AUTOINIT=1(vtkRenderingContextOpenGL2)"
  "-DvtkRenderingCore_AUTOINIT=3(vtkInteractionStyle,vtkRenderingFreeType,vtkRenderingOpenGL2)"
)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  roslib
  dynamic_reconfigure
  message_filters

  sensor_msgs
  pcl_conversions
  pcl_ros 
  cv_bridge
)

generate_dynamic_reconfigure_options(
  cfg/GdutSegThresholds.cfg
)

include_directories(
  BEFORE
  ${SPCONV_INCLUDE_DIR}
  ${SPCONV_PARSER_DIR}
  third_party/stb
  third_party/pybind11/include
  third_party/cuOSD/src
  src/common
  src/
  ${TensorRT_INCLUDE_DIRS}
  ${Protobuf_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
  include_ros
)

catkin_package()

link_directories(
  build
  ${CUDA_LIBS}
)

if(arch STREQUAL "aarch64")
  link_directories(/usr/lib/aarch64-linux-gnu /usr/aarch64-linux-gnu/lib)
endif()

set(CMAKE_BUILD_RPATH "${CUDA_LIBS};${SPCONV_ROOT}/libspconv/lib/x86_64_cuda12.8")
set(CMAKE_INSTALL_RPATH "${CUDA_LIBS};${SPCONV_ROOT}/libspconv/lib/x86_64_cuda12.8")

file(GLOB_RECURSE CORE_FILES CONFIGURE_DEPENDS
  src/bevfusion/*.cu 
  src/bevfusion/*.cpp
  src/common/tensor.cu
  src/common/tensorrt.cpp
)

set(SPCONV_PARSER_FILES
  ${SPCONV_PARSER_DIR}/onnx-parser.cpp
  third_party/3DSparseConvolution/src/onnx/onnx-ml.pb.cpp
  third_party/3DSparseConvolution/src/onnx/onnx-operators-ml.pb.cpp
)

# Compile the generated ONNX protobuf sources without Conda's protobuf 4.x
# headers.  ROS Noetic and these generated files use the system protobuf 3.6
# ABI, while the rest of the target still needs Conda's TensorRT headers.
add_library(spconv_parser STATIC ${SPCONV_PARSER_FILES})
set_target_properties(spconv_parser PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  INCLUDE_DIRECTORIES "${SPCONV_INCLUDE_DIR};${SPCONV_PARSER_DIR};${CMAKE_CURRENT_SOURCE_DIR}/third_party/3DSparseConvolution/src;${Protobuf_INCLUDE_DIRS}"
)
target_link_libraries(spconv_parser ${Protobuf_LIBRARIES} ${SPCONV_LIBRARY})

file(GLOB_RECURSE VISUALIZE_FILES 
  third_party/cuOSD/src/*.cpp
  third_party/cuOSD/src/*.cu
  src/common/visualize.cu
)

####################### libbevfusion_core.so ##########################
cuda_add_library(bevfusion_core SHARED 
  ${CORE_FILES}
)

target_link_libraries(bevfusion_core
  libcudart.so
  libcublasLt.so
  ${NVINFER_LIBRARY}
  ${SPCONV_LIBRARY}
  ${Protobuf_LIBRARIES}
  ${NVINFER_PLUGIN_LIBRARY}
  spconv_parser
)

####################### bevfusion program #############################
# cuda_add_executable(demo
#   src_ros/demo.cpp src_ros/bevfusion_plugin.cpp
#   ${VISUALIZE_FILES}
# )

# target_link_libraries(demo
#   bevfusion_core
#   libnvinfer.so
#   libspconv.so
#   ${Protobuf_LIBRARIES}
#   libnvinfer_plugin.so
  
#   ${OpenCV_LIBS}
#   ${PCL_LIBRARIES}
#   ${catkin_LIBRARIES}
# )

# bevfusion_node
cuda_add_executable(bevfusion_node
  src_ros/bevfusion_ros_node.cpp 
  src_ros/bevfusion_ros.cpp 
  src_ros/bevfusion_plugin.cpp
  ${VISUALIZE_FILES}
)

add_dependencies(bevfusion_node ${PROJECT_NAME}_gencfg)

target_link_libraries(bevfusion_node
  bevfusion_core
  ${NVINFER_LIBRARY}
  ${SPCONV_LIBRARY}
  ${Protobuf_LIBRARIES}
  ${NVINFER_PLUGIN_LIBRARY}
  
  ${OpenCV_LIBS}
  ${PCL_LIBRARIES}
  ${catkin_LIBRARIES}
)
