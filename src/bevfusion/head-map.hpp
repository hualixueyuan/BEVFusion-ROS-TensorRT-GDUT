/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: MIT
 */

#ifndef __HEAD_MAP_HPP__
#define __HEAD_MAP_HPP__

#include <memory>
#include <string>

#include "common/dtype.hpp"

namespace bevfusion {
namespace head {
namespace map {

struct MapHeadParameter {
  std::string model;
  bool enabled = false;
};

struct MapView {
  const nvtype::half* data = nullptr;
  int classes = 0;
  int height = 0;
  int width = 0;

  bool valid() const { return data != nullptr && classes > 0 && height > 0 && width > 0; }
  size_t numel() const { return static_cast<size_t>(classes) * height * width; }
};

class MapHead {
 public:
  virtual MapView forward(const nvtype::half* fusion_feature, void* stream) = 0;
  virtual void print() = 0;
};

std::shared_ptr<MapHead> create_maphead(const MapHeadParameter& param);

};  // namespace map
};  // namespace head
};  // namespace bevfusion

#endif  // __HEAD_MAP_HPP__
