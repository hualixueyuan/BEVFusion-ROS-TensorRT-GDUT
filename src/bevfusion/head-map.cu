/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: MIT
 */

#include "head-map.hpp"

#include <cuda_fp16.h>

#include <numeric>
#include <vector>

#include "common/check.hpp"
#include "common/tensorrt.hpp"

namespace bevfusion {
namespace head {
namespace map {

class MapHeadImplement : public MapHead {
 public:
  virtual ~MapHeadImplement() {
    if (output_) checkRuntime(cudaFree(output_));
  }

  bool init(const MapHeadParameter& param) {
    if (!param.enabled) return false;
    engine_ = TensorRT::load(param.model);
    if (engine_ == nullptr) return false;
    if (engine_->has_dynamic_dim()) {
      printf("Dynamic shapes are not supported for map head.\n");
      return false;
    }

    auto shape = engine_->static_dims("map_logits");
    Asserts(engine_->dtype("map_logits") == TensorRT::DType::HALF, "Invalid map head output data type.");
    Asserts(shape.size() == 4 && shape[0] == 1, "Map head output must be NCHW with batch size 1.");
    view_.classes = shape[1];
    view_.height = shape[2];
    view_.width = shape[3];

    size_t volume = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
    checkRuntime(cudaMalloc(&output_, volume * sizeof(half)));
    view_.data = reinterpret_cast<nvtype::half*>(output_);
    return true;
  }

  virtual void print() override { engine_->print("Map Head"); }

  virtual MapView forward(const nvtype::half* fusion_feature, void* stream) override {
    std::vector<const void*> bindings(engine_->num_bindings(), nullptr);
    bindings[engine_->index("middle")] = fusion_feature;
    bindings[engine_->index("map_logits")] = output_;
    Asserts(engine_->forward(bindings, stream), "Failed to execute map-head TensorRT engine.");
    return view_;
  }

 private:
  std::shared_ptr<TensorRT::Engine> engine_;
  half* output_ = nullptr;
  MapView view_;
};

std::shared_ptr<MapHead> create_maphead(const MapHeadParameter& param) {
  std::shared_ptr<MapHeadImplement> instance(new MapHeadImplement());
  if (!instance->init(param)) instance.reset();
  return instance;
}

};  // namespace map
};  // namespace head
};  // namespace bevfusion
