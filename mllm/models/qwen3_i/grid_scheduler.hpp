// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <vector>

#include "mllm/models/qwen3_i/grid_task.hpp"

namespace mllm::models::qwen3_i {

struct GridCell {
  std::unique_ptr<GridTask> load_kv;
  std::unique_ptr<GridTask> load_h;
  std::unique_ptr<GridTask> compute;
  CacheRange range;
};

class GridScheduler {
 public:
  GridScheduler(int num_layers, int num_chunks, GenerationState& state, ParameterLoader& parameter_loader);

  void setLayerParamTask(int layer, std::unique_ptr<GridTask> load_param);
  void setCellTasks(int layer, int chunk, std::unique_ptr<GridTask> load_kv, std::unique_ptr<GridTask> load_h,
                    std::unique_ptr<GridTask> compute);
  void run();

  [[nodiscard]] int numLayers() const { return num_layers_; }
  [[nodiscard]] int numChunks() const { return num_chunks_; }
  [[nodiscard]] GridCell& getCell(int layer, int chunk) { return grid_[layer * num_chunks_ + chunk]; }
  [[nodiscard]] const GridCell& getCell(int layer, int chunk) const { return grid_[layer * num_chunks_ + chunk]; }

 protected:
  virtual GridTask* selectNext();
  [[nodiscard]] bool isComputeReady(int layer, int chunk) const;
  GenerationState& state_;
  ParameterLoader& parameter_loader_;

 private:
  int num_layers_;
  int num_chunks_;
  std::vector<GridCell> grid_;
  std::vector<std::unique_ptr<GridTask>> layer_param_tasks_;
};

}  // namespace mllm::models::qwen3_i
