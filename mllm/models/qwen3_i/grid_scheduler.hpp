// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <vector>

#include "mllm/models/qwen3_i/grid_task.hpp"

namespace mllm::models::qwen3_i {

class GridScheduler {
 public:
  GridScheduler(int num_layers, int num_chunks, GenerationState& state, ParameterLoader& parameter_loader);

  void setLayerParamTask(int layer, std::unique_ptr<GridTask> load_param);
  void initGridCell(int layer, int chunk, GridCell cell) { grid_[layer * num_chunks_ + chunk].emplace(std::move(cell)); }

  void run();

  [[nodiscard]] GridCell& getCell(int layer, int chunk) { return *grid_[layer * num_chunks_ + chunk]; }
  [[nodiscard]] const GridCell& getCell(int layer, int chunk) const { return *grid_[layer * num_chunks_ + chunk]; }

 protected:
  virtual GridTask* selectNext();
  [[nodiscard]] bool isComputeReady(int layer, int chunk) const;
  GenerationState& state_;
  ParameterLoader& parameter_loader_;
  std::vector<std::unique_ptr<GridTask>> layer_param_tasks_;
  const int num_layers_;
  const int num_chunks_;

 private:
  std::vector<std::optional<GridCell>> grid_;
};

}  // namespace mllm::models::qwen3_i
