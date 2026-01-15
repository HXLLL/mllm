// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <optional>
#include <vector>

#include "mllm/models/qwen3_i/grid_task.hpp"

namespace mllm::models::qwen3_i {

class GridScheduler {
 public:
  GridScheduler(int num_layers, int num_chunks, GenerationState& state, ParameterLoader& parameter_loader);
  virtual ~GridScheduler() = default;

  void initLayerParamTask(int layer, LoadParamTask load_param);
  void initGridCell(int layer, int chunk, GridCell cell);

  void run();

 protected:
  virtual GridTask* selectNext() = 0;
  [[nodiscard]] virtual bool isDone() const = 0;

  [[nodiscard]] bool isInitDone() const;
  [[nodiscard]] bool isComputeReady(int layer, int chunk) const;

  [[nodiscard]] GridCell& getCell(int layer, int chunk) { return *grid_[layer * num_chunks_ + chunk]; }
  [[nodiscard]] const GridCell& getCell(int layer, int chunk) const { return *grid_[layer * num_chunks_ + chunk]; }
  [[nodiscard]] LoadParamTask& getLayerParamTask(int layer) { return *layer_param_tasks_[layer]; }
  [[nodiscard]] const LoadParamTask& getLayerParamTask(int layer) const { return *layer_param_tasks_[layer]; }

  GenerationState& state_;
  ParameterLoader& parameter_loader_;
  const int num_layers_;
  const int num_chunks_;

 private:
  std::vector<std::optional<LoadParamTask>> layer_param_tasks_;
  std::vector<std::optional<GridCell>> grid_;
};

}  // namespace mllm::models::qwen3_i
