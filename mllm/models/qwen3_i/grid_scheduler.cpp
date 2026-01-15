// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/models/qwen3_i/grid_scheduler.hpp"
#include "mllm/models/qwen3_i/generation_state.hpp"

namespace mllm::models::qwen3_i {

GridScheduler::GridScheduler(int num_layers, int num_chunks, GenerationState& state, ParameterLoader& parameter_loader)
    : state_(state),
      parameter_loader_(parameter_loader),
      num_layers_(num_layers),
      num_chunks_(num_chunks),
      grid_(num_layers * num_chunks),
      layer_param_tasks_(num_layers) {}

void GridScheduler::setLayerParamTask(int layer, std::unique_ptr<GridTask> load_param) {
  layer_param_tasks_[layer] = std::move(load_param);
}

void GridScheduler::setCellTasks(int layer, int chunk, std::unique_ptr<GridTask> load_kv, std::unique_ptr<GridTask> load_h,
                                 std::unique_ptr<GridTask> compute) {
  auto& cell = getCell(layer, chunk);
  cell.load_kv = std::move(load_kv);
  cell.load_h = std::move(load_h);
  cell.compute = std::move(compute);
}

void GridScheduler::run() {
  while (true) {
    GridTask* task = selectNext();
    if (!task) { break; }
    task->execute();
  }
}

bool GridScheduler::isComputeReady(int layer, int chunk) const {
  const auto& range = getCell(layer, chunk).range;
  if (!state_.isParamLoaded(layer)) return false;
  if (!state_.isHLoaded(range)) return false;
  if (range.offset != 0 && !state_.isKVLoaded({layer, 0, range.offset})) return false;
  return true;
}

}  // namespace mllm::models::qwen3_i
