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
      layer_param_tasks_(num_layers),
      grid_(num_layers * num_chunks) {}

void GridScheduler::initLayerParamTask(int layer, LoadParamTask load_param) {
  layer_param_tasks_[layer].emplace(std::move(load_param));
}

void GridScheduler::initGridCell(int layer, int chunk, GridCell cell) {
  grid_[layer * num_chunks_ + chunk].emplace(std::move(cell));
}

bool GridScheduler::isInitDone() const {
  // Check all layer param tasks are initialized
  for (int layer = 0; layer < num_layers_; ++layer) {
    if (!layer_param_tasks_[layer].has_value()) {
      return false;
    }
  }

  // Check all grid cells are initialized
  for (int layer = 0; layer < num_layers_; ++layer) {
    for (int chunk = 0; chunk < num_chunks_; ++chunk) {
      if (!grid_[layer * num_chunks_ + chunk].has_value()) {
        return false;
      }
    }
  }

  return true;
}

void GridScheduler::run() {
  MLLM_RT_ASSERT(isInitDone());

  while (!isDone()) {
    GridTask* task = selectNext();
    if (!task) { continue; }
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
