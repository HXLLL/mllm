// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/models/qwen3_i/simple_grid_scheduler.hpp"

namespace mllm::models::qwen3_i {

GridTask* SimpleGridScheduler::selectNext() {
  // Check layer param tasks first
  for (int layer = 0; layer < num_layers_; ++layer) {
    auto& task = getLayerParamTask(layer);
    if (!task.isCompleted()) {
      return &task;
    }
  }

  // Check all grid cells for LoadKV, LoadH, and Compute tasks
  for (int layer = 0; layer < num_layers_; ++layer) {
    for (int chunk = 0; chunk < num_chunks_; ++chunk) {
      auto& cell = getCell(layer, chunk);

      // Check Compute (must also be ready)
      if (isComputeReady(layer, chunk)) {
        return &cell.compute;
      }
    }
  }

  done_ = true;

  return nullptr;
}

bool SimpleGridScheduler::isDone() const {
  return done_;
}

}  // namespace mllm::models::qwen3_i
