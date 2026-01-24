// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/models/qwen3_i/simple_grid_scheduler.hpp"

namespace mllm::models::qwen3_i {

Task* SimpleGridScheduler::selectNext() {
  // Check layer param tasks first
  for (int layer = 0; layer < numLayers(); ++layer) {
    auto* task = getLayerParamTask(layer);
    if (task && !task->isCompleted()) {
      return task;
    }
  }

  // Check all grid cells for LoadKV, LoadH, and Compute tasks
  for (int layer = 0; layer < numLayers(); ++layer) {
    for (int chunk = 0; chunk < numChunks(); ++chunk) {
      auto& cell = getCell(layer, chunk);

      auto* load_kv = cell.load_kv.get();
      if (load_kv && !load_kv->isCompleted()) {
        return load_kv;
      }

      auto* load_h = cell.load_h.get();
      if (load_h && !load_h->isCompleted()) {
        return load_h;
      }

      // Check Compute (must also be ready)
      auto* compute = cell.compute.get();
      if (compute && !compute->isCompleted() && compute->isReady()) {
        return compute;
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
