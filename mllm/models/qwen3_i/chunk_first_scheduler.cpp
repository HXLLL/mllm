// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/models/qwen3_i/chunk_first_scheduler.hpp"

namespace mllm::models::qwen3_i {

Task* ChunkFirstScheduler::selectNext() {
  if (phase_ == Phase::LoadParam) {
    if (auto* task = selectNextParamTask()) return task;
    phase_ = Phase::Compute;
  }
  if (phase_ == Phase::Compute) {
    if (auto* task = selectNextComputeTask()) return task;
    phase_ = Phase::Done;
  }
  return nullptr;
}

Task* ChunkFirstScheduler::selectNextParamTask() {
  for (; param_layer_ < numLayers(); ++param_layer_) {
    auto* task = getLayerParamTask(param_layer_);
    if (task && !task->isCompleted()) return task;
  }
  return nullptr;
}

Task* ChunkFirstScheduler::selectNextComputeTask() {
  for (; current_chunk_ < numChunks(); ++current_chunk_, current_layer_ = 0) {
    for (; current_layer_ <= numLayers(); ++current_layer_) {
      auto& cell = getCell(current_layer_, current_chunk_);
      if (cell.load_kv && !cell.load_kv->isCompleted()) { return cell.load_kv.get(); }
      if (cell.load_h && !cell.load_h->isCompleted()) { return cell.load_h.get(); }
      if (cell.compute && !cell.compute->isCompleted() && cell.compute->isReady()) {
        return cell.compute.get();
      }
    }
  }
  return nullptr;
}

bool ChunkFirstScheduler::isDone() const { return phase_ == Phase::Done; }

}  // namespace mllm::models::qwen3_i
