// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/models/qwen3_i/grid_scheduler.hpp"
#include "mllm/models/qwen3_i/generation_state.hpp"
#include "mllm/models/qwen3_i/grid_task.hpp"

namespace mllm::models::qwen3_i {

GridScheduler::GridScheduler(int num_layers, int seq_len, int chunk_size, GenerationState& state,
                             ParameterLoader& parameter_loader)
    : state_(state),
      parameter_loader_(parameter_loader),
      num_layers_(num_layers),
      seq_len_(seq_len),
      chunk_size_(chunk_size),
      num_chunks_((seq_len_ + chunk_size_ - 1) / chunk_size_),
      layer_param_tasks_(num_layers),
      grid_(num_layers, std::vector<GridCell>(num_chunks_)) {
  for (int layer = 0; layer < num_layers_; ++layer) {
    for (int chunk = 0; chunk < num_chunks_; ++chunk) {
      int chunk_start = chunk * chunk_size_;
      int chunk_end = std::min(chunk_start + chunk_size_, seq_len_);
      int len = chunk_end - chunk_start;
      CacheRange range{layer, chunk_start, len};
      grid_[layer].emplace_back(makeCellContext(layer, chunk, range));
    }
  }
}

void GridScheduler::addLayerParamTask(int layer, const std::vector<std::string>& param_names) {
  layer_param_tasks_[layer].emplace(layer, param_names, parameter_loader_, state_);
}

void GridScheduler::initCellContext(int layer, int chunk_id, CacheRange range) {
  grid_[layer][chunk] = GridCell(range, chunk, state_, &h2kv, &kv2h, sin_emb, cos_emb);
}

bool GridScheduler::isInitDone() const {
}

void GridScheduler::run() {
  MLLM_RT_ASSERT(isInitDone());

  while (!isDone()) {
    Task* task = selectNext();
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

CellContext GridScheduler::makeCellContext(int layer, int chunk_id, CacheRange range) {
  return CellContext{layer, chunk_id, range};
}

}  // namespace mllm::models::qwen3_i
