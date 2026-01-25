// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/models/qwen3_i/grid_scheduler.hpp"
#include "mllm/models/qwen3_i/grid_task.hpp"
#include "mllm/utils/Log.hpp"

namespace mllm::models::qwen3_i {

GridScheduler::GridScheduler(std::vector<LayerContext> layers, std::vector<ChunkContext> chunks, RuntimeContext& ctx)
    : num_layers_(layers.size()),
      num_chunks_(chunks.size()),
      layers_(std::move(layers)),
      chunks_(std::move(chunks)),
      grid_(num_layers_),
      ctx_(ctx) {
    for (int i = 0; i < num_layers_; ++i) {
      for (int j = 0; j < num_chunks_; ++j) {
        auto& chunk = chunks_[j];
        int count = chunk.end_pos - chunk.start_pos;
        grid_[i].emplace_back(CellIndex{i, j, CacheRange{i, chunk.start_pos, count}});
      }
    }
  }

void GridScheduler::run() {
  while (!isDone()) {
    Task* task = selectNext();
    if (!task) { 
      MLLM_INFO("No task to execute");
      continue; 
    }
    if (!task->isReady()) {
      MLLM_ERROR_EXIT(ExitCode::kCoreError, "Task is not ready: {}", task->name());
      continue;
    }
    task->execute();
    MLLM_INFO("Task completed: {}, status: {}", task->name(), task->status());
  }
}

void GridScheduler::initTasks() {
  layer_param_tasks_.resize(num_layers_);
  for (int layer = 0; layer < num_layers_; ++layer) {
    addLayerParamTask(layer);
  }

  for (int layer = 0; layer < num_layers_; ++layer) {
    for (int chunk = 0; chunk < num_chunks_; ++chunk) {
      const auto& r = grid_[layer][chunk].idx.range;

      addComputeTask(layer, chunk);
    }
  }
}

void GridScheduler::addLayerParamTask(int layer) {
  layer_param_tasks_[layer] = std::make_unique<LoadParamTask>(layer, ctx_);
}

void GridScheduler::addLoadKVTask(int layer, int chunk_id) {
  auto& cell = grid_[layer][chunk_id];
  cell.load_kv = std::make_unique<LoadKVTask>(cell.idx, ctx_);
}

void GridScheduler::addLoadHTask(int layer, int chunk_id) {
  auto& cell = grid_[layer][chunk_id];
  cell.load_h = std::make_unique<LoadHTask>(cell.idx, ctx_);
}

void GridScheduler::addComputeTask(int layer, int chunk_id) {
  auto& cell = grid_[layer][chunk_id];
  auto& h2kv = layers_[layer].h2kv;
  auto& kv2h = layers_[layer].kv2h;
  auto& sin_emb = chunks_[chunk_id].sin_emb;
  auto& cos_emb = chunks_[chunk_id].cos_emb;
  cell.compute = std::make_unique<ComputeTask>(cell.idx, *h2kv, *kv2h, sin_emb, cos_emb, ctx_);
}

}  // namespace mllm::models::qwen3_i
