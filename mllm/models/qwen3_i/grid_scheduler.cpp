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
      grid_(num_layers_ + 1),
      ctx_(ctx) {
    for (int i = 0; i <= num_layers_; ++i) {
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
    MLLM_INFO("Task completed: {}, status: {}", task->name(), toString(task->status()));
  }
}

void GridScheduler::initTasks() {
  // 1. 计算每个 chunk 的 min_watermark，确定边界
  std::vector<int> chunk_min_wm(num_chunks_);
  for (int chunk = 0; chunk < num_chunks_; ++chunk) {
    auto& c = chunks_[chunk];
    chunk_min_wm[chunk] = ctx_.state().getMinWatermark(c.start_pos, c.end_pos - c.start_pos);
  }

  // 2. 为需要计算的 cell 创建任务
  std::vector<bool> layer_needs_compute(num_layers_, false);

  for (int chunk = 0; chunk < num_chunks_; ++chunk) {
    int min_wm = chunk_min_wm[chunk];

    if (min_wm >= 1) {
      auto& c = chunks_[chunk];
      CacheRange h_range{min_wm, c.start_pos, c.end_pos - c.start_pos};
      if (!ctx_.state().isHLoaded(h_range)) {
        addLoadHTask(min_wm, chunk);
      }
    }

    if (min_wm >= num_layers_) continue;

    // 第一个需要计算的 layer（min_wm=-1 时从 0 开始）
    int first_compute = std::max(0, min_wm);

    // 添加 compute tasks
    for (int layer = first_compute; layer < num_layers_; ++layer) {
      addComputeTask(layer, chunk);
      layer_needs_compute[layer] = true;
    }
  }

  // 3. LoadKV：对于每个需要计算的 layer，已完成的 chunk 需要从文件加载 KV
  for (int layer = 0; layer < num_layers_; ++layer) {
    if (!layer_needs_compute[layer]) continue;

    for (int chunk = 0; chunk < num_chunks_; ++chunk) {
      // chunk 在该 layer 已完成（watermark > layer）
      if (chunk_min_wm[chunk] > layer) {
        addLoadKVTask(layer, chunk);
      }
    }
  }

  // 4. 只为需要计算的 layer 创建 LoadParamTask
  layer_param_tasks_.resize(num_layers_);
  for (int layer = 0; layer < num_layers_; ++layer) {
    if (layer_needs_compute[layer]) {
      addLayerParamTask(layer);
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
