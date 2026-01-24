// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <vector>

#include "mllm/models/qwen3_i/grid_task.hpp"

namespace mllm::models::qwen3_i {

struct GridCell {
  explicit GridCell(const CellIndex& idx) : idx(idx) {}
  std::unique_ptr<LoadKVTask> load_kv;
  std::unique_ptr<LoadHTask> load_h;
  std::unique_ptr<ComputeTask> compute;
  CellIndex idx;
};

struct LayerContext {
  int layer;
  std::vector<std::string> param_names;
  H2KV* h2kv;
  KV2H* kv2h;
};

struct ChunkContext {
  int chunk_id;
  int start_pos;
  int end_pos;
  Tensor sin_emb;
  Tensor cos_emb;
};

class GridScheduler {
 public:
  GridScheduler(std::vector<LayerContext> layers, std::vector<ChunkContext> chunks,
                GenerationState& state, ParameterLoader& parameter_loader);
  virtual ~GridScheduler() = default;

  void run();
  void initTasks();

 private:
  [[nodiscard]] virtual Task* selectNext() = 0;
  [[nodiscard]] virtual bool isDone() const = 0;

  [[nodiscard]] bool isComputeReady(int layer, int chunk) const;

  /* layer tasks*/
  void addLayerParamTask(int layer);

  /* cell tasks */
  void addLoadKVTask(int layer, int chunk_id);
  void addLoadHTask(int layer, int chunk_id);
  void addComputeTask(int layer, int chunk_id);

  const int num_layers_;
  const int num_chunks_;

  std::vector<LayerContext> layers_;
  std::vector<ChunkContext> chunks_;
  std::vector<std::unique_ptr<LoadParamTask>> layer_param_tasks_;
  std::vector<std::vector<GridCell>> grid_;

  GenerationState& state_;
  ParameterLoader& parameter_loader_;
};

}  // namespace mllm::models::qwen3_i
