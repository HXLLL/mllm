// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <vector>

#include "mllm/models/qwen3_i/grid_task.hpp"

namespace mllm::models::qwen3_i {

struct GridCell {
  explicit GridCell(const CellContext& ctx) : ctx(ctx) {}

  std::optional<LoadKVTask> load_kv;
  std::optional<LoadHTask> load_h;
  std::optional<ComputeTask> compute;
  CellContext ctx;
};

class GridScheduler {
 public:
  GridScheduler(int num_layers, int seq_len, int chunk_size, GenerationState& state,
                ParameterLoader& parameter_loader);
  virtual ~GridScheduler() = default;

  /* layer tasks*/
  void addLayerParamTask(int layer, const std::vector<std::string>& param_names);

  /* cell tasks */
  void initCellContext(int layer, int chunk_id, CacheRange range);
  void addLoadKVTask(int layer, int chunk_id);
  void addLoadHTask(int layer, int chunk_id);
  void addComputeTask(int layer, int chunk_id);

  void run();

 protected:
  virtual void initTasks();
  virtual Task* selectNext() = 0;
  [[nodiscard]] virtual bool isDone() const = 0;

  [[nodiscard]] bool isInitDone() const;
  [[nodiscard]] bool isComputeReady(int layer, int chunk) const;

  [[nodiscard]] GridCell& getCell(int layer, int chunk) { return grid_[layer][chunk]; }
  [[nodiscard]] const GridCell& getCell(int layer, int chunk) const { return grid_[layer][chunk]; }

  GenerationState& state_;
  ParameterLoader& parameter_loader_;
  const int num_layers_;
  const int seq_len_;
  const int chunk_size_;
  const int num_chunks_;

 private:
  CellContext makeCellContext(int layer, int chunk_id, CacheRange range);

  std::vector<std::optional<LoadParamTask>> layer_param_tasks_;
  std::vector<std::vector<GridCell>> grid_;
};

}  // namespace mllm::models::qwen3_i
