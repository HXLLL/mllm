// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <vector>

#include "mllm/models/qwen3_i/generation_state.hpp"
#include "mllm/models/qwen3_i/parameter_loader.hpp"
#include "mllm/core/Tensor.hpp"

namespace mllm::models::qwen3_i {

// Forward declarations
class H2KV;
class KV2H;

enum class TaskType {
  kLoadParam,
  kLoadKV,
  kLoadH,
  kCompute
};

class GridTask {
 public:
  GridTask(int layer, int chunk_id, const CacheRange& range, GenerationState& state) : state_(state), layer_(layer), chunk_id_(chunk_id), range_(range) {}

  virtual ~GridTask() = default;
  virtual void execute() = 0;
  [[nodiscard]] virtual std::string name() const = 0;
  [[nodiscard]] virtual TaskType taskType() const = 0;

  [[nodiscard]] int layer() const { return layer_; }
  [[nodiscard]] int chunkId() const { return chunk_id_; }
  [[nodiscard]] const CacheRange& range() const { return range_; }
  [[nodiscard]] bool isCompleted() const { return completed_; }
  [[nodiscard]] bool isIO() const { return taskType() == TaskType::kLoadParam || taskType() == TaskType::kLoadKV || taskType() == TaskType::kLoadH; }
  [[nodiscard]] bool isCompute() const { return taskType() == TaskType::kCompute; }

 protected:
  void markCompleted() { completed_ = true; }
  GenerationState& state_;

 private:
  int layer_;
  int chunk_id_;
  CacheRange range_;
  bool completed_ = false;
};

class LoadParamTask : public GridTask {
 public:
  LoadParamTask(int layer, ParameterLoader& loader, const std::vector<std::string>& param_names, GenerationState& state);

  void execute() override;
  [[nodiscard]] std::string name() const override;
  [[nodiscard]] TaskType taskType() const override { return TaskType::kLoadParam; }

 private:
  ParameterLoader& loader_;
  std::vector<std::string> param_names_;
};

// Load K/V cache from checkpoint
class LoadKVTask : public GridTask {
 public:
  LoadKVTask(int layer, int chunk_id, const CacheRange& range, GenerationState& state);

  void execute() override;
  [[nodiscard]] std::string name() const override;
  [[nodiscard]] TaskType taskType() const override { return TaskType::kLoadKV; }
};

// Load hidden state from checkpoint
class LoadHTask : public GridTask {
 public:
  LoadHTask(int layer, int chunk_id, const CacheRange& range, GenerationState& state);

  void execute() override;
  [[nodiscard]] std::string name() const override;
  [[nodiscard]] TaskType taskType() const override { return TaskType::kLoadH; }
};

// Compute: H2KV + KV2H
class ComputeTask : public GridTask {
 public:
  ComputeTask(int layer, int chunk_id, const CacheRange& range, H2KV& h2kv, KV2H& kv2h, GenerationState& state,
              const Tensor& sin_emb, const Tensor& cos_emb);

  void execute() override;
  [[nodiscard]] std::string name() const override;
  [[nodiscard]] TaskType taskType() const override { return TaskType::kCompute; }

 private:
  H2KV& h2kv_;
  KV2H& kv2h_;
  Tensor sin_emb_;
  Tensor cos_emb_;
};

struct GridCell {
  LoadKVTask load_kv;
  LoadHTask load_h;
  ComputeTask compute;
  CacheRange range;

  GridCell(const CacheRange& range, int chunk, GenerationState& state, H2KV& h2kv, KV2H& kv2h, const Tensor& sin_emb,
           const Tensor& cos_emb);
};

}  // namespace mllm::models::qwen3_i
