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
  kIO,
  kCompute,
};

enum class TaskStatus {
  kPending,
  kCompleted,
  kFailed
};

class Task {
 public:
  Task() = default;
  virtual ~Task() = default;

  virtual void execute() = 0;
  [[nodiscard]] virtual bool isReady() const = 0;
  [[nodiscard]] virtual std::string name() const = 0;
  [[nodiscard]] virtual TaskType taskType() const = 0;

  [[nodiscard]] TaskStatus status() const { return status_; }

 protected:
  void complete() { status_ = TaskStatus::kCompleted; MLLM_INFO("{} completed", name()); }
  void fail() { status_ = TaskStatus::kFailed; MLLM_INFO("{} failed", name()); }

  TaskStatus status_{TaskStatus::kPending};
};

class LoadParamTask : public Task {
 public:
  LoadParamTask(int layer, const std::vector<std::string>& param_names, ParameterLoader& loader,
                GenerationState& state)
      : layer_(layer), param_names_(param_names), loader_(loader), state_(state) {}

  void execute() override;
  [[nodiscard]] std::string name() const override {
    return fmt::format("LoadParam[{}]", layer_);
  }
  [[nodiscard]] TaskType taskType() const override { return TaskType::kIO; }
  [[nodiscard]] bool isReady() const override { return true; }

 private:
  int layer_;
  std::vector<std::string> param_names_;

  ParameterLoader& loader_;
  GenerationState& state_;
};

struct CellIndex {
  int layer;
  int chunk_id;
  CacheRange range;
};

// Load K/V cache from checkpoint
class LoadKVTask : public Task {
 public:
  LoadKVTask(const CellIndex& idx, GenerationState& state) : Task(), idx_(idx), state_(state) {}

  void execute() override;
  [[nodiscard]] std::string name() const override {
    return fmt::format("LoadKV[{}][{}]", idx_.layer, idx_.chunk_id);
  }
  [[nodiscard]] TaskType taskType() const override { return TaskType::kIO; }
  [[nodiscard]] bool isReady() const override { return true; }

 private:
  CellIndex idx_;
  GenerationState& state_;
};

// Load hidden state from checkpoint
class LoadHTask : public Task {
 public:
  LoadHTask(const CellIndex& idx, GenerationState& state) : Task(), idx_(idx), state_(state) {}

  void execute() override;
  [[nodiscard]] std::string name() const override {
    return fmt::format("LoadH[{}][{}]", idx_.layer, idx_.chunk_id);
  }
  [[nodiscard]] TaskType taskType() const override { return TaskType::kIO; }
  [[nodiscard]] bool isReady() const override { return true; }

 private:
  CellIndex idx_;
  GenerationState& state_;
};

// Compute: H2KV + KV2H

class ComputeTask : public Task {
 public:
  ComputeTask(const CellIndex& idx, H2KV& h2kv, KV2H& kv2h, const Tensor &sin_emb, const Tensor &cos_emb, GenerationState& state)
      : Task(), idx_(idx), h2kv_(h2kv), kv2h_(kv2h), sin_emb_(sin_emb), cos_emb_(cos_emb), state_(state) {}
  void execute() override;
  [[nodiscard]] std::string name() const override {
    return fmt::format("Compute[{}][{}]", idx_.layer, idx_.chunk_id);
  }
  [[nodiscard]] TaskType taskType() const override { return TaskType::kCompute; }
  [[nodiscard]] bool isReady() const override {
    return state_.isParamLoaded(idx_.layer) && state_.isHLoaded(idx_.range)
           && state_.isKVLoaded({idx_.layer, 0, idx_.range.offset});
  }

 private:
  CellIndex idx_;
  H2KV& h2kv_;
  KV2H& kv2h_;
  Tensor sin_emb_;
  Tensor cos_emb_;
  GenerationState& state_;
};

}  // namespace mllm::models::qwen3_i
