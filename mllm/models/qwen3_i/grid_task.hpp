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
                GenerationState& state);

  void execute() override;
  [[nodiscard]] std::string name() const override;
  [[nodiscard]] TaskType taskType() const override { return TaskType::kIO; }

 private:
  int layer_;
  std::vector<std::string> param_names_;

  ParameterLoader& loader_;
  GenerationState& state_;
};

struct GridContext {
  int layer;
  int chunk_id;
  CacheRange range;
  GenerationState& state;
};

// Load K/V cache from checkpoint
class LoadKVTask : public Task {
 public:
  explicit LoadKVTask(const GridContext& ctx);

  void execute() override;
  [[nodiscard]] std::string name() const override;
  [[nodiscard]] TaskType taskType() const override { return TaskType::kIO; }
 private:
  GridContext ctx_;
};

// Load hidden state from checkpoint
class LoadHTask : public Task {
 public:
  explicit LoadHTask(const GridContext& ctx);

  void execute() override;
  [[nodiscard]] std::string name() const override;
  [[nodiscard]] TaskType taskType() const override { return TaskType::kIO; }
 private:
  GridContext ctx_;
};

// Compute: H2KV + KV2H
class ComputeTask : public Task {
 public:
  ComputeTask(const GridContext& ctx, H2KV& h2kv, KV2H& kv2h, Tensor sin_emb, Tensor cos_emb);
  void execute() override;
  [[nodiscard]] std::string name() const override;
  [[nodiscard]] TaskType taskType() const override { return TaskType::kCompute; }

 private:
  GridContext ctx_;
  H2KV& h2kv_;
  KV2H& kv2h_;
  Tensor sin_emb_;
  Tensor cos_emb_;
};

}  // namespace mllm::models::qwen3_i
