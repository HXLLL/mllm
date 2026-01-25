// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <string>

#include "mllm/models/qwen3_i/runtime_context.hpp"
#include "mllm/core/Tensor.hpp"

namespace mllm::models::qwen3_i {

// Forward declarations
class H2KV;
class KV2H;

enum class TaskType {
  kIO,
  kCompute,
};

inline const char* toString(TaskType type) {
  switch (type) {
    case TaskType::kIO: return "IO";
    case TaskType::kCompute: return "Compute";
  }
  return "Unknown";
}

enum class TaskStatus {
  kPending,
  kCompleted,
  kFailed
};

inline const char* toString(TaskStatus status) {
  switch (status) {
    case TaskStatus::kPending: return "Pending";
    case TaskStatus::kCompleted: return "Completed";
    case TaskStatus::kFailed: return "Failed";
  }
  return "Unknown";
}

class Task {
 public:
  Task() = default;
  virtual ~Task() = default;

  virtual void execute() = 0;
  [[nodiscard]] virtual bool isReady() const = 0;
  [[nodiscard]] virtual std::string name() const = 0;
  [[nodiscard]] virtual TaskType taskType() const = 0;

  [[nodiscard]] TaskStatus status() const { return status_; }
  [[nodiscard]] bool isCompleted() const { return status_ == TaskStatus::kCompleted; }

 protected:
  void complete() { status_ = TaskStatus::kCompleted; }
  void fail() { status_ = TaskStatus::kFailed; }

  TaskStatus status_{TaskStatus::kPending};
};

class LoadParamTask : public Task {
 public:
  LoadParamTask(int layer, RuntimeContext& ctx) : layer_(layer), ctx_(ctx) {}

  void execute() override;
  [[nodiscard]] std::string name() const override {
    return fmt::format("LoadParam[{}]", layer_);
  }
  [[nodiscard]] TaskType taskType() const override { return TaskType::kIO; }
  [[nodiscard]] bool isReady() const override { return true; }

 private:
  int layer_;
  RuntimeContext& ctx_;
};

struct CellIndex {
  int layer;
  int chunk_id;
  CacheRange range;
};

// Load K/V cache from checkpoint
class LoadKVTask : public Task {
 public:
  LoadKVTask(const CellIndex& idx, RuntimeContext& ctx) : Task(), idx_(idx), ctx_(ctx) {}

  void execute() override;
  [[nodiscard]] std::string name() const override {
    return fmt::format("LoadKV[{}][{}]", idx_.layer, idx_.chunk_id);
  }
  [[nodiscard]] TaskType taskType() const override { return TaskType::kIO; }
  [[nodiscard]] bool isReady() const override { return true; }

 private:
  CellIndex idx_;
  RuntimeContext& ctx_;
};

// Load hidden state from checkpoint
class LoadHTask : public Task {
 public:
  LoadHTask(const CellIndex& idx, RuntimeContext& ctx) : Task(), idx_(idx), ctx_(ctx) {}

  void execute() override;
  [[nodiscard]] std::string name() const override {
    return fmt::format("LoadH[{}][{}]", idx_.layer, idx_.chunk_id);
  }
  [[nodiscard]] TaskType taskType() const override { return TaskType::kIO; }
  [[nodiscard]] bool isReady() const override { return true; }

 private:
  CellIndex idx_;
  RuntimeContext& ctx_;
};

// Compute: H2KV + KV2H

class ComputeTask : public Task {
 public:
  ComputeTask(const CellIndex& idx, H2KV& h2kv, KV2H& kv2h, const Tensor& sin_emb, const Tensor& cos_emb, RuntimeContext& ctx)
      : Task(), idx_(idx), h2kv_(h2kv), kv2h_(kv2h), sin_emb_(sin_emb), cos_emb_(cos_emb), ctx_(ctx) {}
  void execute() override;
  [[nodiscard]] std::string name() const override {
    return fmt::format("Compute[{}][{}]", idx_.layer, idx_.chunk_id);
  }
  [[nodiscard]] TaskType taskType() const override { return TaskType::kCompute; }
  [[nodiscard]] bool isReady() const override {
    return ctx_.loader().isLayerLoaded(idx_.layer) && ctx_.state().isHLoaded(idx_.range)
           && ctx_.state().isKVLoaded({idx_.layer, 0, idx_.range.offset});
  }

 private:
  CellIndex idx_;
  H2KV& h2kv_;
  KV2H& kv2h_;
  Tensor sin_emb_;
  Tensor cos_emb_;
  RuntimeContext& ctx_;
};

}  // namespace mllm::models::qwen3_i
