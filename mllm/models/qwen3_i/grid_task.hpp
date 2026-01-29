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

enum class TaskStatus { kPending, kSubmitted, kCompleted, kFailed };

inline const char* toString(TaskStatus status) {
  switch (status) {
    case TaskStatus::kPending: return "Pending";
    case TaskStatus::kSubmitted: return "Submitted";
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

  virtual bool submit() { return false; }
  [[nodiscard]] virtual bool isAsync() const { return false; }
  virtual void onIOComplete(StateIOType, const CacheRange&, bool) {}

  [[nodiscard]] TaskStatus status() const { return status_; }
  [[nodiscard]] bool isPending() const { return status_ == TaskStatus::kPending; }
  [[nodiscard]] bool isSubmitted() const { return status_ == TaskStatus::kSubmitted; }
  [[nodiscard]] bool isCompleted() const { return status_ == TaskStatus::kCompleted; }

 protected:
  void markSubmitted() { status_ = TaskStatus::kSubmitted; }
  void complete() { status_ = TaskStatus::kCompleted; }
  void fail() { status_ = TaskStatus::kFailed; }

  TaskStatus status_{TaskStatus::kPending};
};

class LoadParamTask : public Task {
 public:
  LoadParamTask(int layer, RuntimeContext& ctx) : layer_(layer), ctx_(ctx) {}

  void execute() override;
  [[nodiscard]] std::string name() const override { return fmt::format("LoadParam[{}]", layer_); }
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
  bool submit() override;
  [[nodiscard]] bool isAsync() const override { return true; }
  void onIOComplete(StateIOType type, const CacheRange& range, bool success) override;
  [[nodiscard]] std::string name() const override {
    return fmt::format("LoadKV[{}][{}]", idx_.layer, idx_.chunk_id);
  }
  [[nodiscard]] TaskType taskType() const override { return TaskType::kIO; }
  [[nodiscard]] bool isReady() const override { return true; }

 private:
  CellIndex idx_;
  RuntimeContext& ctx_;
  bool k_done_{false};
  bool v_done_{false};
};

// Load hidden state from checkpoint
class LoadHTask : public Task {
 public:
  LoadHTask(const CellIndex& idx, RuntimeContext& ctx) : Task(), idx_(idx), ctx_(ctx) {}

  void execute() override;
  bool submit() override;
  [[nodiscard]] bool isAsync() const override { return true; }
  void onIOComplete(StateIOType type, const CacheRange& range, bool success) override;
  [[nodiscard]] std::string name() const override {
    return fmt::format("LoadH[{}][{}]", idx_.layer, idx_.chunk_id);
  }
  [[nodiscard]] TaskType taskType() const override { return TaskType::kIO; }
  [[nodiscard]] bool isReady() const override { return true; }

 private:
  CellIndex idx_;
  RuntimeContext& ctx_;
  bool done_{false};
};

// Compute: H2KV + KV2H

class ComputeTask : public Task {
 public:
  ComputeTask(const CellIndex& idx, H2KV& h2kv, KV2H& kv2h, const Tensor& sin_emb,
              const Tensor& cos_emb, RuntimeContext& ctx)
      : Task(),
        idx_(idx),
        h2kv_(h2kv),
        kv2h_(kv2h),
        sin_emb_(sin_emb),
        cos_emb_(cos_emb),
        ctx_(ctx) {}
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

class WriteKVTask : public Task {
 public:
  WriteKVTask(const CellIndex& idx, RuntimeContext& ctx) : Task(), idx_(idx), ctx_(ctx) {}

  void execute() override;
  bool submit() override;
  [[nodiscard]] bool isAsync() const override { return true; }
  [[nodiscard]] std::string name() const override {
    return fmt::format("WriteKV[{}][{}]", idx_.layer, idx_.chunk_id);
  }
  [[nodiscard]] TaskType taskType() const override { return TaskType::kIO; }
  [[nodiscard]] bool isReady() const override { return true; }

 private:
  CellIndex idx_;
  RuntimeContext& ctx_;
};

class WriteHTask : public Task {
 public:
  WriteHTask(const CellIndex& idx, RuntimeContext& ctx) : Task(), idx_(idx), ctx_(ctx) {}

  void execute() override;
  bool submit() override;
  [[nodiscard]] bool isAsync() const override { return true; }
  [[nodiscard]] std::string name() const override {
    return fmt::format("WriteH[{}][{}]", idx_.layer, idx_.chunk_id);
  }
  [[nodiscard]] TaskType taskType() const override { return TaskType::kIO; }
  [[nodiscard]] bool isReady() const override { return true; }

 private:
  CellIndex idx_;
  RuntimeContext& ctx_;
};

enum class FsyncTarget { kK, kV, kH, kWatermark };

class FsyncTask : public Task {
 public:
  FsyncTask(FsyncTarget target, RuntimeContext& ctx) : Task(), target_(target), ctx_(ctx) {}

  void execute() override;
  bool submit() override;
  [[nodiscard]] bool isAsync() const override { return true; }
  [[nodiscard]] std::string name() const override {
    const char* names[] = {"K", "V", "H", "Watermark"};
    return fmt::format("Fsync[{}]", names[static_cast<int>(target_)]);
  }
  [[nodiscard]] TaskType taskType() const override { return TaskType::kIO; }
  [[nodiscard]] bool isReady() const override { return true; }

 private:
  FsyncTarget target_;
  RuntimeContext& ctx_;
};

class WriteWatermarkTask : public Task {
 public:
  explicit WriteWatermarkTask(RuntimeContext& ctx) : Task(), ctx_(ctx) {}

  void execute() override;
  [[nodiscard]] std::string name() const override { return "WriteWatermark"; }
  [[nodiscard]] TaskType taskType() const override { return TaskType::kIO; }
  [[nodiscard]] bool isReady() const override { return true; }

 private:
  RuntimeContext& ctx_;
};

}  // namespace mllm::models::qwen3_i
