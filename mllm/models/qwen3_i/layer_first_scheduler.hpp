// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/models/qwen3_i/grid_scheduler.hpp"

namespace mllm::models::qwen3_i {

class LayerFirstScheduler : public GridScheduler {
 public:
  using GridScheduler::GridScheduler;

 protected:
  Task* selectNext() override;
  [[nodiscard]] bool isDone() const override;

 private:
  Task* selectNextParamTask();
  Task* selectNextComputeTask();

  enum class Phase { LoadParam, Compute, Done };

  Phase phase_ = Phase::LoadParam;
  int param_layer_ = 0;
  int current_layer_ = 0;
  int current_chunk_ = 0;
};

}  // namespace mllm::models::qwen3_i
