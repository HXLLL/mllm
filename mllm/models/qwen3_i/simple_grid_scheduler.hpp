// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/models/qwen3_i/grid_scheduler.hpp"

namespace mllm::models::qwen3_i {

class SimpleGridScheduler : public GridScheduler {
 public:
  SimpleGridScheduler(int num_layers, int num_chunks, GenerationState& state, ParameterLoader& parameter_loader)
      : GridScheduler(num_layers, num_chunks, state, parameter_loader) {}

 protected:
  GridTask* selectNext() override;
  [[nodiscard]] bool isDone() const override;

 private:
  bool done_{};
};

}  // namespace mllm::models::qwen3_i
