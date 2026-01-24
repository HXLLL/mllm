// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/models/qwen3_i/grid_scheduler.hpp"

namespace mllm::models::qwen3_i {

class SimpleGridScheduler : public GridScheduler {
 public:
  using GridScheduler::GridScheduler;

 protected:
  Task* selectNext() override;
  [[nodiscard]] bool isDone() const override;

 private:
  bool done_{};
};

}  // namespace mllm::models::qwen3_i
