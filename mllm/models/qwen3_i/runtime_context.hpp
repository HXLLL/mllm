// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/models/qwen3_i/generation_state.hpp"
#include "mllm/models/qwen3_i/parameter_loader.hpp"

namespace mllm::models::qwen3_i {

// Runtime context for task execution.
// Bundles GenerationState and ParameterLoader so tasks only need one reference.
class RuntimeContext {
 public:
  RuntimeContext(GenerationState& state, ParameterLoader& loader)
      : state_(state), loader_(loader) {}

  GenerationState& state() { return state_; }
  ParameterLoader& loader() { return loader_; }

  [[nodiscard]] const GenerationState& state() const { return state_; }
  [[nodiscard]] const ParameterLoader& loader() const { return loader_; }

 private:
  GenerationState& state_;
  ParameterLoader& loader_;
};

}  // namespace mllm::models::qwen3_i
