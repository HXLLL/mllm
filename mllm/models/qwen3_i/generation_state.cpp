#include "mllm/models/qwen3_i/generation_state.hpp"
#include <nlohmann/json.hpp>

namespace mllm::models::qwen3_i {

GenerationState::GenerationState() = default;

void GenerationState::save(const std::filesystem::path& path) const {
  MLLM_INFO("GenerationState: Saving state to {}", path.string());
}

GenerationState GenerationState::load(const std::filesystem::path& path) {
  MLLM_INFO("GenerationState: Loading state from {}", path.string());
  return {};
}

void GenerationState::sync_cache() {
  MLLM_INFO("GenerationState: Syncing cache");
}

void GenerationState::update_kv(int layer_idx, int token_idx, const Tensor &k, const Tensor &v) {
  MLLM_INFO("GenerationState: Updating KV cache");
}

std::optional<std::array<Tensor, 2>> GenerationState::get_kv(int layer_idx, int token_idx) const {
  MLLM_INFO("GenerationState: Getting KV cache");
  return std::nullopt;
}

}  // namespace mllm::models::qwen3_i
