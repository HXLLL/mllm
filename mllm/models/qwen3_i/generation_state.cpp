#include <nlohmann/json.hpp>

#include "mllm/models/qwen3_i/generation_state.hpp"

namespace mllm::models::qwen3_i {

GenerationState::GenerationState(const std::filesystem::path& path, int max_length, int layer_nums, int q_heads, int kv_heads,
                                 int kv_dim)
    : path_(path), kv_cache_(max_length, layer_nums, q_heads, kv_heads, kv_dim, kFloat32, kFloat32, kCPU, false) {}

void GenerationState::save() const {
  MLLM_INFO("GenerationState: Saving state to {}", path_.string());
}


GenerationState::ptr GenerationState::create_or_recover(const Qwen3Config& cfg, const std::filesystem::path& path) {
  if (std::filesystem::exists(path)) {
    return recover(cfg, path);
  } else {
    return create(cfg, path);
  }
}

GenerationState::ptr GenerationState::recover(const Qwen3Config& cfg, const std::filesystem::path& path) {
  MLLM_INFO("GenerationState: Recovering state from {}", path.string());
  return std::make_shared<GenerationState>(path, cfg.max_cache_length, cfg.num_hidden_layers, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim);
}

GenerationState::ptr GenerationState::create(const Qwen3Config& cfg, const std::filesystem::path& path) {
  MLLM_INFO("GenerationState: Creating state at {}", path.string());
  return std::make_shared<GenerationState>(path, cfg.max_cache_length, cfg.num_hidden_layers, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim);
}

void GenerationState::start_generation(const Tensor& token_ids) {
  input_tokens_ = token_ids.clone();
}












void GenerationState::sync_cache() {
  // MLLM_INFO("GenerationState: Syncing cache");
}

void GenerationState::checkpoint() const {
  
}

void GenerationState::update_kv(int layer_idx, int offset, int count, const Tensor &k, const Tensor &v) {
  kv_cache_.updateKVCache(layer_idx, k, v);
}

void GenerationState::update_h(int layer_idx, int offset, int count, const Tensor &h) {
  // h_cache_.updateHiddenStateCache(layer_idx, offset, count, h);
}

std::optional<std::array<Tensor, 2>> GenerationState::get_kv(int layer_idx, int offset, int count) {
  // MLLM_INFO("GenerationState: Getting KV cache for layer {}, token offset {}, token count {}", layer_idx, offset, count);
  return {kv_cache_.getKVCache(layer_idx)};
}

}  // namespace mllm::models::qwen3_i
