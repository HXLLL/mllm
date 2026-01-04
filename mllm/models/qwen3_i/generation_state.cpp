#include <fstream>
#include <filesystem>
#include <nlohmann/json.hpp>
#include "mllm/utils/Common.hpp"
#include "mllm/utils/Log.hpp"

#include "mllm/models/qwen3_i/generation_state.hpp"

namespace mllm::models::qwen3_i {

GenerationState::GenerationState(const std::filesystem::path& path, int max_length, int layer_nums, int q_heads, int kv_heads,
                                 int kv_dim, int hidden_size)
    : num_output_tokens_(0),
      max_length_(max_length),
      layer_nums_(layer_nums),
      q_heads_(q_heads),
      kv_heads_(kv_heads),
      kv_dim_(kv_dim),
      hidden_size_(hidden_size),
      output_tokens_(Tensor::empty({max_length_, hidden_size_}, kFloat32, kCPU).alloc()),
      kv_cache_(max_length, layer_nums, q_heads, kv_heads, kv_dim, kFloat32, kFloat32, kCPU, false),
      path_(path) {}

GenerationState::ptr GenerationState::create_or_recover(const Qwen3Config& cfg, const std::filesystem::path& path) {
  if (std::filesystem::exists(path)) {
    return recover(cfg, path);
  } else {
    return create(cfg, path);
  }
}

GenerationState::ptr GenerationState::recover(const Qwen3Config& cfg, const std::filesystem::path& path) {
  MLLM_INFO("GenerationState: Recovering state from {}", path.string());
  return std::make_shared<GenerationState>(path, cfg.max_cache_length, cfg.num_hidden_layers, cfg.num_attention_heads,
                                           cfg.num_key_value_heads, cfg.head_dim, cfg.hidden_size);
}

GenerationState::ptr GenerationState::create(const Qwen3Config& cfg, const std::filesystem::path& path) {
  MLLM_INFO("GenerationState: Creating state at {}", path.string());
  return std::make_shared<GenerationState>(path, cfg.max_cache_length, cfg.num_hidden_layers, cfg.num_attention_heads,
                                           cfg.num_key_value_heads, cfg.head_dim, cfg.hidden_size);
}

void GenerationState::start_generation(const Tensor& token_ids) {
  input_tokens_ = token_ids.toVector<int64_t>();
  mllm::print(token_ids.shape());
}

void GenerationState::append_output_token(const Tensor& token) {
  MLLM_RT_ASSERT(token.shape()[0] == 1 && token.shape()[1] == 1 && token.shape()[2] == hidden_size_);
  MLLM_RT_ASSERT(token.dtype() == kFloat32);
  auto src = token.cptrAt<float_t>({0, 0, 0});
  auto dst = output_tokens_.ptrAt<float_t>({num_output_tokens_, 0});
  std::memcpy(dst, src, hidden_size_ * sizeof(float_t));
  num_output_tokens_++;
}

void GenerationState::save() const {
  nlohmann::json json_data;

  json_data["num_output_tokens"] = num_output_tokens_;
  json_data["max_length"] = max_length_;
  json_data["layer_nums"] = layer_nums_;
  json_data["q_heads"] = q_heads_;
  json_data["kv_heads"] = kv_heads_;
  json_data["kv_dim"] = kv_dim_;
  json_data["hidden_size"] = hidden_size_;
  json_data["input_tokens"] = input_tokens_;
  std::filesystem::path metadata_path = path_ / "metadata.json";
  std::ofstream metadata_file(metadata_path);
  metadata_file << json_data.dump(2);

  std::filesystem::path output_tokens_path = path_ / "output_tokens.bin";
  std::ofstream output_tokens_file(output_tokens_path, std::ios::binary);
  size_t num_elements = num_output_tokens_ * hidden_size_;
  const float_t* out_ptr = output_tokens_.cptrAt<float_t>({0, 0});
  output_tokens_file.write(reinterpret_cast<const char*>(out_ptr), num_elements * sizeof(float_t));
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
