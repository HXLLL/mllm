#include <fstream>
#include <filesystem>
#include <nlohmann/json.hpp>
#include "mllm/utils/Common.hpp"
#include "mllm/utils/Log.hpp"

#include "mllm/models/qwen3_i/generation_state.hpp"

namespace mllm::models::qwen3_i {

GenerationState::InitParams GenerationState::InitParams::make_default(const Qwen3Config& cfg, const std::filesystem::path& path) {
  return {
      .path = path,
      .max_length = cfg.max_cache_length,
      .layer_nums = cfg.num_hidden_layers,
      .q_heads = cfg.num_attention_heads,
      .kv_heads = cfg.num_key_value_heads,
      .kv_dim = cfg.head_dim,
      .hidden_size = cfg.hidden_size,
      .num_output_tokens = 0,
      .input_tokens = {},
      .output_tokens = {},
      .k_cache = {},
      .v_cache = {},
      .h_cache = {},
  };
}

GenerationState::GenerationState(InitParams&& params)
    : path_(params.path),
      max_length_(params.max_length),
      layer_nums_(params.layer_nums),
      q_heads_(params.q_heads),
      kv_heads_(params.kv_heads),
      kv_dim_(params.kv_dim),
      hidden_size_(params.hidden_size),
      num_output_tokens_(params.num_output_tokens),
      input_tokens_(std::move(params.input_tokens)),
      output_tokens_(std::move(params.output_tokens)),
      k_cache_(std::move(params.k_cache)),
      v_cache_(std::move(params.v_cache)),
      h_cache_(std::move(params.h_cache)) {}

GenerationState::ptr GenerationState::create_or_recover(const Qwen3Config& cfg, const std::filesystem::path& path) {
  if (std::filesystem::exists(path)) {
    return recover(cfg, path);
  } else {
    return create(cfg, path);
  }
}

GenerationState::ptr GenerationState::recover(const Qwen3Config& cfg, const std::filesystem::path& path) {
  MLLM_ERROR_EXIT(-1, "not implemented");
  return nullptr;
}

GenerationState::ptr GenerationState::create(const Qwen3Config& cfg, const std::filesystem::path& path) {
  MLLM_INFO("GenerationState: Creating state at {}", path.string());
  auto layer_nums = cfg.num_hidden_layers;

  auto params = InitParams::make_default(cfg, path);
  params.input_tokens = {};
  params.output_tokens = Tensor::empty({cfg.max_cache_length, cfg.hidden_size}, kFloat32, kCPU).alloc();
  params.k_cache.reserve(layer_nums);
  params.v_cache.reserve(layer_nums);
  params.h_cache.reserve(layer_nums + 1);
  for (int i = 0; i < layer_nums; ++i) {
    params.k_cache.emplace_back(Tensor::empty({cfg.num_attention_heads, cfg.max_cache_length, cfg.head_dim}, kFloat32, kCPU).alloc());
    params.v_cache.emplace_back(Tensor::empty({cfg.num_attention_heads, cfg.max_cache_length, cfg.head_dim}, kFloat32, kCPU).alloc());
    params.h_cache.emplace_back(Tensor::empty({cfg.max_cache_length, cfg.hidden_size}, kFloat32, kCPU).alloc());
  }
  params.h_cache.emplace_back(Tensor::empty({cfg.max_cache_length, cfg.hidden_size}, kFloat32, kCPU).alloc());
  return std::make_shared<GenerationState>(std::move(params));
}

void GenerationState::start_prefill(const Tensor& token_ids) {
  MLLM_RT_ASSERT(token_ids.shape()[0] == 1 && token_ids.dtype() == kInt64);
  auto seq_len = token_ids.shape()[1];
  for (int i = 0; i < seq_len; ++i) {
    input_tokens_.push_back(*token_ids.cptrAt<int64_t>({0, i}));
  }
}

void GenerationState::start_decode(const Tensor& token_id) {
  MLLM_RT_ASSERT(token_id.shape() == std::vector<int32_t>({1, 1}) && token_id.dtype() == kInt64);
}

void GenerationState::append_output_token(const Tensor& token) {
  MLLM_RT_ASSERT(token.shape() == std::vector<int32_t>({1, 1, hidden_size_}) && token.dtype() == kFloat32);
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

  std::filesystem::path h_cache_path = path_ / "h_cache.bin";
  std::ofstream h_cache_file(h_cache_path, std::ios::binary);
  for (int i = 0; i < layer_nums_; ++i) {
    auto h_ptr = h_cache_[i].cptrAt<mllm_byte_t>({0, 0});
    h_cache_file.write(reinterpret_cast<const char*>(h_ptr), max_length_ * hidden_size_ * bytesOfType(kFloat32) / lanesOfType(kFloat32));
  }

  std::filesystem::path kv_cache_path = path_ / "kv_cache.bin";
  std::ofstream kv_cache_file(kv_cache_path, std::ios::binary);
  for (int i = 0; i < layer_nums_; ++i) {
    auto k_ptr = k_cache_[i].cptrAt<mllm_byte_t>({0, 0, 0});
    kv_cache_file.write(reinterpret_cast<const char*>(k_ptr), max_length_ * kv_heads_ * kv_dim_ * bytesOfType(kFloat32) / lanesOfType(kFloat32));
  }
  for (int i = 0; i < layer_nums_; ++i) {
    auto v_ptr = v_cache_[i].cptrAt<mllm_byte_t>({0, 0, 0});
    kv_cache_file.write(reinterpret_cast<const char*>(v_ptr), max_length_ * kv_heads_ * kv_dim_ * bytesOfType(kFloat32) / lanesOfType(kFloat32));
  }
}

void GenerationState::sync_cache() {
  // MLLM_INFO("GenerationState: Syncing cache");
}

void GenerationState::update_kv(int layer_idx, int offset, int count, const Tensor &k, const Tensor &v) {
  MLLM_RT_ASSERT(k.shape() == std::vector<int32_t>({1, kv_heads_, count, kv_dim_}));
  MLLM_RT_ASSERT(v.shape() == std::vector<int32_t>({1, kv_heads_, count, kv_dim_}));
  
  auto repeat_times = q_heads_ / kv_heads_;

  for (int h = 0; h < kv_heads_; ++h) {
    auto k_ptr = k.cptrAt<mllm_byte_t>({0, h, 0, 0});
    auto v_ptr = v.cptrAt<mllm_byte_t>({0, h, 0, 0});
    for (int r = 0; r < repeat_times; ++r) {
      auto k_cache_ptr = k_cache_[layer_idx].ptrAt<mllm_byte_t>({h * repeat_times + r, offset, 0});
      auto v_cache_ptr = v_cache_[layer_idx].ptrAt<mllm_byte_t>({h * repeat_times + r, offset, 0});
      std::memcpy(k_cache_ptr, k_ptr, count * kv_dim_ * bytesOfType(kFloat32) / lanesOfType(kFloat32));
      std::memcpy(v_cache_ptr, v_ptr, count * kv_dim_ * bytesOfType(kFloat32) / lanesOfType(kFloat32));
    }
  }
}

void GenerationState::update_h(int layer_idx, int offset, int count, const Tensor &h) {
  MLLM_RT_ASSERT(h.shape() == std::vector<int32_t>({1, count, hidden_size_}) && h.dtype() == kFloat32);
  auto h_ptr = h.cptrAt<mllm_byte_t>({0, 0, 0});
  auto h_cache_ptr = h_cache_[layer_idx].ptrAt<mllm_byte_t>({offset, 0});
  std::memcpy(h_cache_ptr, h_ptr, count * hidden_size_ * bytesOfType(kFloat32) / lanesOfType(kFloat32));
}

std::array<Tensor, 2> GenerationState::get_kv(int layer_idx) {
  MLLM_ERROR_EXIT(ExitCode::kCoreError, "to be implemented");
  return { k_cache_[layer_idx], v_cache_[layer_idx] };
}

std::array<Tensor, 2> GenerationState::get_kv(int layer_idx, int offset, int count) {
  return {{
    k_cache_[layer_idx][{kAll, {offset, offset + count}, kAll}],
    v_cache_[layer_idx][{kAll, {offset, offset + count}, kAll}],
  }};
}

}  // namespace mllm::models::qwen3_i
