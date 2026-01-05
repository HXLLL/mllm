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

GenerationState::ptr GenerationState::create_or_recover(const Qwen3Config& cfg, const std::filesystem::path& path) {
  if (std::filesystem::exists(path)) {
    return recover(cfg, path);
  } else {
    return create(cfg, path);
  }
}

GenerationState::ptr GenerationState::recover(const Qwen3Config& cfg, const std::filesystem::path& path) {
  MLLM_INFO("GenerationState: Recovering state from {}", path.string());
  
  std::ifstream metadata_file(path / "metadata.json");
  nlohmann::json json_data;
  metadata_file >> json_data;
  metadata_file.close();
  
  int num_output_tokens = json_data.value("num_output_tokens", 0);
  int max_length = json_data.value("max_length", cfg.max_cache_length);
  int layer_nums = json_data.value("layer_nums", cfg.num_hidden_layers);
  int q_heads = json_data.value("q_heads", cfg.num_attention_heads);
  int kv_heads = json_data.value("kv_heads", cfg.num_key_value_heads);
  int kv_dim = json_data.value("kv_dim", cfg.head_dim);
  int hidden_size = json_data.value("hidden_size", cfg.hidden_size);
  std::vector<int64_t> input_tokens = json_data.value("input_tokens", std::vector<int64_t>());
  
  auto params = InitParams::make_default(cfg, path);
  params.num_output_tokens = num_output_tokens;
  params.max_length = max_length;
  params.layer_nums = layer_nums;
  params.q_heads = q_heads;
  params.kv_heads = kv_heads;
  params.kv_dim = kv_dim;
  params.hidden_size = hidden_size;
  params.input_tokens = input_tokens;
  
  params.output_tokens = Tensor::empty({max_length, hidden_size}, kFloat32, kCPU).alloc();
  std::ifstream output_tokens_file(path / "output_tokens.bin", std::ios::binary);
  auto out_ptr = params.output_tokens.ptrAt<char>({0, 0});
  output_tokens_file.read(out_ptr, num_output_tokens * hidden_size * sizeof(float_t));
  
  std::ifstream kv_cache_file(path / "kv_cache.bin", std::ios::binary);

  params.k_cache.reserve(layer_nums);
  for (int i = 0; i < layer_nums; ++i) {
    params.k_cache.emplace_back(Tensor::empty({q_heads, max_length, kv_dim}, kFloat32, kCPU).alloc());
    auto k_ptr = params.k_cache[i].ptrAt<char>({0, 0, 0});
    kv_cache_file.read(k_ptr, max_length * kv_heads * kv_dim * bytesOfType(kFloat32) / lanesOfType(kFloat32));
  }
  params.v_cache.reserve(layer_nums);
  for (int i = 0; i < layer_nums; ++i) {
    params.v_cache.emplace_back(Tensor::empty({q_heads, max_length, kv_dim}, kFloat32, kCPU).alloc());
    auto v_ptr = params.v_cache[i].ptrAt<char>({0, 0, 0});
    kv_cache_file.read(v_ptr, max_length * kv_heads * kv_dim * bytesOfType(kFloat32) / lanesOfType(kFloat32));
  }
  kv_cache_file.close();
  
  std::ifstream h_cache_file(path / "h_cache.bin", std::ios::binary);
  params.h_cache.reserve(layer_nums + 1);
  for (int i = 0; i < layer_nums + 1; ++i) {
    params.h_cache.emplace_back(Tensor::empty({max_length, hidden_size}, kFloat32, kCPU).alloc());
    auto h_ptr = params.h_cache[i].ptrAt<char>({0, 0});
    h_cache_file.read(h_ptr, max_length * hidden_size * bytesOfType(kFloat32) / lanesOfType(kFloat32));
  }
  h_cache_file.close();

  return std::make_shared<GenerationState>(std::move(params));
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

void GenerationState::prefill_done() {
  prefill_done_ = true;
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

  std::ofstream metadata_file(path_ / "metadata.json");
  metadata_file << json_data.dump(2);
  metadata_file.close();

  std::ofstream output_tokens_file(path_ / "output_tokens.bin", std::ios::binary);
  auto out_ptr = output_tokens_.cptrAt<char>({0, 0});
  output_tokens_file.write(out_ptr, num_output_tokens_ * hidden_size_ * sizeof(float_t));
  output_tokens_file.close();

  std::ofstream h_cache_file(path_ / "h_cache.bin", std::ios::binary);
  for (int i = 0; i < layer_nums_; ++i) {
    auto h_ptr = h_cache_[i].cptrAt<char>({0, 0});
    h_cache_file.write(h_ptr, max_length_ * hidden_size_ * bytesOfType(kFloat32) / lanesOfType(kFloat32));
  }
  h_cache_file.close();

  std::ofstream kv_cache_file(path_ / "kv_cache.bin", std::ios::binary);
  for (int i = 0; i < layer_nums_; ++i) {
    auto k_ptr = k_cache_[i].cptrAt<char>({0, 0, 0});
    kv_cache_file.write(k_ptr, max_length_ * kv_heads_ * kv_dim_ * bytesOfType(kFloat32) / lanesOfType(kFloat32));
  }
  for (int i = 0; i < layer_nums_; ++i) {
    auto v_ptr = v_cache_[i].cptrAt<char>({0, 0, 0});
    kv_cache_file.write(v_ptr, max_length_ * kv_heads_ * kv_dim_ * bytesOfType(kFloat32) / lanesOfType(kFloat32));
  }
  kv_cache_file.close();
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
