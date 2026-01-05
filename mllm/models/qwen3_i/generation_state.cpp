#include <fstream>
#include <filesystem>
#include <nlohmann/json.hpp>
#include "mllm/utils/Common.hpp"
#include "mllm/utils/Log.hpp"

#include "mllm/models/qwen3_i/generation_state.hpp"

namespace mllm::models::qwen3_i {

namespace fs = std::filesystem;

template<typename... Args>
static inline std::ofstream open_ofstream(const fs::path& path, Args&&... args) {
  std::ofstream ofs;
  ofs.open(path, std::forward<Args>(args)...);
  if (!ofs.is_open()) { MLLM_ERROR_EXIT(ExitCode::kIOError, "Failed to open file {} for writing", path.string()); }
  return ofs;
}

template<typename... Args>
static inline std::ifstream open_ifstream(const fs::path& path, Args&&... args) {
  std::ifstream ifs;
  ifs.open(path, std::forward<Args>(args)...);
  if (!ifs.is_open()) { MLLM_ERROR_EXIT(ExitCode::kIOError, "Failed to open file {} for reading", path.string()); }
  return ifs;
}

GenerationState::GenerationState(const Qwen3Config& cfg, const fs::path& path)
    : path_(path),
      max_length_(cfg.max_cache_length),
      layer_nums_(cfg.num_hidden_layers),
      q_heads_(cfg.num_attention_heads),
      kv_heads_(cfg.num_key_value_heads),
      kv_dim_(cfg.head_dim),
      hidden_size_(cfg.hidden_size) {}

void GenerationState::load() {
  MLLM_INFO("GenerationState: Loading state from {}", path_.string());

  loadMetadata();

  output_tokens_ = Tensor::empty({max_length_, hidden_size_}, kFloat32, kCPU).alloc();
  {
    auto output_tokens_file = open_ifstream(path_ / "output_tokens.bin", std::ios::binary);
    auto out_ptr = output_tokens_.ptrAt<char>({0, 0});
    output_tokens_file.read(out_ptr, num_output_tokens_ * hidden_size_ * ELEMENT_SIZE);
  }

  {
    auto kv_cache_file = open_ifstream(path_ / "kv_cache.bin", std::ios::binary);

    k_cache_.reserve(layer_nums_);
    for (int i = 0; i < layer_nums_; ++i) {
      k_cache_.emplace_back(Tensor::empty({1, q_heads_, max_length_, kv_dim_}, kFloat32, kCPU).alloc());
      auto k_ptr = k_cache_[i].ptrAt<char>({0, 0, 0, 0});
      kv_cache_file.read(k_ptr, max_length_ * kv_heads_ * kv_dim_ * ELEMENT_SIZE);
    }
    v_cache_.reserve(layer_nums_);
    for (int i = 0; i < layer_nums_; ++i) {
      v_cache_.emplace_back(Tensor::empty({1, q_heads_, max_length_, kv_dim_}, kFloat32, kCPU).alloc());
      auto v_ptr = v_cache_[i].ptrAt<char>({0, 0, 0, 0});
      kv_cache_file.read(v_ptr, max_length_ * kv_heads_ * kv_dim_ * ELEMENT_SIZE);
    }
  }

  {
    auto h_cache_file = open_ifstream(path_ / "h_cache.bin", std::ios::binary);
    h_cache_.reserve(layer_nums_ + 1);
    for (int i = 0; i < layer_nums_ + 1; ++i) {
      h_cache_.emplace_back(Tensor::empty({1, max_length_, hidden_size_}, kFloat32, kCPU).alloc());
      auto h_ptr = h_cache_[i].ptrAt<char>({0, 0, 0});
      h_cache_file.read(h_ptr, max_length_ * hidden_size_ * ELEMENT_SIZE);
    }
  }
}

void GenerationState::create() {
  input_tokens_ = {};
  output_tokens_ = Tensor::empty({max_length_, hidden_size_}, kFloat32, kCPU).alloc();
  k_cache_.reserve(layer_nums_);
  v_cache_.reserve(layer_nums_);
  h_cache_.reserve(layer_nums_ + 1);
  for (int i = 0; i < layer_nums_; ++i) {
    k_cache_.emplace_back(Tensor::empty({1, q_heads_, max_length_, kv_dim_}, kFloat32, kCPU).alloc());
    v_cache_.emplace_back(Tensor::empty({1, q_heads_, max_length_, kv_dim_}, kFloat32, kCPU).alloc());
    h_cache_.emplace_back(Tensor::empty({1, max_length_, hidden_size_}, kFloat32, kCPU).alloc());
  }
  h_cache_.emplace_back(Tensor::empty({1, max_length_, hidden_size_}, kFloat32, kCPU).alloc());

  save();
}

void GenerationState::save() const {
  if (!fs::exists(path_)) {
    fs::create_directories(path_);
  }

  saveMetadata();

  {
    auto output_tokens_file = open_ofstream(path_ / "output_tokens.bin", std::ios::binary);
    auto out_ptr = output_tokens_.cptrAt<char>({0, 0});
    output_tokens_file.write(out_ptr, num_output_tokens_ * hidden_size_ * ELEMENT_SIZE);
  }

  {
    auto h_cache_file = open_ofstream(path_ / "h_cache.bin", std::ios::binary);
    for (int i = 0; i <= layer_nums_; ++i) {
      auto h_ptr = h_cache_[i].cptrAt<char>({0, 0, 0});
      h_cache_file.write(h_ptr, max_length_ * hidden_size_ * ELEMENT_SIZE);
    }
  }

  {
    auto kv_cache_file = open_ofstream(path_ / "kv_cache.bin", std::ios::binary);
    for (int i = 0; i < layer_nums_; ++i) {
      auto k_ptr = k_cache_[i].cptrAt<char>({0, 0, 0, 0});
      kv_cache_file.write(k_ptr, max_length_ * kv_heads_ * kv_dim_ * ELEMENT_SIZE);
    }
    for (int i = 0; i < layer_nums_; ++i) {
      auto v_ptr = v_cache_[i].cptrAt<char>({0, 0, 0, 0});
      kv_cache_file.write(v_ptr, max_length_ * kv_heads_ * kv_dim_ * ELEMENT_SIZE);
    }
  }
}

void GenerationState::sync_cache() const {
  save();
  // TODO: use a more efficient approach
}

void GenerationState::start(const Tensor& token_ids) {
  MLLM_RT_ASSERT(token_ids.shape()[0] == 1 && token_ids.dtype() == kInt64);
  auto seq_len = token_ids.shape()[1];
  for (int i = 0; i < seq_len; ++i) { input_tokens_.push_back(*token_ids.cptrAt<int64_t>({0, i})); }
  started_ = 1;
}

int GenerationState::hasStarted() const { return started_; }

void GenerationState::start_decode(const Tensor& token_id) {
  MLLM_RT_ASSERT(token_id.shape() == std::vector<int32_t>({1, 1}) && token_id.dtype() == kInt64);
}

void GenerationState::appendOutputToken(const Tensor& token) {
  MLLM_RT_ASSERT(token.shape() == std::vector<int32_t>({1, 1, hidden_size_}) && token.dtype() == kFloat32);
  auto src = token.cptrAt<mllm_byte_t>({0, 0, 0});
  auto dst = output_tokens_.ptrAt<mllm_byte_t>({num_output_tokens_, 0});
  std::memcpy(dst, src, hidden_size_ * ELEMENT_SIZE);
  num_output_tokens_++;
}

void GenerationState::set_prefill_done() { prefill_done_ = 1; }

int GenerationState::prefill_done() const { return prefill_done_; }

const std::vector<int64_t>& GenerationState::getInputTokens() const { return input_tokens_; }

void GenerationState::updateKV(int layer_idx, int offset, int count, const Tensor& k, const Tensor& v) {
  MLLM_RT_ASSERT(k.shape() == std::vector<int32_t>({1, kv_heads_, count, kv_dim_}));
  MLLM_RT_ASSERT(v.shape() == std::vector<int32_t>({1, kv_heads_, count, kv_dim_}));

  auto repeat_times = q_heads_ / kv_heads_;

  for (int h = 0; h < kv_heads_; ++h) {
    auto k_ptr = k.cptrAt<mllm_byte_t>({0, h, 0, 0});
    auto v_ptr = v.cptrAt<mllm_byte_t>({0, h, 0, 0});
    for (int r = 0; r < repeat_times; ++r) {
      auto k_cache_ptr = k_cache_[layer_idx].ptrAt<mllm_byte_t>({0, h * repeat_times + r, offset, 0});
      auto v_cache_ptr = v_cache_[layer_idx].ptrAt<mllm_byte_t>({0, h * repeat_times + r, offset, 0});
      std::memcpy(k_cache_ptr, k_ptr, count * kv_dim_ * ELEMENT_SIZE);
      std::memcpy(v_cache_ptr, v_ptr, count * kv_dim_ * ELEMENT_SIZE);
    }
  }
}

void GenerationState::updateH(int layer_idx, int offset, int count, const Tensor& h) {
  MLLM_RT_ASSERT(h.shape() == std::vector<int32_t>({1, count, hidden_size_}) && h.dtype() == kFloat32);
  auto h_ptr = h.cptrAt<mllm_byte_t>({0, 0, 0});
  auto h_cache_ptr = h_cache_[layer_idx].ptrAt<mllm_byte_t>({0, offset, 0});
  std::memcpy(h_cache_ptr, h_ptr, count * hidden_size_ * ELEMENT_SIZE);
}

Tensor GenerationState::getH(int layer_idx, int offset, int count) {
  return h_cache_[layer_idx][{0, {offset, offset + count}, kAll}];
}

std::array<Tensor, 2> GenerationState::getKV(int layer_idx) {
  MLLM_ERROR_EXIT(ExitCode::kCoreError, "to be implemented");
  return {k_cache_[layer_idx][{0, kAll, kAll, kAll}], v_cache_[layer_idx][{0, kAll, kAll, kAll}]};
}

std::array<Tensor, 2> GenerationState::getKV(int layer_idx, int offset, int count) {
  return {{
      k_cache_[layer_idx][{0, kAll, {offset, offset + count}, kAll}],
      v_cache_[layer_idx][{0, kAll, {offset, offset + count}, kAll}],
  }};
}

void GenerationState::loadMetadata() {
  nlohmann::json json_data;
  {
    auto metadata_file = open_ifstream(path_ / "metadata.json");
    metadata_file >> json_data;
  }

  num_output_tokens_ = json_data.at("num_output_tokens");
  max_length_ = json_data.at("max_length");
  layer_nums_ = json_data.at("layer_nums");
  q_heads_ = json_data.at("q_heads");
  kv_heads_ = json_data.at("kv_heads");
  kv_dim_ = json_data.at("kv_dim");
  hidden_size_ = json_data.at("hidden_size");
  prefill_done_ = json_data.at("prefill_done");
  started_ = json_data.at("started");
  input_tokens_ = json_data.at("input_tokens").get<std::vector<int64_t>>();
}

void GenerationState::saveMetadata() const {
  nlohmann::json json_data;
  json_data["num_output_tokens"] = num_output_tokens_;
  json_data["max_length"] = max_length_;
  json_data["layer_nums"] = layer_nums_;
  json_data["q_heads"] = q_heads_;
  json_data["kv_heads"] = kv_heads_;
  json_data["kv_dim"] = kv_dim_;
  json_data["hidden_size"] = hidden_size_;
  json_data["input_tokens"] = input_tokens_;
  json_data["prefill_done"] = prefill_done_;
  json_data["started"] = started_;
  {
    auto metadata_file = open_ofstream(path_ / "metadata.json");
    metadata_file << json_data.dump(2);
  }
}

}  // namespace mllm::models::qwen3_i
