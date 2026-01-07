#pragma once

#include <queue>
#include <unordered_map>
#include "mllm/core/Tensor.hpp"
#include "mllm/mllm.hpp"
#include "mllm/models/qwen3/configuration_qwen3.hpp"

namespace mllm::models::qwen3_i {

struct PairHash {
  std::size_t operator()(const std::pair<int, int>& p) const noexcept {
    uint64_t hi = static_cast<uint32_t>(p.first);
    uint64_t lo = static_cast<uint32_t>(p.second);
    return std::hash<uint64_t>{}((hi << 32) | lo);
  }
};

class GenerationState {
 public:
  using Qwen3Config = models::qwen3::Qwen3Config;

  explicit GenerationState(const Qwen3Config& cfg, const std::filesystem::path& path);

  void load();
  void create();
  void save() const;
  void sync_cache() const;

  void start(const Tensor& token_ids);
  [[nodiscard]] int hasStarted() const;

  void start_decode(const Tensor& token_id);

  [[nodiscard]] const std::vector<int64_t>& getInputTokens() const;

  [[nodiscard]] int getMinWatermark(int offset, int count) const;
  [[nodiscard]] bool isPositionComplete(int pos) const;

  void updateKV(int layer_idx, int offset, int count, const Tensor& k, const Tensor& v);
  [[nodiscard]] std::array<Tensor, 2> getKV(int layer_idx);
  [[nodiscard]] std::array<Tensor, 2> getKV(int layer_idx, int offset, int count);

  void updateH(int layer_idx, int offset, int count, const Tensor& h);
  Tensor getH(int layer_idx, int offset, int count);

 private:
  const size_t ELEMENT_SIZE = bytesOfType(kFloat32) / lanesOfType(kFloat32);

  void loadMetadata();
  void saveMetadata() const;

  std::filesystem::path path_;

  int max_length_;
  int layer_nums_;
  int q_heads_;
  int kv_heads_;
  int kv_dim_;
  int hidden_size_;
  int started_ = 0;

  std::vector<int8_t> layer_watermark_;  // -1 = not computed, N = h_cache_[0..N] valid

  std::vector<int64_t> input_tokens_;
  std::vector<Tensor> k_cache_;  // Shape: [layer_nums, 1, q_heads, max_cache_length, kv_dims]
  std::vector<Tensor> v_cache_;  // Shape: [layer_nums, 1, q_heads, max_cache_length, kv_dims]
  std::vector<Tensor> h_cache_;  // Shape: [layer_nums + 1, 1, max_cache_length, hidden_size]

  std::unordered_map<std::pair<int, int>, bool, PairHash> dirty;
  std::queue<std::pair<int, int>> dirty_queue;
};

}  // namespace mllm::models::qwen3_i
