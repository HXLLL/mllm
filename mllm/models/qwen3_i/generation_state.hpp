#pragma once

#include <queue>
#include <unordered_map>
#include <optional>
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
  using ptr = std::shared_ptr<GenerationState>;
  using Qwen3Config = models::qwen3::Qwen3Config;

  struct InitParams {
    std::filesystem::path path;
    int max_length;
    int layer_nums;
    int q_heads;
    int kv_heads;
    int kv_dim;
    int hidden_size;
    int num_output_tokens;
    std::vector<int64_t> input_tokens;
    Tensor output_tokens;
    std::vector<Tensor> k_cache;  // Shape: [layer_nums, q_heads, max_cache_length, kv_dims]
    std::vector<Tensor> v_cache;  // Shape: [layer_nums, q_heads, max_cache_length, kv_dims]

    static InitParams make_default(const Qwen3Config& cfg, const std::filesystem::path& path);
  };

  explicit GenerationState(InitParams&& params);

  static ptr create_or_recover(const Qwen3Config& cfg, const std::filesystem::path& path);
  static ptr recover(const Qwen3Config& cfg, const std::filesystem::path& path);
  static ptr create(const Qwen3Config& cfg, const std::filesystem::path& path);

  void start_prefill(const Tensor& token_ids);
  void start_decode(const Tensor& token_id);
  void append_output_token(const Tensor& token);

  void save() const;
  void sync_cache();

  void update_kv(int layer_idx, int offset, int count, const Tensor &k, const Tensor &v);
  void update_h(int layer_idx, int offset, int count, const Tensor &h);
  [[nodiscard]] std::array<Tensor, 2>get_kv(int layer_idx);
  [[nodiscard]] std::array<Tensor, 2>get_kv(int layer_idx, int offset, int count);
  void clear();

 private:
  std::filesystem::path path_;

  int max_length_;
  int layer_nums_;
  int q_heads_;
  int kv_heads_;
  int kv_dim_;
  int hidden_size_;
  int num_output_tokens_;

  std::vector<int64_t> input_tokens_;
  Tensor output_tokens_;
  std::vector<Tensor> k_cache_;  // Shape: [layer_nums, q_heads, max_cache_length, kv_dims]
  std::vector<Tensor> v_cache_;  // Shape: [layer_nums, q_heads, max_cache_length, kv_dims]

  std::unordered_map<std::pair<int, int>, bool, PairHash> dirty;
  std::queue<std::pair<int, int>> dirty_queue;
};

}  // namespace mllm::models::qwen3_i
