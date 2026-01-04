#pragma once

#include <queue>
#include <unordered_map>
#include <optional>
#include <unordered_map>
#include "mllm/core/Tensor.hpp"
#include "mllm/mllm.hpp"
#include "mllm/models/qwen3/configuration_qwen3.hpp"

namespace mllm::models::qwen3_i {

class KVCache {
 public:
  void checkpoint() const;
  void update_kv(int layer_idx, int offset, int count, const Tensor &k, const Tensor &v);
  [[nodiscard]] std::array<Tensor, 2> get_kv(int layer_idx);
  [[nodiscard]] std::array<Tensor, 2> get_kv(int layer_idx, int offset, int count);
 private:
  DeviceTypes device_type_;
  DataTypes k_dtype_;
  DataTypes v_dtype_;
  int32_t max_cache_length_;
  int32_t layer_nums_;
  int32_t q_heads_;
  int32_t kv_heads_;
  int32_t kv_dims_;

  Tensor k_cache_;  // Shape: [layer_nums, q_heads, max_cache_length, kv_dims]
  Tensor v_cache_;  // Shape: [layer_nums, q_heads, max_cache_length, kv_dims]

  std::unordered_map<std::pair<int, int>, bool> dirty;
  std::queue<std::pair<int, int>> dirty_queue;
};

class GenerationState {
 public:
  using ptr = std::shared_ptr<GenerationState>;
  using Qwen3Config = models::qwen3::Qwen3Config;

  GenerationState(const std::filesystem::path& path, int max_length, int layer_nums, int q_heads, int kv_heads, int kv_dim);

  static ptr create_or_recover(const Qwen3Config& cfg, const std::filesystem::path& path);
  static ptr recover(const Qwen3Config& cfg, const std::filesystem::path& path);
  static ptr create(const Qwen3Config& cfg, const std::filesystem::path& path);

  void start_generation(const Tensor& token_ids);

  void save() const;
  void checkpoint() const;
  void sync_cache();
  void update_kv(int layer_idx, int offset, int count, const Tensor &k, const Tensor &v);
  void update_h(int layer_idx, int offset, int count, const Tensor &h);
  [[nodiscard]] std::optional<std::array<Tensor, 2>>get_kv(int layer_idx, int offset, int count);
  void clear();

 private:
  Tensor input_tokens_;
  nn::StaticCache kv_cache_;
  std::filesystem::path path_;
};

}  // namespace mllm::models::qwen3_i
