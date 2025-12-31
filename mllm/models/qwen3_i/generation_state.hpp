#pragma once

#include <optional>
#include "mllm/core/Tensor.hpp"
#include "mllm/mllm.hpp"
#include "mllm/models/qwen3/configuration_qwen3.hpp"

namespace mllm::models::qwen3_i {

class GenerationState {
 public:
  using ptr = std::shared_ptr<GenerationState>;
  using Qwen3Config = models::qwen3::Qwen3Config;

  GenerationState(const std::filesystem::path& path, int max_length, int layer_nums, int q_heads, int kv_heads, int kv_dim);

  static ptr create_or_recover(const Qwen3Config& cfg, const std::filesystem::path& path);
  static ptr recover(const Qwen3Config& cfg, const std::filesystem::path& path);
  static ptr create(const Qwen3Config& cfg, const std::filesystem::path& path);

  void save() const;
  void sync_cache();
  void update_kv(int layer_idx, int offset, int count, const Tensor &k, const Tensor &v);
  void update_h(int layer_idx, int offset, int count, const Tensor &h);
  [[nodiscard]] std::optional<std::array<Tensor, 2>>get_kv(int layer_idx, int offset, int count);
  void clear();

 private:
  nn::StaticCache kv_cache_;
  std::filesystem::path path_;
};

}  // namespace mllm::models::qwen3_i
