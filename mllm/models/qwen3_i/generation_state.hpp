#pragma once

#include <optional>
#include "mllm/core/Tensor.hpp"
#include "mllm/mllm.hpp"
#include "mllm/nn/lmcache/PersistentCache.hpp"

namespace mllm::models::qwen3_i {

class GenerationState {
 public:
  using ptr = std::shared_ptr<GenerationState>;

  GenerationState();

  static ptr create_or_recover(const std::filesystem::path& path);
  static ptr recover(const std::filesystem::path& path);
  static ptr create(const std::filesystem::path& path);

  void save(const std::filesystem::path& path) const;

  void sync_cache();
  void update_kv(int layer_idx, int token_idx, const Tensor &k, const Tensor &v);
  [[nodiscard]] std::optional<std::array<Tensor, 2>>get_kv(int layer_idx, int token_idx) const;
  void clear();

 private:
  nn::PersistentCache* kv_cache;
};

}  // namespace mllm::models::qwen3_i
