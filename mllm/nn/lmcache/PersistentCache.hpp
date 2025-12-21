// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/nn/lmcache/StaticCache.hpp"
#include "mllm/core/Tensor.hpp"
#include <filesystem>
#include <memory>

namespace mllm::nn {

/* Options for creating a PersistentCache. */
struct PersistentCacheOptions {
  std::filesystem::path working_dir = "./kvcache";
  int32_t max_cache_length = 1024;
  int32_t layer_nums = 1;
  int32_t q_heads = 1;
  int32_t kv_heads = 1;
  int32_t kv_dims = 1;
  DataTypes k_dtype = DataTypes::kFloat16;
  DataTypes v_dtype = DataTypes::kFloat16;
  DeviceTypes device_type = DeviceTypes::kCPU;
};

/* A persistent KV cache that can be saved to and restored from disk.
 *
 * Directory structure:
 *   working_dir/
 *     metadata.json   - Configuration and state information
 *     kv.bin          - Binary KV cache data
 *
 * Usage:
 *   auto cache = PersistentCache::create(options);
 *   // ... use the cache ...
 *   cache->sync();
 *   // Later: auto recovered = PersistentCache::recover("./kvcache");
 */
class PersistentCache : public AbstractStaticCache {
 public:
  using ptr_t = std::shared_ptr<PersistentCache>;

  /* Creates a new PersistentCache in memory. */
  [[nodiscard]] static ptr_t create(const PersistentCacheOptions& options);

  /* Recovers a PersistentCache from disk. Returns nullptr on failure. */
  [[nodiscard]] static ptr_t recover(const std::filesystem::path& working_dir);

  ~PersistentCache();

  // Non-copyable, non-movable (mmap resources)
  PersistentCache(const PersistentCache&) = delete;
  PersistentCache& operator=(const PersistentCache&) = delete;
  PersistentCache(PersistentCache&&) = delete;
  PersistentCache& operator=(PersistentCache&&) = delete;

  /* Synchronizes the cache to disk. Returns true on success. */
  [[nodiscard]] bool sync();

  // AbstractStaticCache interface
  [[nodiscard]] int32_t getCurrentSeqCnt(int32_t layer_idx) const override;
  [[nodiscard]] int32_t getLayerNums() const override { return layer_nums_; }
  void setCurrentSeqCnt(int32_t seq) override;
  void clearCache() override;
  std::array<Tensor, 2> updateKVCache(int32_t layer_idx, Tensor k, Tensor v) override;

  /* Get current KV cache view without modifying seq_cnt. */
  [[nodiscard]] std::array<Tensor, 2> getKVCache(int32_t layer_idx) const;

  /* Mark cache as dirty (call after directly modifying cache data). */
  void markDirty() noexcept { is_dirty_ = true; }

  [[nodiscard]] bool isDirty() const noexcept { return is_dirty_; }
  [[nodiscard]] const std::filesystem::path& workingDir() const noexcept { return working_dir_; }

  explicit PersistentCache(const PersistentCacheOptions& options);

 private:
  [[nodiscard]] bool initMmap();
  void initTensorsFromMmap();
  [[nodiscard]] bool saveMetadata() const;

  [[nodiscard]] std::filesystem::path metadataPath() const { return working_dir_ / "metadata.json"; }
  [[nodiscard]] std::filesystem::path cachePath() const { return working_dir_ / "kv.bin"; }
  [[nodiscard]] size_t cacheBytes() const;

  // Configuration
  std::filesystem::path working_dir_;
  DeviceTypes device_type_ = DeviceTypes::kCPU;
  DataTypes k_dtype_ = DataTypes::kFloat16;
  DataTypes v_dtype_ = DataTypes::kFloat16;
  int32_t max_cache_length_ = 0;
  int32_t layer_nums_ = 0;
  int32_t q_heads_ = 0;
  int32_t kv_heads_ = 0;
  int32_t kv_dims_ = 0;

  // mmap state
  int fd_ = -1;
  void* mapped_ptr_ = nullptr;
  size_t map_size_ = 0;

  // Cache state
  Tensor k_cache_;  // Shape: [layer_nums, q_heads, max_cache_length, kv_dims]
  Tensor v_cache_;  // Shape: [layer_nums, q_heads, max_cache_length, kv_dims]
  std::vector<int32_t> seq_cnt_;        // Current sequence count per layer
  std::vector<int32_t> saved_seq_cnt_;  // Persisted sequence count per layer
  bool is_dirty_ = false;
};

}  // namespace mllm::nn
