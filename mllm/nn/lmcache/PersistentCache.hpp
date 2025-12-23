// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/nn/lmcache/StaticCache.hpp"
#include "mllm/core/Tensor.hpp"
#include <filesystem>
#include <memory>
#include <vector>

namespace mllm::nn {

/**
 * @brief A persistent KV cache that can be saved to and restored from disk.
 *
 * This cache uses memory-mapped files (mmap) for efficient persistence.
 * The KV data is stored in a binary file (kv.bin) and metadata including
 * sequence counts and pending tokens is stored in JSON (metadata.json).
 *
 * Directory structure:
 *   working_dir/
 *     metadata.json   - Configuration, sequence counts, and pending tokens
 *     kv.bin          - Binary KV cache data (mmap-backed)
 *
 * Usage:
 *   // Create new cache
 *   PersistentCache cache(working_dir, max_cache_length, layer_nums, q_heads,
 *                         kv_heads, kv_dims, k_dtype, v_dtype, device_type);
 *   // ... use the cache ...
 *   cache.sync();  // Persist to disk
 *
 *   // Later: recover from disk
 *   auto recovered = PersistentCache::recover("./kvcache");
 */
class PersistentCache : public AbstractStaticCache {
 public:
  using ptr_t = std::shared_ptr<PersistentCache>;

  /**
   * @brief Creates a new PersistentCache.
   * @throws std::runtime_error if initialization fails.
   */
  PersistentCache(const std::filesystem::path& working_dir, 
                  int32_t max_cache_length,
                  int32_t layer_nums, 
                  int32_t q_heads, 
                  int32_t kv_heads, 
                  int32_t kv_dims,
                  DataTypes k_dtype, 
                  DataTypes v_dtype, 
                  DeviceTypes device_type = kCPU);

  /**
   * @brief Recovers a PersistentCache from disk.
   * @return Shared pointer to recovered cache, or nullptr on failure.
   */
  [[nodiscard]] static ptr_t recover(const std::filesystem::path& working_dir);

  ~PersistentCache();

  // Non-copyable, non-movable (mmap resources)
  PersistentCache(const PersistentCache&) = delete;
  PersistentCache& operator=(const PersistentCache&) = delete;
  PersistentCache(PersistentCache&&) = delete;
  PersistentCache& operator=(PersistentCache&&) = delete;

  /**
   * @brief Synchronizes the cache to disk.
   * @return true on success, false on failure.
   */
  [[nodiscard]] bool sync();

  // AbstractStaticCache interface
  [[nodiscard]] int32_t getCurrentSeqCnt(int32_t layer_idx) const override;
  [[nodiscard]] int32_t getLayerNums() const override { return layer_nums_; }
  void setCurrentSeqCnt(int32_t seq) override;
  void clearCache() override;
  std::array<Tensor, 2> updateKVCache(int32_t layer_idx, Tensor k, Tensor v) override;

  /**
   * @brief Get K and V cache views for a specific layer.
   */
  std::array<Tensor, 2> getKVCache(int32_t layer_idx);

  /**
   * @brief Mark cache as dirty (call after directly modifying cache data).
   */
  void markDirty() noexcept { is_dirty_ = true; }

  [[nodiscard]] bool isDirty() const noexcept { return is_dirty_; }
  [[nodiscard]] const std::filesystem::path& workingDir() const noexcept { return working_dir_; }

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

  // Cache tensors (backed by mmap)
  Tensor k_cache_;  // Shape: [layer_nums, q_heads, max_cache_length, kv_dims]
  Tensor v_cache_;  // Shape: [layer_nums, q_heads, max_cache_length, kv_dims]
  
  // Sequence tracking
  std::vector<int32_t> seq_cnt_;        // Current sequence count per layer
  std::vector<int32_t> saved_seq_cnt_;  // Last synced sequence count per layer
  bool is_dirty_ = false;
};

}  // namespace mllm::nn
