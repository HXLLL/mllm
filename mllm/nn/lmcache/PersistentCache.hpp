// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/nn/lmcache/StaticCache.hpp"
#include "mllm/core/Tensor.hpp"
#include <string>

namespace mllm::nn {

class PersistentCache : public AbstractStaticCache {
 public:
  PersistentCache(const std::string& cache_file_path, int32_t max_cache_length, int32_t layer_nums,
                  int32_t q_heads, int32_t kv_heads, int32_t kv_dims, DataTypes k_dtype, DataTypes v_dtype,
                  DeviceTypes device_type = kCPU);

  ~PersistentCache();

  void setCurrentSeqCnt(int32_t seq) override;

  void clearCache() override;

  [[nodiscard]] int32_t getCurrentSeqCnt(int32_t layer_idx) const override;

  [[nodiscard]] int32_t getLayerNums() const override { return layer_nums_; }

  std::array<Tensor, 2> updateKVCache(int32_t layer_idx, Tensor k, Tensor v) override;

  [[nodiscard]] inline Tensor getKCacheBuffer(int32_t layer_idx) const { return k_cache_[layer_idx]; }

  [[nodiscard]] inline Tensor getVCacheBuffer(int32_t layer_idx) const { return v_cache_[layer_idx]; }

 private:
  void initializeFile();
  void mapFile();
  void unmapFile();
  [[nodiscard]] size_t calculateFileSize() const;
  [[nodiscard]] void* getCacheDataPtr(int32_t layer_idx, bool is_k) const;

  std::string cache_file_path_;
  DeviceTypes device_type_;
  DataTypes k_dtype_;
  DataTypes v_dtype_;
  int32_t max_cache_length_;
  int32_t layer_nums_;
  int32_t q_heads_;
  int32_t kv_heads_;
  int32_t kv_dims_;

  std::vector<Tensor> k_cache_;
  std::vector<Tensor> v_cache_;
  std::vector<int32_t> current_seq_cnt_;

  void* mapped_memory_ = nullptr;
  size_t mapped_size_ = 0;
  int fd_ = -1;
  bool file_created_ = false;

  // File format:
  // Header: [layer_nums(4), max_cache_length(4), q_heads(4), kv_heads(4), kv_dims(4), k_dtype(4), v_dtype(4), reserved(4)]
  // For each layer:
  //   [current_seq_cnt(4)]
  //   [k_cache_data]
  //   [v_cache_data]
  static constexpr size_t HEADER_SIZE = 32;  // 8 * 4 bytes
  [[nodiscard]] size_t getLayerOffset(int32_t layer_idx) const;
  [[nodiscard]] size_t getKCacheOffset(int32_t layer_idx) const;
  [[nodiscard]] size_t getVCacheOffset(int32_t layer_idx) const;
  [[nodiscard]] size_t getSingleCacheSize() const;
};

}  // namespace mllm::nn

