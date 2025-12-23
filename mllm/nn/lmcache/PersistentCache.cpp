// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/nn/lmcache/PersistentCache.hpp"

#include <cstring>
#include <fstream>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

#include <nlohmann/json.hpp>

#include "mllm/core/DataTypes.hpp"
#include "mllm/core/TensorStorage.hpp"
#include "mllm/utils/Common.hpp"
#include "mllm/utils/Log.hpp"
#include "mllm/utils/UnsafeMacros.hpp"

namespace mllm::nn {

// ============================================================================
// Constructor / Destructor
// ============================================================================

PersistentCache::PersistentCache(
    const std::filesystem::path& working_dir, 
    int32_t max_cache_length,
    int32_t layer_nums, 
    int32_t q_heads, 
    int32_t kv_heads, 
    int32_t kv_dims,
    DataTypes k_dtype, 
    DataTypes v_dtype, 
    DeviceTypes device_type)
    : working_dir_(working_dir),
      device_type_(device_type),
      k_dtype_(k_dtype),
      v_dtype_(v_dtype),
      max_cache_length_(max_cache_length),
      layer_nums_(layer_nums),
      q_heads_(q_heads),
      kv_heads_(kv_heads),
      kv_dims_(kv_dims),
      seq_cnt_(layer_nums, 0),
      saved_seq_cnt_(layer_nums, 0),
      is_dirty_(true) {
  
  MLLM_RT_ASSERT(device_type_ == kCPU);

  // Create working directory
  std::error_code ec;
  std::filesystem::create_directories(working_dir_, ec);
  if (ec) {
    MLLM_ERROR_EXIT(ExitCode::kIOError, "Failed to create PersistentCache directory: {}", working_dir_.string());
  }

  // Initialize memory-mapped file
  if (!initMmap()) {
    MLLM_ERROR_EXIT(ExitCode::kIOError, "Failed to initialize mmap for PersistentCache: {}", working_dir_.string());
  }
  initTensorsFromMmap();
}

PersistentCache::~PersistentCache() {
  // Clear tensor references before unmapping
  k_cache_ = Tensor::nil();
  v_cache_ = Tensor::nil();

  if (mapped_ptr_ && mapped_ptr_ != MAP_FAILED) {
    munmap(mapped_ptr_, map_size_);
  }
  if (fd_ >= 0) {
    close(fd_);
  }
}

// ============================================================================
// Static Factory
// ============================================================================

PersistentCache::ptr_t PersistentCache::recover(const std::filesystem::path& working_dir) {
  const auto metadata_path = working_dir / "metadata.json";
  std::ifstream file(metadata_path);
  if (!file) return nullptr;

  try {
    nlohmann::json j;
    file >> j;

    // Read cache configuration
    const auto max_cache_length = j.at("max_cache_length").get<int32_t>();
    const auto layer_nums = j.at("layer_nums").get<int32_t>();
    const auto q_heads = j.at("q_heads").get<int32_t>();
    const auto kv_heads = j.at("kv_heads").get<int32_t>();
    const auto kv_dims = j.at("kv_dims").get<int32_t>();
    const auto k_dtype = static_cast<DataTypes>(j.at("k_dtype").get<int>());
    const auto v_dtype = static_cast<DataTypes>(j.at("v_dtype").get<int>());

    // Create cache instance
    auto cache = std::make_shared<PersistentCache>(
        working_dir, max_cache_length, layer_nums, q_heads, kv_heads,
        kv_dims, k_dtype, v_dtype, kCPU);

    // Mark as not dirty since we're recovering existing data
    cache->is_dirty_ = false;

    // Restore sequence counts
    if (j.contains("saved_seq_cnts")) {
      const auto& cnts = j["saved_seq_cnts"];
      for (int32_t i = 0; i < layer_nums && i < static_cast<int32_t>(cnts.size()); ++i) {
        cache->seq_cnt_[i] = cache->saved_seq_cnt_[i] = cnts[i].get<int32_t>();
      }
    }

    return cache;
  } catch (const std::exception& e) {
    MLLM_ERROR_EXIT(ExitCode::kIOError, "Failed to recover PersistentCache from {}: {}", 
                    working_dir.string(), e.what());
    return nullptr;
  }
}

// ============================================================================
// Memory Mapping
// ============================================================================

size_t PersistentCache::cacheBytes() const {
  return static_cast<size_t>(layer_nums_) * q_heads_ * max_cache_length_ * kv_dims_ *
         bytesOfType(k_dtype_) / lanesOfType(k_dtype_);
}

bool PersistentCache::initMmap() {
  map_size_ = cacheBytes() * 2;  // K + V

  fd_ = open(cachePath().c_str(), O_RDWR | O_CREAT, 0644);
  if (fd_ < 0) return false;

  if (ftruncate(fd_, static_cast<off_t>(map_size_)) < 0) {
    close(fd_);
    fd_ = -1;
    return false;
  }

  mapped_ptr_ = mmap(nullptr, map_size_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
  if (mapped_ptr_ == MAP_FAILED) {
    close(fd_);
    fd_ = -1;
    mapped_ptr_ = nullptr;
    return false;
  }

  return true;
}

void PersistentCache::initTensorsFromMmap() {
  const std::vector<int32_t> shape = {layer_nums_, q_heads_, max_cache_length_, kv_dims_};
  const size_t half_size = cacheBytes();

  k_cache_ = Tensor::empty(shape, k_dtype_, device_type_);
  k_cache_.impl()->storage()->ptr_ = static_cast<char*>(mapped_ptr_);
  k_cache_.impl()->storage()->mem_type_ = kManual;

  v_cache_ = Tensor::empty(shape, v_dtype_, device_type_);
  v_cache_.impl()->storage()->ptr_ = static_cast<char*>(mapped_ptr_) + half_size;
  v_cache_.impl()->storage()->mem_type_ = kManual;
}

// ============================================================================
// Persistence
// ============================================================================

bool PersistentCache::saveMetadata() const {
  nlohmann::json j = {
      {"max_cache_length", max_cache_length_},
      {"layer_nums", layer_nums_},
      {"q_heads", q_heads_},
      {"kv_heads", kv_heads_},
      {"kv_dims", kv_dims_},
      {"k_dtype", static_cast<int>(k_dtype_)},
      {"v_dtype", static_cast<int>(v_dtype_)},
      {"saved_seq_cnts", seq_cnt_},
  };

  std::ofstream file(metadataPath());
  if (!file) return false;
  file << j.dump(2);
  return file.good();
}

bool PersistentCache::sync() {
  if (!is_dirty_) return true;

  // msync writes back dirty pages asynchronously
  if (msync(mapped_ptr_, map_size_, MS_ASYNC) != 0) return false;
  if (!saveMetadata()) return false;

  saved_seq_cnt_ = seq_cnt_;
  is_dirty_ = false;
  return true;
}

// ============================================================================
// Cache Operations
// ============================================================================

int32_t PersistentCache::getCurrentSeqCnt(int32_t layer_idx) const {
  return seq_cnt_[layer_idx];
}

void PersistentCache::setCurrentSeqCnt(int32_t seq) {
  std::fill(seq_cnt_.begin(), seq_cnt_.end(), seq);
  is_dirty_ = true;
}

void PersistentCache::clearCache() {
  std::fill(seq_cnt_.begin(), seq_cnt_.end(), 0);
  std::fill(saved_seq_cnt_.begin(), saved_seq_cnt_.end(), 0);
  is_dirty_ = true;
}

__MLLM_UNSAFE_OPT_BEGIN_O3
std::array<Tensor, 2> PersistentCache::updateKVCache(int32_t layer_idx, Tensor k, Tensor v) {
  MLLM_RT_ASSERT_EQ(k.shape()[1], kv_heads_);
  MLLM_RT_ASSERT_EQ(v.shape()[1], kv_heads_);

  const auto seq_len = k.shape()[2];
  const auto repeat = q_heads_ / kv_heads_;
  const auto cur_seq = seq_cnt_[layer_idx];
  const size_t copy_bytes = seq_len * kv_dims_ * bytesOfType(k_dtype_) / lanesOfType(k_dtype_);

  // Copy K and V to cache with GQA expansion
  for (int h = 0; h < kv_heads_; ++h) {
    for (int r = 0; r < repeat; ++r) {
      const int dst_h = h * repeat + r;
      std::memcpy(k_cache_.offsettedPtr<char>({layer_idx, dst_h, cur_seq, 0}),
                  k.offsettedPtr<char>({0, h, 0, 0}), copy_bytes);
      std::memcpy(v_cache_.offsettedPtr<char>({layer_idx, dst_h, cur_seq, 0}),
                  v.offsettedPtr<char>({0, h, 0, 0}), copy_bytes);
    }
  }

  seq_cnt_[layer_idx] += seq_len;
  is_dirty_ = true;

  const auto new_seq = seq_cnt_[layer_idx];
  return {
      k_cache_[{layer_idx, kAll, {kAll, new_seq}, kAll}],
      v_cache_[{layer_idx, kAll, {kAll, new_seq}, kAll}],
  };
}
__MLLM_UNSAFE_OPT_END

std::array<Tensor, 2> PersistentCache::getKVCache(int32_t layer_idx) {
  const auto cur_seq = seq_cnt_[layer_idx];
  return {
      k_cache_[{layer_idx, kAll, {kAll, cur_seq}, kAll}],
      v_cache_[{layer_idx, kAll, {kAll, cur_seq}, kAll}],
  };
}

}  // namespace mllm::nn
