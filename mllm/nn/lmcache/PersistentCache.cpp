// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/nn/lmcache/PersistentCache.hpp"
#include "mllm/core/TensorStorage.hpp"
#include "mllm/core/TensorViewImpl.hpp"
#include "mllm/core/DataTypes.hpp"
#include "mllm/utils/Common.hpp"
#include "mllm/utils/UnsafeMacros.hpp"
#include <cstring>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fstream>

namespace mllm::nn {

PersistentCache::PersistentCache(const std::string& cache_file_path, int32_t max_cache_length, int32_t layer_nums,
                                 int32_t q_heads, int32_t kv_heads, int32_t kv_dims, DataTypes k_dtype, DataTypes v_dtype,
                                 DeviceTypes device_type)
    : cache_file_path_(cache_file_path),
      device_type_(device_type),
      k_dtype_(k_dtype),
      v_dtype_(v_dtype),
      max_cache_length_(max_cache_length),
      layer_nums_(layer_nums),
      q_heads_(q_heads),
      kv_heads_(kv_heads),
      kv_dims_(kv_dims) {
  // Check if file exists
  std::ifstream test_file(cache_file_path_);
  bool file_exists = test_file.good();
  test_file.close();

  if (!file_exists) {
    initializeFile();
    file_created_ = true;
  } else {
    // Verify file header matches our parameters
    std::ifstream file(cache_file_path_, std::ios::binary);
    if (!file) {
      MLLM_ERROR_EXIT(ExitCode::kIOError, "Failed to open cache file: {}", cache_file_path_);
    }

    int32_t file_layer_nums, file_max_cache_length, file_q_heads, file_kv_heads, file_kv_dims;
    int32_t file_k_dtype, file_v_dtype, reserved;
    file.read(reinterpret_cast<char*>(&file_layer_nums), sizeof(int32_t));
    file.read(reinterpret_cast<char*>(&file_max_cache_length), sizeof(int32_t));
    file.read(reinterpret_cast<char*>(&file_q_heads), sizeof(int32_t));
    file.read(reinterpret_cast<char*>(&file_kv_heads), sizeof(int32_t));
    file.read(reinterpret_cast<char*>(&file_kv_dims), sizeof(int32_t));
    file.read(reinterpret_cast<char*>(&file_k_dtype), sizeof(int32_t));
    file.read(reinterpret_cast<char*>(&file_v_dtype), sizeof(int32_t));
    file.read(reinterpret_cast<char*>(&reserved), sizeof(int32_t));
    file.close();

    if (file_layer_nums != layer_nums || file_max_cache_length != max_cache_length || file_q_heads != q_heads ||
        file_kv_heads != kv_heads || file_kv_dims != kv_dims || file_k_dtype != static_cast<int32_t>(k_dtype) ||
        file_v_dtype != static_cast<int32_t>(v_dtype)) {
      MLLM_ERROR_EXIT(ExitCode::kIOError,
                      "Cache file parameters mismatch. Expected: layers={}, max_len={}, q_heads={}, kv_heads={}, "
                      "kv_dims={}, k_dtype={}, v_dtype={}. "
                      "Got: layers={}, max_len={}, q_heads={}, kv_heads={}, kv_dims={}, k_dtype={}, v_dtype={}",
                      layer_nums, max_cache_length, q_heads, kv_heads, kv_dims, static_cast<int32_t>(k_dtype),
                      static_cast<int32_t>(v_dtype), file_layer_nums, file_max_cache_length, file_q_heads,
                      file_kv_heads, file_kv_dims, file_k_dtype, file_v_dtype);
    }
  }

  mapFile();

  // Initialize tensors from mapped memory
  k_cache_.reserve(layer_nums_);
  v_cache_.reserve(layer_nums_);
  current_seq_cnt_.resize(layer_nums_);

  for (int32_t i = 0; i < layer_nums_; ++i) {
    // Read current_seq_cnt from file
    int32_t* seq_cnt_ptr = reinterpret_cast<int32_t*>(static_cast<char*>(mapped_memory_) + getLayerOffset(i));
    current_seq_cnt_[i] = *seq_cnt_ptr;

    // Create tensor views for k_cache and v_cache
    // Shape: [1, q_heads, max_cache_length, kv_dims]
    std::vector<int32_t> cache_shape = {1, q_heads_, max_cache_length_, kv_dims_};
    size_t cache_size = getSingleCacheSize();

    // K cache
    auto k_storage = TensorStorage::create(cache_shape, k_dtype_, device_type_);
    k_storage->mem_type_ = TensorMemTypes::kManual;
    k_storage->ptr_ = getCacheDataPtr(i, true);
    auto k_view = TensorViewImpl::create(cache_shape, k_storage);
    k_cache_.emplace_back(k_view);

    // V cache
    auto v_storage = TensorStorage::create(cache_shape, v_dtype_, device_type_);
    v_storage->mem_type_ = TensorMemTypes::kManual;
    v_storage->ptr_ = getCacheDataPtr(i, false);
    auto v_view = TensorViewImpl::create(cache_shape, v_storage);
    v_cache_.emplace_back(v_view);
  }
}

PersistentCache::~PersistentCache() {
  unmapFile();
}

void PersistentCache::initializeFile() {
  size_t file_size = calculateFileSize();

  // Create and initialize file
  std::ofstream file(cache_file_path_, std::ios::binary | std::ios::trunc);
  if (!file) {
    MLLM_ERROR_EXIT(ExitCode::kIOError, "Failed to create cache file: {}", cache_file_path_);
  }

  // Write header
  file.write(reinterpret_cast<const char*>(&layer_nums_), sizeof(int32_t));
  file.write(reinterpret_cast<const char*>(&max_cache_length_), sizeof(int32_t));
  file.write(reinterpret_cast<const char*>(&q_heads_), sizeof(int32_t));
  file.write(reinterpret_cast<const char*>(&kv_heads_), sizeof(int32_t));
  file.write(reinterpret_cast<const char*>(&kv_dims_), sizeof(int32_t));
  file.write(reinterpret_cast<const char*>(&k_dtype_), sizeof(int32_t));
  file.write(reinterpret_cast<const char*>(&v_dtype_), sizeof(int32_t));
  int32_t reserved = 0;
  file.write(reinterpret_cast<const char*>(&reserved), sizeof(int32_t));

  // Initialize each layer: write current_seq_cnt (0) and zero-initialize cache data
  size_t k_cache_size = static_cast<size_t>(q_heads_) * max_cache_length_ * kv_dims_ *
                         (bytesOfType(k_dtype_) / lanesOfType(k_dtype_));
  size_t v_cache_size = static_cast<size_t>(q_heads_) * max_cache_length_ * kv_dims_ *
                         (bytesOfType(v_dtype_) / lanesOfType(v_dtype_));

  for (int32_t i = 0; i < layer_nums_; ++i) {
    int32_t seq_cnt = 0;
    file.write(reinterpret_cast<const char*>(&seq_cnt), sizeof(int32_t));

    // Zero-initialize k_cache
    std::vector<char> k_zeros(k_cache_size, 0);
    file.write(k_zeros.data(), k_cache_size);

    // Zero-initialize v_cache
    std::vector<char> v_zeros(v_cache_size, 0);
    file.write(v_zeros.data(), v_cache_size);
  }

  file.close();

  // Truncate file to exact size (in case of any padding)
  if (truncate(cache_file_path_.c_str(), static_cast<off_t>(file_size)) != 0) {
    MLLM_ERROR_EXIT(ExitCode::kIOError, "Failed to truncate cache file to size: {}", file_size);
  }
}

void PersistentCache::mapFile() {
  size_t file_size = calculateFileSize();

  fd_ = open(cache_file_path_.c_str(), O_RDWR);
  if (fd_ == -1) {
    MLLM_ERROR_EXIT(ExitCode::kIOError, "Failed to open cache file for mapping: {}", cache_file_path_);
  }

  mapped_memory_ = mmap(nullptr, file_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
  if (mapped_memory_ == MAP_FAILED) {
    close(fd_);
    fd_ = -1;
    MLLM_ERROR_EXIT(ExitCode::kIOError, "Failed to map cache file to memory: {}", cache_file_path_);
  }

  mapped_size_ = file_size;
}

void PersistentCache::unmapFile() {
  if (mapped_memory_ != nullptr && mapped_memory_ != MAP_FAILED) {
    msync(mapped_memory_, mapped_size_, MS_SYNC);
    munmap(mapped_memory_, mapped_size_);
    mapped_memory_ = nullptr;
    mapped_size_ = 0;
  }
  if (fd_ != -1) {
    close(fd_);
    fd_ = -1;
  }
}

size_t PersistentCache::calculateFileSize() const {
  // Calculate actual sizes for k and v caches
  size_t k_cache_size = static_cast<size_t>(q_heads_) * max_cache_length_ * kv_dims_ *
                        (bytesOfType(k_dtype_) / lanesOfType(k_dtype_));
  size_t v_cache_size = static_cast<size_t>(q_heads_) * max_cache_length_ * kv_dims_ *
                        (bytesOfType(v_dtype_) / lanesOfType(v_dtype_));
  return HEADER_SIZE + layer_nums_ * (sizeof(int32_t) + k_cache_size + v_cache_size);
}

size_t PersistentCache::getSingleCacheSize() const {
  // Size for one cache tensor: [1, q_heads, max_cache_length, kv_dims]
  // Note: k and v may have different dtypes, but we assume they have the same size
  // for simplicity. If needed, this can be extended to handle different sizes.
  size_t k_cache_size = static_cast<size_t>(q_heads_) * max_cache_length_ * kv_dims_ *
                        (bytesOfType(k_dtype_) / lanesOfType(k_dtype_));
  size_t v_cache_size = static_cast<size_t>(q_heads_) * max_cache_length_ * kv_dims_ *
                        (bytesOfType(v_dtype_) / lanesOfType(v_dtype_));
  // Return the maximum size to ensure we have enough space
  return (k_cache_size > v_cache_size) ? k_cache_size : v_cache_size;
}

size_t PersistentCache::getLayerOffset(int32_t layer_idx) const {
  size_t k_cache_size = static_cast<size_t>(q_heads_) * max_cache_length_ * kv_dims_ *
                         (bytesOfType(k_dtype_) / lanesOfType(k_dtype_));
  size_t v_cache_size = static_cast<size_t>(q_heads_) * max_cache_length_ * kv_dims_ *
                         (bytesOfType(v_dtype_) / lanesOfType(v_dtype_));
  return HEADER_SIZE + layer_idx * (sizeof(int32_t) + k_cache_size + v_cache_size);
}

size_t PersistentCache::getKCacheOffset(int32_t layer_idx) const {
  return getLayerOffset(layer_idx) + sizeof(int32_t);
}

size_t PersistentCache::getVCacheOffset(int32_t layer_idx) const {
  size_t k_cache_size = static_cast<size_t>(q_heads_) * max_cache_length_ * kv_dims_ *
                         (bytesOfType(k_dtype_) / lanesOfType(k_dtype_));
  return getKCacheOffset(layer_idx) + k_cache_size;
}

void* PersistentCache::getCacheDataPtr(int32_t layer_idx, bool is_k) const {
  size_t offset = is_k ? getKCacheOffset(layer_idx) : getVCacheOffset(layer_idx);
  return static_cast<char*>(mapped_memory_) + offset;
}

void PersistentCache::setCurrentSeqCnt(int32_t seq) {
  for (int32_t layer_idx = 0; layer_idx < layer_nums_; ++layer_idx) {
    current_seq_cnt_[layer_idx] = seq;
    // Write to mapped memory (will be synced to disk)
    int32_t* seq_cnt_ptr = reinterpret_cast<int32_t*>(static_cast<char*>(mapped_memory_) + getLayerOffset(layer_idx));
    *seq_cnt_ptr = seq;
  }
}

void PersistentCache::clearCache() {
  for (int32_t layer_idx = 0; layer_idx < layer_nums_; ++layer_idx) {
    current_seq_cnt_[layer_idx] = 0;
    // Write to mapped memory
    int32_t* seq_cnt_ptr = reinterpret_cast<int32_t*>(static_cast<char*>(mapped_memory_) + getLayerOffset(layer_idx));
    *seq_cnt_ptr = 0;

    // Zero out cache data
    void* k_ptr = getCacheDataPtr(layer_idx, true);
    void* v_ptr = getCacheDataPtr(layer_idx, false);
    size_t k_cache_size = static_cast<size_t>(q_heads_) * max_cache_length_ * kv_dims_ *
                           (bytesOfType(k_dtype_) / lanesOfType(k_dtype_));
    size_t v_cache_size = static_cast<size_t>(q_heads_) * max_cache_length_ * kv_dims_ *
                           (bytesOfType(v_dtype_) / lanesOfType(v_dtype_));
    std::memset(k_ptr, 0, k_cache_size);
    std::memset(v_ptr, 0, v_cache_size);
  }
}

int32_t PersistentCache::getCurrentSeqCnt(int32_t layer_idx) const {
  return current_seq_cnt_[layer_idx];
}

__MLLM_UNSAFE_OPT_BEGIN_O3
std::array<Tensor, 2> PersistentCache::updateKVCache(int32_t layer_idx, Tensor k, Tensor v) {
  // Eager mode only (non-FA2)
  // The input should be [B, H, S, D]
  MLLM_RT_ASSERT_EQ(k.shape()[1], kv_heads_);
  MLLM_RT_ASSERT_EQ(v.shape()[1], kv_heads_);

  auto inputs_seq_len = k.shape()[2];

  auto repeat_times = q_heads_ / kv_heads_;

  switch (device_type_) {
    case kCPU: {
      for (int h = 0; h < kv_heads_; ++h) {
        for (int r = 0; r < repeat_times; ++r) {
          // clang-format off
          auto k_cache_ptr = k_cache_[layer_idx].offsettedPtr<mllm_byte_t>({0, h * repeat_times + r, current_seq_cnt_[layer_idx], 0});
          auto v_cache_ptr = v_cache_[layer_idx].offsettedPtr<mllm_byte_t>({0, h * repeat_times + r, current_seq_cnt_[layer_idx], 0});
          // clang-format on
          auto k_ptr = k.offsettedPtr<mllm_byte_t>({0, h, 0, 0});
          auto v_ptr = v.offsettedPtr<mllm_byte_t>({0, h, 0, 0});
          // Copy
          std::memcpy(k_cache_ptr, k_ptr, inputs_seq_len * kv_dims_ * bytesOfType(k_dtype_) / lanesOfType(k_dtype_));
          std::memcpy(v_cache_ptr, v_ptr, inputs_seq_len * kv_dims_ * bytesOfType(v_dtype_) / lanesOfType(v_dtype_));
        }
      }
      break;
    }
    default: {
      for (int h = 0; h < kv_heads_; ++h) {
        for (int r = 0; r < repeat_times; ++r) {
          // clang-format off
          k[{kAll, h, kAll, kAll}].copy2(k_cache_[layer_idx][{kAll, h * repeat_times + r, {current_seq_cnt_[layer_idx], current_seq_cnt_[layer_idx] + inputs_seq_len}, kAll}]);
          v[{kAll, h, kAll, kAll}].copy2(v_cache_[layer_idx][{kAll, h * repeat_times + r, {current_seq_cnt_[layer_idx], current_seq_cnt_[layer_idx] + inputs_seq_len}, kAll}]);
          // clang-format on
        }
      }
      break;
    }
  }

  // Update sequence length.
  current_seq_cnt_[layer_idx] += inputs_seq_len;

  // Write current_seq_cnt to mapped memory
  int32_t* seq_cnt_ptr = reinterpret_cast<int32_t*>(static_cast<char*>(mapped_memory_) + getLayerOffset(layer_idx));
  *seq_cnt_ptr = current_seq_cnt_[layer_idx];

  return {
      k_cache_[layer_idx][{kAll, kAll, {kAll, current_seq_cnt_[layer_idx]}, kAll}],
      v_cache_[layer_idx][{kAll, kAll, {kAll, current_seq_cnt_[layer_idx]}, kAll}],
  };
}
__MLLM_UNSAFE_OPT_END

}  // namespace mllm::nn

