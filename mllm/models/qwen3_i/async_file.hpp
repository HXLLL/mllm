// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

/**
 * Asynchronous I/O file abstraction with buffer pool support.
 *
 * AsyncFile: True async I/O using Linux AIO (io_submit/io_getevents)
 *            - Multiple in-flight requests supported
 *            - Buffer pool for O_DIRECT alignment
 *            - Non-blocking poll for completions
 */

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <functional>
#include <memory>
#include <vector>
#include <queue>
#include <unordered_map>

#include <fcntl.h>
#include <unistd.h>
#include <libaio.h>

#include "mllm/utils/Common.hpp"
#include "mllm/utils/Log.hpp"

namespace mllm::models::qwen3_i {

namespace fs = std::filesystem;

inline constexpr size_t kAsyncIOAlignment = 4096;
inline constexpr int32_t kDefaultMaxInflight = 64;
inline constexpr size_t kDefaultBufferSize = 1024 * 1024;

using RequestId = uint64_t;

enum class AsyncIOType {
  kRead,
  kWrite,
};

enum class AsyncIOStatus {
  kPending,
  kSubmitted,
  kCompleted,
  kFailed,
};

struct AsyncIOCompletion {
  RequestId request_id;
  AsyncIOStatus status;
  ssize_t bytes_transferred;
  int error_code;
};

using AsyncIOCallback = std::function<void(RequestId, bool, ssize_t)>;

class AlignedBuffer {
 public:
  AlignedBuffer() = default;

  explicit AlignedBuffer(size_t size) : size_(alignUp(size)) {
    int ret = posix_memalign(&data_, kAsyncIOAlignment, size_);
    if (ret != 0) {
      MLLM_ERROR_EXIT(ExitCode::kIOError, "posix_memalign failed: {} ({})", ret, strerror(ret));
    }
  }

  ~AlignedBuffer() {
    if (data_ != nullptr) { free(data_); }
  }

  AlignedBuffer(const AlignedBuffer&) = delete;
  AlignedBuffer& operator=(const AlignedBuffer&) = delete;

  AlignedBuffer(AlignedBuffer&& other) noexcept : data_(other.data_), size_(other.size_) {
    other.data_ = nullptr;
    other.size_ = 0;
  }

  AlignedBuffer& operator=(AlignedBuffer&& other) noexcept {
    if (this != &other) {
      if (data_ != nullptr) { free(data_); }
      data_ = other.data_;
      size_ = other.size_;
      other.data_ = nullptr;
      other.size_ = 0;
    }
    return *this;
  }

  [[nodiscard]] void* data() const { return data_; }
  [[nodiscard]] size_t size() const { return size_; }

 private:
  static size_t alignUp(size_t size) {
    return (size + kAsyncIOAlignment - 1) & ~(kAsyncIOAlignment - 1);
  }

  void* data_ = nullptr;
  size_t size_ = 0;
};

class BufferPool {
 public:
  BufferPool() = default;

  BufferPool(int32_t count, size_t buffer_size) : buffer_size_(buffer_size) {
    buffers_.reserve(count);
    for (int32_t i = 0; i < count; ++i) {
      buffers_.emplace_back(buffer_size);
      free_indices_.push(i);
    }
  }

  [[nodiscard]] std::pair<int32_t, void*> borrow() {
    if (free_indices_.empty()) { return {-1, nullptr}; }
    int32_t idx = free_indices_.front();
    free_indices_.pop();
    return {idx, buffers_[idx].data()};
  }

  void release(int32_t idx) {
    MLLM_RT_ASSERT(idx >= 0 && idx < static_cast<int32_t>(buffers_.size()));
    free_indices_.push(idx);
  }

  [[nodiscard]] bool hasAvailable() const { return !free_indices_.empty(); }
  [[nodiscard]] int32_t availableCount() const {
    return static_cast<int32_t>(free_indices_.size());
  }
  [[nodiscard]] size_t bufferSize() const { return buffer_size_; }

 private:
  std::vector<AlignedBuffer> buffers_;
  std::queue<int32_t> free_indices_;
  size_t buffer_size_ = 0;
};

struct AsyncIORequest {
  RequestId id;
  AsyncIOType type;
  AsyncIOStatus status;

  int fd;
  off_t file_offset;
  size_t user_count;
  void* user_buffer;

  off_t aligned_offset;
  size_t aligned_count;
  size_t prefix_skip;

  int32_t buffer_idx;
  void* aligned_buffer;

  AsyncIOCallback callback;
  struct iocb cb;
};

class AsyncFile {
 public:
  AsyncFile() = default;

  explicit AsyncFile(const fs::path& path, int32_t max_inflight = kDefaultMaxInflight,
                     size_t buffer_size = kDefaultBufferSize)
      : path_(path), max_inflight_(max_inflight), buffer_pool_(max_inflight, alignUp(buffer_size)) {
    fd_ = ::open(path.c_str(), O_RDWR | O_CREAT | O_DIRECT, 0644);
    if (fd_ < 0) {
      MLLM_ERROR_EXIT(ExitCode::kIOError, "Failed to open file {} with O_DIRECT, errno: {} ({})",
                      path.string(), errno, strerror(errno));
    }

    int ret = io_setup(max_inflight, &ctx_);
    if (ret < 0) {
      ::close(fd_);
      MLLM_ERROR_EXIT(ExitCode::kIOError, "io_setup failed: {} ({})", -ret, strerror(-ret));
    }
  }

  ~AsyncFile() {
    if (ctx_ != nullptr) { io_destroy(ctx_); }
    if (fd_ >= 0) { ::close(fd_); }
  }

  AsyncFile(const AsyncFile&) = delete;
  AsyncFile& operator=(const AsyncFile&) = delete;

  AsyncFile(AsyncFile&& other) noexcept
      : path_(std::move(other.path_)),
        fd_(other.fd_),
        ctx_(other.ctx_),
        max_inflight_(other.max_inflight_),
        buffer_pool_(std::move(other.buffer_pool_)),
        inflight_requests_(std::move(other.inflight_requests_)),
        next_request_id_(other.next_request_id_) {
    other.fd_ = -1;
    other.ctx_ = nullptr;
  }

  AsyncFile& operator=(AsyncFile&& other) noexcept {
    if (this != &other) {
      if (ctx_ != nullptr) { io_destroy(ctx_); }
      if (fd_ >= 0) { ::close(fd_); }

      path_ = std::move(other.path_);
      fd_ = other.fd_;
      ctx_ = other.ctx_;
      max_inflight_ = other.max_inflight_;
      buffer_pool_ = std::move(other.buffer_pool_);
      inflight_requests_ = std::move(other.inflight_requests_);
      next_request_id_ = other.next_request_id_;

      other.fd_ = -1;
      other.ctx_ = nullptr;
    }
    return *this;
  }

  RequestId submitRead(void* buf, size_t count, off_t offset, AsyncIOCallback callback = nullptr) {
    return submitRequest(AsyncIOType::kRead, buf, count, offset, std::move(callback));
  }

  RequestId submitWrite(const void* buf, size_t count, off_t offset,
                        AsyncIOCallback callback = nullptr) {
    return submitRequest(AsyncIOType::kWrite, const_cast<void*>(buf), count, offset,
                         std::move(callback));
  }

  std::vector<AsyncIOCompletion> poll(int32_t max_events = 16) {
    std::vector<AsyncIOCompletion> completions;
    if (inflight_requests_.empty()) { return completions; }

    std::vector<struct io_event> events(max_events);
    struct timespec timeout = {0, 0};

    int ret = io_getevents(ctx_, 0, max_events, events.data(), &timeout);
    if (ret < 0) {
      MLLM_WARN("io_getevents failed: {} ({})", -ret, strerror(-ret));
      return completions;
    }

    for (int i = 0; i < ret; ++i) {
      auto* req = static_cast<AsyncIORequest*>(events[i].data);
      AsyncIOCompletion completion;
      completion.request_id = req->id;

      if (static_cast<ssize_t>(events[i].res) < 0) {
        completion.status = AsyncIOStatus::kFailed;
        completion.bytes_transferred = 0;
        completion.error_code = -static_cast<int>(events[i].res);
      } else {
        completion.status = AsyncIOStatus::kCompleted;
        completion.bytes_transferred = static_cast<ssize_t>(events[i].res);
        completion.error_code = 0;

        if (req->type == AsyncIOType::kRead) {
          memcpy(req->user_buffer, static_cast<char*>(req->aligned_buffer) + req->prefix_skip,
                 req->user_count);
        }
      }

      if (req->callback) {
        req->callback(req->id, completion.status == AsyncIOStatus::kCompleted,
                      completion.bytes_transferred);
      }

      buffer_pool_.release(req->buffer_idx);
      inflight_requests_.erase(req->id);

      completions.push_back(completion);
    }

    return completions;
  }

  std::vector<AsyncIOCompletion> waitAny(int32_t max_events = 16) {
    std::vector<AsyncIOCompletion> completions;
    if (inflight_requests_.empty()) { return completions; }

    std::vector<struct io_event> events(max_events);

    int ret = io_getevents(ctx_, 1, max_events, events.data(), nullptr);
    if (ret < 0) {
      MLLM_ERROR_EXIT(ExitCode::kIOError, "io_getevents failed: {} ({})", -ret, strerror(-ret));
    }

    for (int i = 0; i < ret; ++i) {
      auto* req = static_cast<AsyncIORequest*>(events[i].data);
      AsyncIOCompletion completion;
      completion.request_id = req->id;

      if (static_cast<ssize_t>(events[i].res) < 0) {
        completion.status = AsyncIOStatus::kFailed;
        completion.bytes_transferred = 0;
        completion.error_code = -static_cast<int>(events[i].res);
      } else {
        completion.status = AsyncIOStatus::kCompleted;
        completion.bytes_transferred = static_cast<ssize_t>(events[i].res);
        completion.error_code = 0;

        if (req->type == AsyncIOType::kRead) {
          memcpy(req->user_buffer, static_cast<char*>(req->aligned_buffer) + req->prefix_skip,
                 req->user_count);
        }
      }

      if (req->callback) {
        req->callback(req->id, completion.status == AsyncIOStatus::kCompleted,
                      completion.bytes_transferred);
      }

      buffer_pool_.release(req->buffer_idx);
      inflight_requests_.erase(req->id);

      completions.push_back(completion);
    }

    return completions;
  }

  [[nodiscard]] bool hasCapacity() const { return buffer_pool_.hasAvailable(); }
  [[nodiscard]] int32_t inflightCount() const {
    return static_cast<int32_t>(inflight_requests_.size());
  }
  [[nodiscard]] int32_t availableSlots() const { return buffer_pool_.availableCount(); }
  [[nodiscard]] int32_t maxInflight() const { return max_inflight_; }

  void fsync() { ::fsync(fd_); }
  void fdatasync() { ::fdatasync(fd_); }

  [[nodiscard]] int fd() const { return fd_; }
  [[nodiscard]] const fs::path& path() const { return path_; }

 private:
  static size_t alignUp(size_t size) {
    return (size + kAsyncIOAlignment - 1) & ~(kAsyncIOAlignment - 1);
  }

  std::tuple<off_t, size_t, size_t> alignParams(off_t offset, size_t count) const {
    off_t aligned_offset = offset & ~static_cast<off_t>(kAsyncIOAlignment - 1);
    size_t prefix_skip = offset - aligned_offset;
    size_t aligned_count = alignUp(prefix_skip + count);
    return {aligned_offset, prefix_skip, aligned_count};
  }

  RequestId submitRequest(AsyncIOType type, void* buf, size_t count, off_t offset,
                          AsyncIOCallback callback) {
    auto [buffer_idx, aligned_buffer] = buffer_pool_.borrow();
    if (buffer_idx < 0) {
      MLLM_WARN("AsyncFile: no buffer available for request");
      return 0;
    }

    auto [aligned_offset, prefix_skip, aligned_count] = alignParams(offset, count);

    if (aligned_count > buffer_pool_.bufferSize()) {
      buffer_pool_.release(buffer_idx);
      MLLM_ERROR_EXIT(ExitCode::kIOError, "Request size {} exceeds buffer size {}", aligned_count,
                      buffer_pool_.bufferSize());
    }

    RequestId req_id = ++next_request_id_;
    auto& req = inflight_requests_[req_id];
    req.id = req_id;
    req.type = type;
    req.status = AsyncIOStatus::kSubmitted;
    req.fd = fd_;
    req.file_offset = offset;
    req.user_count = count;
    req.user_buffer = buf;
    req.aligned_offset = aligned_offset;
    req.aligned_count = aligned_count;
    req.prefix_skip = prefix_skip;
    req.buffer_idx = buffer_idx;
    req.aligned_buffer = aligned_buffer;
    req.callback = std::move(callback);

    if (type == AsyncIOType::kRead) {
      io_prep_pread(&req.cb, fd_, aligned_buffer, aligned_count, aligned_offset);
    } else {
      // Read-modify-write for unaligned O_DIRECT writes
      if (prefix_skip > 0 || (prefix_skip + count) % kAsyncIOAlignment != 0) {
        ssize_t ret = ::pread(fd_, aligned_buffer, aligned_count, aligned_offset);
        if (ret < 0) {
          buffer_pool_.release(buffer_idx);
          inflight_requests_.erase(req_id);
          MLLM_WARN("AsyncFile: RMW read failed: {}", strerror(errno));
          return 0;
        }
      }
      memcpy(static_cast<char*>(aligned_buffer) + prefix_skip, buf, count);
      io_prep_pwrite(&req.cb, fd_, aligned_buffer, aligned_count, aligned_offset);
    }

    req.cb.data = &req;

    struct iocb* cbs[1] = {&req.cb};
    int ret = io_submit(ctx_, 1, cbs);
    if (ret != 1) {
      buffer_pool_.release(buffer_idx);
      inflight_requests_.erase(req_id);
      MLLM_WARN("AsyncFile: io_submit failed: {} ({})", -ret, strerror(-ret));
      return 0;
    }

    return req_id;
  }

  fs::path path_;
  int fd_ = -1;
  io_context_t ctx_ = nullptr;
  int32_t max_inflight_ = 0;

  BufferPool buffer_pool_;
  std::unordered_map<RequestId, AsyncIORequest> inflight_requests_;
  RequestId next_request_id_ = 0;
};

}  // namespace mllm::models::qwen3_i
