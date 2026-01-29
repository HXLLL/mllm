#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>

#include <fcntl.h>
#include <unistd.h>
#include <libaio.h>

#include "mllm/utils/Common.hpp"
#include "mllm/utils/Log.hpp"

namespace mllm::models::qwen3_i {

namespace fs = std::filesystem;

// Alignment required for O_DIRECT (typically 512 or 4096)
inline constexpr size_t kDirectIOAlignment = 4096;

// Align size up to alignment boundary
inline size_t alignUp(size_t size, size_t alignment) {
  return (size + alignment - 1) & ~(alignment - 1);
}

// Align pointer down to alignment boundary
inline void* alignDown(void* ptr, size_t alignment) {
  return reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(ptr) & ~(alignment - 1));
}

// Check if pointer is aligned
inline bool isAligned(const void* ptr, size_t alignment) {
  return (reinterpret_cast<uintptr_t>(ptr) % alignment) == 0;
}

/**
 * AIO context for managing async I/O operations.
 * Currently implements synchronous versions (submit + wait immediately).
 */
class AioContext {
 public:
  explicit AioContext(int max_events = 64) {
    ctx_ = 0;
    int ret = io_setup(max_events, &ctx_);
    if (ret < 0) {
      MLLM_ERROR_EXIT(ExitCode::kIOError, "io_setup failed: {} ({})", -ret, strerror(-ret));
    }
  }

  ~AioContext() {
    if (ctx_) { io_destroy(ctx_); }
  }

  AioContext(const AioContext&) = delete;
  AioContext& operator=(const AioContext&) = delete;

  AioContext(AioContext&& other) noexcept : ctx_(other.ctx_) { other.ctx_ = 0; }

  AioContext& operator=(AioContext&& other) noexcept {
    if (this != &other) {
      if (ctx_) { io_destroy(ctx_); }
      ctx_ = other.ctx_;
      other.ctx_ = 0;
    }
    return *this;
  }

  io_context_t get() const { return ctx_; }

 private:
  io_context_t ctx_;
};

/**
 * File descriptor with O_DIRECT and AIO support.
 * Uses bounce buffer internally to handle alignment requirements.
 */
class AioFile {
 public:
  AioFile() = default;

  explicit AioFile(const fs::path& path, size_t bounce_buffer_size = 1024 * 1024)
      : path_(path), bounce_buffer_size_(alignUp(bounce_buffer_size, kDirectIOAlignment)) {
    // Open with O_DIRECT for true async I/O
    fd_ = ::open(path.c_str(), O_RDWR | O_CREAT | O_DIRECT, 0644);
    if (fd_ < 0) {
      MLLM_ERROR_EXIT(ExitCode::kIOError, "Failed to open file {} with O_DIRECT, errno: {} ({})",
                      path.string(), errno, strerror(errno));
    }

    // Allocate aligned bounce buffer
    int ret = posix_memalign(&bounce_buffer_, kDirectIOAlignment, bounce_buffer_size_);
    if (ret != 0) {
      ::close(fd_);
      MLLM_ERROR_EXIT(ExitCode::kIOError, "posix_memalign failed: {} ({})", ret, strerror(ret));
    }
  }

  ~AioFile() {
    if (fd_ >= 0) { ::close(fd_); }
    if (bounce_buffer_) { free(bounce_buffer_); }
  }

  AioFile(const AioFile&) = delete;
  AioFile& operator=(const AioFile&) = delete;

  AioFile(AioFile&& other) noexcept
      : path_(std::move(other.path_)),
        fd_(other.fd_),
        bounce_buffer_(other.bounce_buffer_),
        bounce_buffer_size_(other.bounce_buffer_size_),
        ctx_(std::move(other.ctx_)) {
    other.fd_ = -1;
    other.bounce_buffer_ = nullptr;
  }

  AioFile& operator=(AioFile&& other) noexcept {
    if (this != &other) {
      if (fd_ >= 0) { ::close(fd_); }
      if (bounce_buffer_) { free(bounce_buffer_); }

      path_ = std::move(other.path_);
      fd_ = other.fd_;
      bounce_buffer_ = other.bounce_buffer_;
      bounce_buffer_size_ = other.bounce_buffer_size_;
      ctx_ = std::move(other.ctx_);

      other.fd_ = -1;
      other.bounce_buffer_ = nullptr;
    }
    return *this;
  }

  /**
   * Synchronous pread using AIO.
   * Handles unaligned offset/size by using bounce buffer.
   */
  ssize_t pread(void* buf, size_t count, off_t offset) {
    // Calculate aligned boundaries
    off_t aligned_offset = offset & ~(off_t)(kDirectIOAlignment - 1);
    size_t prefix_skip = offset - aligned_offset;
    size_t aligned_count = alignUp(prefix_skip + count, kDirectIOAlignment);

    // Use bounce buffer for the aligned read
    MLLM_RT_ASSERT(aligned_count <= bounce_buffer_size_);

    // Prepare iocb
    struct iocb cb;
    struct iocb* cbs[1] = {&cb};
    io_prep_pread(&cb, fd_, bounce_buffer_, aligned_count, aligned_offset);

    // Submit
    int ret = io_submit(ctx_.get(), 1, cbs);
    if (ret != 1) {
      MLLM_ERROR_EXIT(ExitCode::kIOError, "io_submit pread failed: {} ({})", -ret, strerror(-ret));
    }

    // Wait for completion
    struct io_event event;
    ret = io_getevents(ctx_.get(), 1, 1, &event, nullptr);
    if (ret != 1) {
      MLLM_ERROR_EXIT(ExitCode::kIOError, "io_getevents failed: {} ({})", -ret, strerror(-ret));
    }

    if (static_cast<ssize_t>(event.res) < 0) {
      MLLM_ERROR_EXIT(ExitCode::kIOError, "AIO pread failed: {} ({})", -event.res,
                      strerror(-event.res));
    }

    // Copy from bounce buffer to user buffer
    memcpy(buf, static_cast<char*>(bounce_buffer_) + prefix_skip, count);
    return count;
  }

  /**
   * Synchronous pwrite using AIO.
   * Handles unaligned offset/size by using read-modify-write with bounce buffer.
   */
  ssize_t pwrite(const void* buf, size_t count, off_t offset) {
    off_t aligned_offset = offset & ~(off_t)(kDirectIOAlignment - 1);
    size_t prefix_skip = offset - aligned_offset;
    size_t aligned_count = alignUp(prefix_skip + count, kDirectIOAlignment);

    MLLM_RT_ASSERT(aligned_count <= bounce_buffer_size_);

    // If not aligned, need read-modify-write
    if (prefix_skip > 0 || (prefix_skip + count) % kDirectIOAlignment != 0) {
      // Read existing data first (using regular pread to avoid recursion complexity)
      ssize_t rd = ::pread(fd_, bounce_buffer_, aligned_count, aligned_offset);
      // Note: For new files or extending writes, pread may return less than requested.
      // That's okay - we'll overwrite the relevant portion anyway.
      (void)rd;
    }

    // Copy user data into bounce buffer at the correct offset
    memcpy(static_cast<char*>(bounce_buffer_) + prefix_skip, buf, count);

    // Prepare iocb for write
    struct iocb cb;
    struct iocb* cbs[1] = {&cb};
    io_prep_pwrite(&cb, fd_, bounce_buffer_, aligned_count, aligned_offset);

    // Submit
    int ret = io_submit(ctx_.get(), 1, cbs);
    if (ret != 1) {
      MLLM_ERROR_EXIT(ExitCode::kIOError, "io_submit pwrite failed: {} ({})", -ret, strerror(-ret));
    }

    // Wait for completion
    struct io_event event;
    ret = io_getevents(ctx_.get(), 1, 1, &event, nullptr);
    if (ret != 1) {
      MLLM_ERROR_EXIT(ExitCode::kIOError, "io_getevents failed: {} ({})", -ret, strerror(-ret));
    }

    if (static_cast<ssize_t>(event.res) < 0) {
      MLLM_ERROR_EXIT(ExitCode::kIOError, "AIO pwrite failed: {} ({})", -event.res,
                      strerror(-event.res));
    }

    return count;
  }

  void fsync() { ::fsync(fd_); }

  [[nodiscard]] int fd() const { return fd_; }
  [[nodiscard]] const fs::path& path() const { return path_; }

 private:
  fs::path path_;
  int fd_ = -1;
  void* bounce_buffer_ = nullptr;
  size_t bounce_buffer_size_ = 0;
  AioContext ctx_;
};

/**
 * Preallocate file with O_DIRECT compatibility.
 * The file size is aligned up to kDirectIOAlignment.
 */
inline void fallocate_file_direct(const fs::path& dir, const std::string& filename, size_t size) {
  if (!fs::exists(dir)) { fs::create_directories(dir); }

  auto filepath = dir / filename;
  size_t aligned_size = alignUp(size, kDirectIOAlignment);

  int fd = open(filepath.c_str(), O_CREAT | O_WRONLY, 0644);
  if (fd < 0) {
    MLLM_ERROR_EXIT(ExitCode::kIOError, "Failed to create file {}", filepath.string());
  }

  if (fallocate(fd, 0, 0, aligned_size) != 0) {
    close(fd);
    MLLM_ERROR_EXIT(ExitCode::kIOError, "Failed to preallocate {} bytes for {}", aligned_size,
                    filepath.string());
  }

  close(fd);
}

}  // namespace mllm::models::qwen3_i
