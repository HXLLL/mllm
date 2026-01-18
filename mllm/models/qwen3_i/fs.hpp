#pragma once

#include <filesystem>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>

#include "mllm/utils/Common.hpp"
#include "mllm/utils/Log.hpp"

namespace mllm::models::qwen3_i {

namespace fs = std::filesystem;

class FileDescriptor {
 public:
  FileDescriptor() : fd_(-1) {}
  explicit FileDescriptor(const fs::path& path) : path_(path) {
    fd_ = open(path.c_str(), O_RDWR | O_CREAT);
    if (fd_ < 0) {
      MLLM_ERROR_EXIT(ExitCode::kIOError, "Failed to open file {}, errno: {} ({})", path.string(),
                      errno, strerror(errno));
    }
  }

  ~FileDescriptor() { close(fd_); }

  size_t read(void* buffer, size_t count) {
    int ret = ::read(fd_, buffer, count); 
    if (ret < 0) { MLLM_ERROR_EXIT(ExitCode::kIOError, "Failed to read from file {}, errno: {} ({})", path_.string(), errno, strerror(errno)); }
    return ret;
  }
   size_t write(const void* buffer, size_t count) { 
    int ret = ::write(fd_, buffer, count); 
    if (ret < 0) { MLLM_ERROR_EXIT(ExitCode::kIOError, "Failed to write to file {}, errno: {} ({})", path_.string(), errno, strerror(errno)); }
    return ret;
  }
  size_t pread(void* buffer, size_t count, off_t offset) {
    int ret = ::pread(fd_, buffer, count, offset);
    if (ret < 0) { MLLM_ERROR_EXIT(ExitCode::kIOError, "Failed to read from file {}, errno: {} ({})", path_.string(), errno, strerror(errno)); }
    return ret;
  }
  size_t pwrite(const void* buffer, size_t count, off_t offset) {
    int ret = ::pwrite(fd_, buffer, count, offset);
    if (ret < 0) { MLLM_ERROR_EXIT(ExitCode::kIOError, "Failed to write to file {}, errno: {} ({})", path_.string(), errno, strerror(errno)); }
    return ret;
  }
  void fsync() { 
    ::fsync(fd_); 
  }
  void seek(off_t offset) { 
    ::lseek(fd_, offset, SEEK_SET); 
  }

  [[nodiscard]] int fd() const { return fd_; }
  [[nodiscard]] fs::path path() const { return path_; }
 private:
  fs::path path_;
  int fd_;
};

static inline void fallocate_file(const fs::path& dir, const std::string& filename, size_t size) {
  if (!fs::exists(dir)) { fs::create_directories(dir); }

  auto filepath = dir / filename;
  int fd = open(filepath.c_str(), O_CREAT | O_WRONLY, 0644);
  if (fd < 0) { MLLM_ERROR_EXIT(ExitCode::kIOError, "Failed to create file {}", filepath.string()); }

  if (fallocate(fd, 0, 0, size) != 0) {
    close(fd);
    MLLM_ERROR_EXIT(ExitCode::kIOError, "Failed to preallocate {} bytes for {}", size, filepath.string());
  }

  close(fd);
}

} // namespace mllm::models::qwen3_i
