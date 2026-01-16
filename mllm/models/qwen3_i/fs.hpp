#include <filesystem>
#include <fstream>
#include <fcntl.h>
#include <unistd.h>

#include "mllm/utils/Common.hpp"
#include "mllm/utils/Log.hpp"

namespace mllm::models::qwen3_i {

namespace fs = std::filesystem;

class FileDescriptor {
 public:
  explicit FileDescriptor(const fs::path& path) : path_(path) {}
  ~FileDescriptor() { close(fd_); }
  [[nodiscard]] int fd() const { return fd_; }
  [[nodiscard]] fs::path path() const { return path_; }
 private:
  fs::path path_;
  int fd_;
};

template<typename... Args>
static inline std::fstream open_fstream(const fs::path& path, Args&&... args) {
  std::fstream stream;
  stream.open(path, std::forward<Args>(args)...);
  if (!stream.is_open()) { MLLM_ERROR_EXIT(ExitCode::kIOError, "Failed to open file {}", path.string()); }
  return stream;
}

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

static inline void fsync_file(const fs::path& path) {
  int fd = open(path.c_str(), O_WRONLY);
  if (fd >= 0) {
    fsync(fd);
    close(fd);
  }
}

} // namespace mllm::models::qwen3_i
