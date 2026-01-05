#pragma once
#include <chrono>
#include <optional>

namespace mllm {

class Timer {
  using clk = std::chrono::high_resolution_clock;
  clk::time_point start_;
  std::optional<clk::time_point> stop_;

public:
  Timer() : start_(clk::now()) {}
  void start() { start_ = clk::now(); stop_.reset(); }
  void stop() { stop_ = clk::now(); }
  [[nodiscard]] int64_t elapsed_ms() const {
    return std::chrono::duration_cast<std::chrono::milliseconds>(stop_.value_or(clk::now()) - start_).count();
  }
};

}  // namespace mllm
