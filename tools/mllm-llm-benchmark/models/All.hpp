// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include <memory>
#include <algorithm>

#include "Qwen3_W4A32_KAI.hpp"
#include "Qwen3_W4A32_KAI_i.hpp"
#include "BenchmarkTemplate.hpp"

std::shared_ptr<BenchmarkTemplate> createBenchmark(const std::string& model_name, const std::string& cfg_path, const std::string& model_path, int32_t cache_length, const std::string& cache_dir, bool intermittent = false) {
  auto tolower = [](const std::string& str) {
    std::string result = str;
    std::transform(result.begin(), result.end(), result.begin(), ::tolower);
    return result;
  };
  auto normalized_model_name = tolower(model_name);
  if (normalized_model_name.find("qwen3") != std::string::npos && normalized_model_name.find("w4a32") != std::string::npos
      && normalized_model_name.find("kai") != std::string::npos) {
    if (intermittent) {
      return std::make_shared<Qwen3_W4A32_KAI_Benchmark_Intermittent>(cfg_path, model_path, cache_length, cache_dir);
    } else {
      return std::make_shared<Qwen3_W4A32_KAI_Benchmark>(cfg_path, model_path, cache_length);
    }
  }
  return nullptr;
}
