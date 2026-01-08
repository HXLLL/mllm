// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <fstream>
#include <memory>
#include <string>
#include <unordered_map>

#include "mllm/core/ParameterFile.hpp"
#include "mllm/core/schema/ModelFileV2.hpp"

namespace mllm {

// On-demand parameter loader for V2 model files.
// Inherits from ParameterFile so it can be used directly with Module::load().
// Tensors are loaded from disk on first pull().
class ParameterLoader {
 public:
  explicit ParameterLoader(const std::string& file_path);
  ~ParameterLoader();

  [[nodiscard]] bool isLoaded(const std::string& name) const;
  void loadTensor(const std::string& name);
  ParameterFile::ptr_t getParameterFile();

 private:
  std::string file_path_;
  std::ifstream file_;
  ParameterFile::ptr_t parameter_file_;
  std::unordered_map<std::string, ModelFileV2ParamsDescriptor> descriptors_;
};

}  // namespace mllm
