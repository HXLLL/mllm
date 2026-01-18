// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <unordered_map>

#include "mllm/core/ParameterFile.hpp"
#include "mllm/core/schema/ModelFileV2.hpp"
#include "mllm/models/qwen3_i/fs.hpp"

namespace mllm::models::qwen3_i {

// On-demand parameter loader for V2 model files.
// Inherits from ParameterFile so it can be used directly with Module::load().
// Tensors are loaded from disk on first pull().
class ParameterLoader {
 public:
  explicit ParameterLoader(const std::string& file_path);
  ~ParameterLoader() = default;

  void load();
  void lazyLoad();

  void loadTensor(const std::string& name);
  [[nodiscard]] bool isLoaded(const std::string& name) const { return parameter_file_->has(name); }

  ParameterFile::ptr_t getParameterFile() const { return parameter_file_; }
  void dumpTensorStatus() const;

 private:
  void loadHeader();

  std::string file_path_;
  FileDescriptor file_;
  ParameterFile::ptr_t parameter_file_;
  ModelFileV2Descriptor header_;
  std::unordered_map<std::string, ModelFileV2ParamsDescriptor> descriptors_;
};

}  // namespace mllm::models::qwen3_i
