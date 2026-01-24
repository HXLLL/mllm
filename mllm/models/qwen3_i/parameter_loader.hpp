// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <unordered_map>
#include <vector>

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

  // Layer-level interface
  void registerLayer(int layer, std::vector<std::string> tensor_names);
  void loadLayer(int layer);
  [[nodiscard]] bool isLayerLoaded(int layer) const;

  ParameterFile::ptr_t getParameterFile() const { return parameter_file_; }
  void dumpTensorStatus() const;

 private:
  void loadHeader();

  std::string file_path_;
  FileDescriptor file_;
  ParameterFile::ptr_t parameter_file_;
  ModelFileV2Descriptor header_;
  std::unordered_map<std::string, ModelFileV2ParamsDescriptor> descriptors_;
  std::vector<std::vector<std::string>> layer_tensors_;
  std::vector<bool> layer_loaded_;
};

}  // namespace mllm::models::qwen3_i
