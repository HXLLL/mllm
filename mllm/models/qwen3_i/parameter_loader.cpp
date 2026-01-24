// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/models/qwen3_i/parameter_loader.hpp"

#include <fmt/core.h>

#include "mllm/core/ParameterFile.hpp"
#include "mllm/core/TensorStorage.hpp"
#include "mllm/core/TensorViewImpl.hpp"
#include "mllm/utils/Common.hpp"
#include "mllm/utils/Log.hpp"

namespace mllm::models::qwen3_i {

ParameterLoader::ParameterLoader(const std::string& file_path)
    : file_path_(file_path),
      file_(file_path),
      parameter_file_(ParameterFile::create(ModelFileVersion::kV2)) {}

void ParameterLoader::load() {
  loadHeader();
  for (const auto& [name, desc] : descriptors_) {
    loadTensor(name);
  }
}

void ParameterLoader::lazyLoad() {
  loadHeader();
}

void ParameterLoader::loadTensor(const std::string& name) {
  if (parameter_file_->has(name)) return;

  auto it = descriptors_.find(name);
  if (it == descriptors_.end()) {
    MLLM_ERROR_EXIT(ExitCode::kIOError, "Tensor not found in file: {}", name);
  }

  const auto& desc = it->second;

  TensorViewImpl::shape_t shape;
  for (size_t j = 0; j < desc.shape_len && j < MLLM_MODEL_FILE_V2_TENSOR_SHAPE_LENGTH; j++) {
    shape.push_back(desc.shape[j]);
  }

  auto s = TensorStorage::create(shape, static_cast<DataTypes>(desc.parameter_type), kCPU);
  auto t = TensorViewImpl::create(shape, s);
  s->name_ = name;
  auto tensor = Tensor(t).alloc();
  file_.pread(s->ptr_, desc.parameter_size, desc.parameter_offset);
  s->mem_type_ = kParamsNormal;

  parameter_file_->push(name, tensor);
}

void ParameterLoader::registerLayer(int layer, std::vector<std::string> tensor_names) {
  if (layer >= static_cast<int>(layer_tensors_.size())) {
    layer_tensors_.resize(layer + 1);
    layer_loaded_.resize(layer + 1, false);
  }
  layer_tensors_[layer] = std::move(tensor_names);
}

void ParameterLoader::loadLayer(int layer) {
  MLLM_RT_ASSERT(layer >= 0 && layer < static_cast<int>(layer_tensors_.size()));
  for (const auto& name : layer_tensors_[layer]) {
    loadTensor(name);
  }
  layer_loaded_[layer] = true;
}

bool ParameterLoader::isLayerLoaded(int layer) const {
  if (layer < 0 || layer >= static_cast<int>(layer_loaded_.size())) return false;
  return layer_loaded_[layer];
}

void ParameterLoader::dumpTensorStatus() const {
  fmt::print("ParameterLoader tensor status (total: {}):\n", header_.num_params);
  int loaded_count = 0;
  for (const auto& [name, desc] : descriptors_) {
    bool loaded = parameter_file_->has(name);
    if (loaded) loaded_count++;
    fmt::print("  {}: {}\n", name, loaded ? "LOADED" : "NOT LOADED");
  }
  fmt::print("Loaded: {}/{}\n", loaded_count, header_.num_params);
}


void ParameterLoader::loadHeader() {
  file_.read(&header_, sizeof(ModelFileV2Descriptor));

  MLLM_RT_ASSERT_EQ(MLLM_MODEL_FILE_V2_MAGIC_NUMBER, header_.magic_number);
  MLLM_RT_ASSERT_EQ(MLLM_MODEL_FILE_V2_VERSION, header_.version);

  // Read all descriptors
  std::vector<ModelFileV2ParamsDescriptor> param_descriptors(header_.num_params);
  size_t param_descriptors_size = header_.num_params * sizeof(ModelFileV2ParamsDescriptor);
  file_.pread(param_descriptors.data(), param_descriptors_size, header_.params_desc_offset);

  // Build name -> descriptor map
  for (const auto& desc : param_descriptors) {
    std::string name(desc._param_name_view());
    descriptors_[name] = desc;
  }
}

}  // namespace mllm::models::qwen3_i
