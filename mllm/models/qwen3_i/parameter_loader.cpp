// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/models/qwen3_i/parameter_loader.hpp"

#include "mllm/core/ParameterFile.hpp"
#include "mllm/core/TensorStorage.hpp"
#include "mllm/core/TensorViewImpl.hpp"
#include "mllm/utils/Common.hpp"
#include "mllm/utils/Log.hpp"

namespace mllm {

ParameterLoader::ParameterLoader(const std::string& file_path)
    : file_path_(file_path), parameter_file_(ParameterFile::create(ModelFileVersion::kV2)) {
  file_.open(file_path, std::ios::binary);
  if (!file_.is_open()) {
    MLLM_ERROR_EXIT(ExitCode::kIOError, "Failed to open parameter file: {}", file_path);
  }

  // Read header
  ModelFileV2Descriptor header;
  file_.read(reinterpret_cast<char*>(&header), sizeof(ModelFileV2Descriptor));

  MLLM_RT_ASSERT_EQ(MLLM_MODEL_FILE_V2_MAGIC_NUMBER, header.magic_number);
  MLLM_RT_ASSERT_EQ(MLLM_MODEL_FILE_V2_VERSION, header.version);

  // Read all descriptors
  std::vector<ModelFileV2ParamsDescriptor> param_descriptors(header.num_params);
  file_.seekg(header.params_desc_offset);
  file_.read(reinterpret_cast<char*>(param_descriptors.data()),
                 header.num_params * sizeof(ModelFileV2ParamsDescriptor));

  // Build name -> descriptor map
  for (const auto& desc : param_descriptors) {
    std::string name(desc._param_name_view());
    descriptors_[name] = desc;
  }
}

ParameterLoader::~ParameterLoader() {
  if (file_.is_open()) {
    file_.close();
  }
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

  file_.seekg(desc.parameter_offset);
  file_.read(static_cast<char*>(s->ptr_), desc.parameter_size);
  s->mem_type_ = kParamsNormal;

  parameter_file_->push(name, tensor);
}

bool ParameterLoader::isLoaded(const std::string& name) const {
  return parameter_file_->has(name);
}

ParameterFile::ptr_t ParameterLoader::getParameterFile() {
  return parameter_file_;
}

}  // namespace mllm
