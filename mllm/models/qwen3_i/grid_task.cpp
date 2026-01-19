// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/models/qwen3_i/grid_task.hpp"

#include "mllm/models/qwen3_i/modeling_qwen3_i.hpp"
#include "mllm/utils/AnyValue.hpp"
#include "mllm/utils/Common.hpp"

namespace mllm::models::qwen3_i {

// LoadParamTask

LoadParamTask::LoadParamTask(int layer, const std::vector<std::string>& param_names,
                             ParameterLoader& loader, GenerationState& state)
    : GridTask(layer, 0, {layer, 0, 0}, state), param_names_(param_names), loader_(loader) {}

void LoadParamTask::execute() {
  for (const auto& name : param_names_) {
    MLLM_RT_ASSERT(!loader_.isLoaded(name));
    loader_.loadTensor(name);
  }
  markCompleted();
}

std::string LoadParamTask::name() const { return "LoadParam[" + std::to_string(layer()) + "]"; }

// LoadKVTask

LoadKVTask::LoadKVTask(int layer, int chunk_id, const CacheRange& range, GenerationState& state)
    : GridTask(layer, chunk_id, range, state) {}

void LoadKVTask::execute() {
  const auto& r = range();
  MLLM_RT_ASSERT(!state_.isKVLoaded(r));
  state_.loadLayerKCache(r);
  state_.loadLayerVCache(r);
  markCompleted();
}

std::string LoadKVTask::name() const { return "LoadKV[" + std::to_string(layer()) + "][" + std::to_string(chunkId()) + "]"; }

// LoadHTask

LoadHTask::LoadHTask(int layer, int chunk_id, const CacheRange& range, GenerationState& state)
    : GridTask(layer, chunk_id, range, state) {}

void LoadHTask::execute() {
  const auto& r = range();
  MLLM_RT_ASSERT(!state_.isHLoaded(r));
  state_.loadLayerHCache(r); 
  markCompleted();
}

std::string LoadHTask::name() const { return "LoadH[" + std::to_string(layer()) + "][" + std::to_string(chunkId()) + "]"; }

// ComputeTask

ComputeTask::ComputeTask(int layer, int chunk_id, const CacheRange& range, H2KV& h2kv, KV2H& kv2h, GenerationState& state,
                         const Tensor& sin_emb, const Tensor& cos_emb)
    : GridTask(layer, chunk_id, range, state), h2kv_(h2kv), kv2h_(kv2h), sin_emb_(sin_emb), cos_emb_(cos_emb) {}

void ComputeTask::execute() {
  const auto& r = range();

  Tensor x = state_.getH(r);

  auto h2kv_result = h2kv_(x, sin_emb_, cos_emb_);
  auto& h = h2kv_result[0];
  auto& q = h2kv_result[1];
  auto& k = h2kv_result[2];
  auto& v = h2kv_result[3];

  state_.updateKV(r, k, v);

  int offset = r.offset;
  Tensor x_out = kv2h_(h, q, k, AnyValue(offset))[0];

  CacheRange next_range{r.layer_idx + 1, r.offset, r.count};
  state_.updateH(next_range, x_out);

  state_.checkpoint();
  markCompleted();
}

std::string ComputeTask::name() const { return "Compute[" + std::to_string(layer()) + "][" + std::to_string(chunkId()) + "]"; }

}  // namespace mllm::models::qwen3_i
