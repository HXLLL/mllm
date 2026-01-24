// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/models/qwen3_i/modeling_qwen3_i.hpp"
#include "mllm/models/qwen3_i/grid_task.hpp"

#include "mllm/utils/AnyValue.hpp"
#include "mllm/utils/Common.hpp"

namespace mllm::models::qwen3_i {

// LoadParamTask

void LoadParamTask::execute() {
  ctx_.loader().loadLayer(layer_);
  complete();
}

// LoadKVTask

void LoadKVTask::execute() {
  const auto& r = idx_.range;
  auto& state = ctx_.state();
  MLLM_RT_ASSERT(!state.isKVLoaded(r));
  state.loadLayerKCache(r);
  state.loadLayerVCache(r);
  complete();
}

// LoadHTask

void LoadHTask::execute() {
  const auto& r = idx_.range;
  auto& state = ctx_.state();
  MLLM_RT_ASSERT(!state.isHLoaded(r));
  state.loadLayerHCache(r);
  complete();
}

// ComputeTask

void ComputeTask::execute() {
  auto& r = idx_.range;
  auto& state = ctx_.state();

  Tensor x = state.getH(r);

  auto h2kv_result = h2kv_(x, sin_emb_, cos_emb_);
  auto& h = h2kv_result[0];
  auto& q = h2kv_result[1];
  auto& k = h2kv_result[2];
  auto& v = h2kv_result[3];

  state.updateKV(r, k, v);

  Tensor x_out = kv2h_(h, q, k, AnyValue(r.offset))[0];

  CacheRange next_range{r.layer_idx + 1, r.offset, r.count};
  state.updateH(next_range, x_out);

  state.checkpoint();
  complete();
}

}  // namespace mllm::models::qwen3_i
