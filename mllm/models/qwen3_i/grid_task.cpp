// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/models/qwen3_i/modeling_qwen3_i.hpp"
#include "mllm/models/qwen3_i/grid_task.hpp"
#include "mllm/models/qwen3_i/qwen3_events.hpp"

#include "mllm/mllm.hpp"
#include "mllm/utils/AnyValue.hpp"
#include "mllm/utils/Common.hpp"

namespace mllm::models::qwen3_i {

// LoadParamTask

void LoadParamTask::execute() {
  auto tracer = Context::instance().tracer();
  tracer->record<LoadParamBeginEvent>(layer_, -1);
  ctx_.loader().loadLayer(layer_);
  tracer->record<LoadParamCompleteEvent>(layer_, -1);
  complete();
}

// LoadKVTask

void LoadKVTask::execute() {
  auto tracer = Context::instance().tracer();
  tracer->record<LoadKVBeginEvent>(idx_.layer, idx_.chunk_id);
  const auto& r = idx_.range;
  auto& state = ctx_.state();
  MLLM_RT_ASSERT(!state.isKVLoaded(r));
  state.loadLayerKCache(r);
  state.loadLayerVCache(r);
  tracer->record<LoadKVCompleteEvent>(idx_.layer, idx_.chunk_id);
  complete();
}

// LoadHTask

void LoadHTask::execute() {
  auto tracer = Context::instance().tracer();
  tracer->record<LoadHBeginEvent>(idx_.layer, idx_.chunk_id);
  const auto& r = idx_.range;
  auto& state = ctx_.state();
  MLLM_RT_ASSERT(!state.isHLoaded(r));
  state.loadLayerHCache(r);
  tracer->record<LoadHCompleteEvent>(idx_.layer, idx_.chunk_id);
  complete();
}

// ComputeTask

void ComputeTask::execute() {
  auto tracer = Context::instance().tracer();
  tracer->record<ComputeBeginEvent>(idx_.layer, idx_.chunk_id);
  auto& r = idx_.range;
  auto& state = ctx_.state();

  Tensor x = state.getH(r);

  auto h2kv_result = h2kv_(x, sin_emb_, cos_emb_);
  auto& h = h2kv_result[0];
  auto& q = h2kv_result[1];
  auto& k = h2kv_result[2];
  auto& v = h2kv_result[3];

  state.updateKV(r, k, v);

  Tensor x_out = kv2h_(h, q, AnyValue(r.offset))[0];

  CacheRange next_range{r.layer_idx + 1, r.offset, r.count};
  state.updateH(next_range, x_out);

  state.checkpoint();
  tracer->record<ComputeCompleteEvent>(idx_.layer, idx_.chunk_id);
  complete();
}

}  // namespace mllm::models::qwen3_i
