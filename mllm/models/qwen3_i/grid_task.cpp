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
    : Task(), layer_(layer), param_names_(param_names), loader_(loader), state_(state) {}

void LoadParamTask::execute() {
  for (const auto& name : param_names_) {
    MLLM_RT_ASSERT(!loader_.isLoaded(name));
    loader_.loadTensor(name);
  }
  complete();
}

std::string LoadParamTask::name() const { return "LoadParam[" + std::to_string(layer_) + "]"; }

// LoadKVTask

void LoadKVTask::execute() {
  const auto& r = ctx_.range;
  MLLM_RT_ASSERT(!ctx_.state.isKVLoaded(r));
  ctx_.state.loadLayerKCache(r);
  ctx_.state.loadLayerVCache(r);
  complete();
}

std::string LoadKVTask::name() const { return "LoadKV[" + std::to_string(ctx_.layer) + "][" + std::to_string(ctx_.chunk_id) + "]"; }

// LoadHTask

LoadHTask::LoadHTask(const GridContext& ctx) : Task(), ctx_(ctx) {}

void LoadHTask::execute() {
  const auto& r = ctx_.range;
  MLLM_RT_ASSERT(!ctx_.state.isHLoaded(r));
  ctx_.state.loadLayerHCache(r); 
  complete();
}

std::string LoadHTask::name() const {
  return "LoadH[" + std::to_string(ctx_.layer) + "][" + std::to_string(ctx_.chunk_id) + "]";
}

// ComputeTask

ComputeTask::ComputeTask(const GridContext& ctx, H2KV& h2kv, KV2H& kv2h, Tensor sin_emb,
                         Tensor cos_emb)
    : Task(),
      ctx_(ctx),
      h2kv_(h2kv),
      kv2h_(kv2h),
      sin_emb_(std::move(sin_emb)),
      cos_emb_(std::move(cos_emb)) {}

void ComputeTask::execute() {
  const auto& r = ctx_.range;

  Tensor x = ctx_.state.getH(r);

  auto h2kv_result = h2kv_(x, sin_emb_, cos_emb_);
  auto& h = h2kv_result[0];
  auto& q = h2kv_result[1];
  auto& k = h2kv_result[2];
  auto& v = h2kv_result[3];

  ctx_.state.updateKV(r, k, v);

  int offset = r.offset;
  Tensor x_out = kv2h_(h, q, k, AnyValue(offset))[0];

  CacheRange next_range{r.layer_idx + 1, r.offset, r.count};
  ctx_.state.updateH(next_range, x_out);

  ctx_.state.checkpoint();
  complete();
}

std::string ComputeTask::name() const {
  return "Compute[" + std::to_string(ctx_.layer) + "][" + std::to_string(ctx_.chunk_id) + "]";
}

}  // namespace mllm::models::qwen3_i
