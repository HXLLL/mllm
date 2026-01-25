// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/models/qwen3_i/modeling_qwen3_i.hpp"
#include "mllm/models/qwen3_i/generation_state.hpp"
#include "mllm/models/qwen3_i/grid_task.hpp"
#include "mllm/models/qwen3_i/qwen3_events.hpp"
#include "mllm/models/qwen3_i/runtime_context.hpp"
#include "mllm/models/qwen3_i/simple_grid_scheduler.hpp"
#include "mllm/nn/Functional.hpp"
#include "mllm/nn/Module.hpp"
#include "mllm/utils/AnyValue.hpp"
#include "mllm/utils/Common.hpp"
#include "mllm/utils/RoPE.hpp"
#include <algorithm>
#include <cmath>
#include <nlohmann/json.hpp>

namespace mllm::models::qwen3_i {

/** Qwen3MLP Implementation **/

Qwen3MLP::Qwen3MLP(const std::string& name, const Qwen3Config& cfg, ParameterLoader& parameter_loader)
    : nn::Module(name), parameter_loader_(parameter_loader) {
  gate_proj_ = reg<nn::Linear>("gate_proj", cfg.hidden_size, cfg.intermediate_size, false, cfg.linear_impl_type);
  up_proj_ = reg<nn::Linear>("up_proj", cfg.hidden_size, cfg.intermediate_size, false, cfg.linear_impl_type);
  down_proj_ = reg<nn::Linear>("down_proj", cfg.intermediate_size, cfg.hidden_size, false, cfg.linear_impl_type);
  silu_ = reg<nn::SiLU>("act");
}

void Qwen3MLP::loadFromDisk() {
  auto prefix = getModuleName() + ".";
  parameter_loader_.loadTensor(prefix + "gate_proj.weight");
  parameter_loader_.loadTensor(prefix + "up_proj.weight");
  parameter_loader_.loadTensor(prefix + "down_proj.weight");
}

std::vector<Tensor> Qwen3MLP::forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) {
  auto x = gate_proj_(inputs[0]);
  x = silu_(x);
  auto y = up_proj_(inputs[0]);
  x = x * y;
  x = down_proj_(x);
  return {x};
}

/** Qwen3Decoder Implementation **/

Qwen3Decoder::Qwen3Decoder(const std::string& name, const Qwen3Config& cfg, GenerationState& state,
                           ParameterLoader& parameter_loader, int idx)
    : nn::Module(name),
      layer_idx_(idx),
      hidden_size_(cfg.hidden_size),
      head_dim_(cfg.head_dim),
      num_attention_heads_(cfg.num_attention_heads),
      num_key_value_heads_(cfg.num_key_value_heads),
      q_proj_(reg<nn::Linear>("self_attn.q_proj", cfg.hidden_size, cfg.head_dim * cfg.num_attention_heads, cfg.attention_bias,
                              cfg.linear_impl_type)),
      k_proj_(reg<nn::Linear>("self_attn.k_proj", cfg.hidden_size, cfg.head_dim * cfg.num_key_value_heads, cfg.attention_bias,
                              cfg.linear_impl_type)),
      v_proj_(reg<nn::Linear>("self_attn.v_proj", cfg.hidden_size, cfg.head_dim * cfg.num_key_value_heads, cfg.attention_bias,
                              cfg.linear_impl_type)),
      o_proj_(reg<nn::Linear>("self_attn.o_proj", cfg.head_dim * cfg.num_attention_heads, cfg.hidden_size, cfg.attention_bias,
                              cfg.linear_impl_type)),
      rms_norm_q_(reg<nn::RMSNorm>("self_attn.q_norm", cfg.rms_norm_eps)),
      rms_norm_k_(reg<nn::RMSNorm>("self_attn.k_norm", cfg.rms_norm_eps)),
      q_rope_(reg<nn::RoPE>("self_attn.q_rope", cfg.rope_theta, cfg.max_position_embeddings)),
      k_rope_(reg<nn::RoPE>("self_attn.k_rope", cfg.rope_theta, cfg.max_position_embeddings)),
      mask_(reg<nn::CausalMask>("self_attn.mask")),
      softmax_(reg<nn::Softmax>("self_attn.softmax", -1)),
      mlp_(reg<Qwen3MLP>("mlp", cfg, parameter_loader)),
      input_layer_norm_(reg<nn::RMSNorm>("input_layernorm", cfg.rms_norm_eps)),
      post_attention_layer_norm_(reg<nn::RMSNorm>("post_attention_layernorm", cfg.rms_norm_eps)),
      state_(state),
      parameter_loader_(parameter_loader) {}

void Qwen3Decoder::loadFromDisk() {
  auto prefix = getModuleName() + ".";
  // Attention projections
  parameter_loader_.loadTensor(prefix + "self_attn.q_proj.weight");
  parameter_loader_.loadTensor(prefix + "self_attn.k_proj.weight");
  parameter_loader_.loadTensor(prefix + "self_attn.v_proj.weight");
  parameter_loader_.loadTensor(prefix + "self_attn.o_proj.weight");
  // QK norms
  parameter_loader_.loadTensor(prefix + "self_attn.q_norm.weight");
  parameter_loader_.loadTensor(prefix + "self_attn.k_norm.weight");
  // Layer norms
  parameter_loader_.loadTensor(prefix + "input_layernorm.weight");
  parameter_loader_.loadTensor(prefix + "post_attention_layernorm.weight");
  // Recurse into MLP
  mlp_.loadFromDisk();
}

std::vector<Tensor> Qwen3Decoder::forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) { return {}; }

/** Layer Tasks Implementation */

H2KV::H2KV(Qwen3Decoder& decoder, GenerationState& state)
    : layer_idx_(decoder.layer_idx_),
      hidden_size_(decoder.hidden_size_),
      head_dim_(decoder.head_dim_),
      num_attention_heads_(decoder.num_attention_heads_),
      num_key_value_heads_(decoder.num_key_value_heads_),
      input_layer_norm_(decoder.input_layer_norm_),
      q_proj_(decoder.q_proj_),
      k_proj_(decoder.k_proj_),
      v_proj_(decoder.v_proj_),
      rms_norm_q_(decoder.rms_norm_q_),
      rms_norm_k_(decoder.rms_norm_k_),
      q_rope_(decoder.q_rope_),
      k_rope_(decoder.k_rope_),
      state_(state),
      parameter_loader_(decoder.parameter_loader_) {}

void H2KV::load(const ParameterFile::ptr_t &param_file) {
  input_layer_norm_.impl()->load(param_file);
  q_proj_.impl()->load(param_file);
  k_proj_.impl()->load(param_file);
  v_proj_.impl()->load(param_file);
  rms_norm_q_.impl()->load(param_file);
  rms_norm_k_.impl()->load(param_file);
  q_rope_.impl()->load(param_file);
  k_rope_.impl()->load(param_file);
  pulled_ = true;
}

std::vector<Tensor> H2KV::forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) {
  const auto& hidden_states = inputs[0];
  const auto& sin_emb = inputs[1];
  const auto& cos_emb = inputs[2];
  assert(hidden_states.shape()[0] == 1);
  const int count = hidden_states.shape()[1];

  if (!pulled_) {
    load(parameter_loader_.getParameterFile());
  }

  // Apply input layer norm
  auto x = input_layer_norm_(hidden_states);
  // Project to Q, K, V
  auto query = q_proj_(x).view({1, count, num_attention_heads_, head_dim_});
  auto key = k_proj_(x).view({1, count, num_key_value_heads_, head_dim_});
  auto value = v_proj_(x).view({1, count, num_key_value_heads_, head_dim_});
  // Apply QK normalization
  query = rms_norm_q_(query);
  key = rms_norm_k_(key);
  // Transpose to [B, H, S, D] for attention computation
  query = query.transpose(1, 2);
  key = key.transpose(1, 2);
  value = value.transpose(1, 2);
  // Apply RoPE
  query = q_rope_(query, sin_emb, cos_emb);
  key = k_rope_(key, sin_emb, cos_emb);

  return {hidden_states, query, key, value};
}

KV2H::KV2H(Qwen3Decoder& decoder, GenerationState& state)
    : layer_idx_(decoder.layer_idx_),
      head_dim_(decoder.head_dim_),
      num_attention_heads_(decoder.num_attention_heads_),
      o_proj_(decoder.o_proj_),
      mask_(decoder.mask_),
      softmax_(decoder.softmax_),
      mlp_(decoder.mlp_),
      post_attention_layer_norm_(decoder.post_attention_layer_norm_),
      state_(state),
      parameter_loader_(decoder.parameter_loader_) {}

void KV2H::load(const ParameterFile::ptr_t &param_file) {
  o_proj_.impl()->load(param_file);
  mask_.impl()->load(param_file);
  softmax_.impl()->load(param_file);
  mlp_.load(param_file);
  post_attention_layer_norm_.impl()->load(param_file);
  pulled_ = true;
}

std::vector<Tensor> KV2H::forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) {
  const auto& hidden_states = inputs[0];
  auto query = inputs[1].clone();
  const auto& key = inputs[2];
  int offset = args[0].get<int>();
  const int count = hidden_states.shape()[1];

  if (!pulled_) {
    load(parameter_loader_.getParameterFile());
  }

  auto [cached_key, cached_value] = state_.getKV({layer_idx_, 0, offset + count});
  MLLM_RT_ASSERT(cached_key.dtype() == kFloat32);

  // Compute attention scores with scaling
  const float scale = 1.f / sqrtf(static_cast<float>(head_dim_));
  Tensor attn = softmax_(mask_(nn::functional::matmul(query, cached_key, false, true) * scale));
  auto output = nn::functional::matmul(attn, cached_value);
  output = output.transpose(1, 2).view({1, count, num_attention_heads_ * head_dim_});
  auto attn_output = o_proj_(output);
  auto residual = attn_output + hidden_states;

  // MLP with residual
  auto mlp_output = mlp_(post_attention_layer_norm_(residual))[0];
  auto result = mlp_output + residual;

  return {result};
}

/** Qwen3Text Implementation **/

Qwen3Text::Qwen3Text(const std::string& name, const Qwen3Config& cfg, GenerationState& state, ParameterLoader& parameter_loader,
                     int chunk_size)
    : nn::Module(name),
      parameter_loader_(parameter_loader),
      chunk_size_(chunk_size),
      num_layers_(cfg.num_hidden_layers),
      state_(state) {
  decode_blocks_ = reg<nn::ModuleListWithIdx<Qwen3Decoder>>("layers", num_layers_, cfg, state, parameter_loader);
  for (auto& block : decode_blocks_.list()) {
    h2kv_.emplace_back(block, state_);
    kv2h_.emplace_back(block, state_);
  }
  norm_ = reg<nn::RMSNorm>("norm", cfg.rms_norm_eps);
  embedding_ = reg<nn::Embedding>("embed_tokens", cfg.vocab_size, cfg.hidden_size);

  for (int i = 0; i < num_layers_; ++i) {
    parameter_loader_.registerLayer(i, collectLayerParamNames(i));
  }
}

void Qwen3Text::loadMiscParams() {
  auto prefix = getModuleName() + ".";
  parameter_loader_.loadTensor(prefix + "embed_tokens.weight");
  parameter_loader_.loadTensor(prefix + "norm.weight");
  norm_.impl()->load(parameter_loader_.getParameterFile());
  embedding_.impl()->load(parameter_loader_.getParameterFile());
  // each layer's parameters are loaded on-demand
}

std::vector<std::string> Qwen3Text::collectLayerParamNames(int layer) const {
  auto prefix = getModuleName() + ".layers." + std::to_string(layer) + ".";
  return {
      prefix + "self_attn.q_proj.weight",
      prefix + "self_attn.k_proj.weight",
      prefix + "self_attn.v_proj.weight",
      prefix + "self_attn.o_proj.weight",
      prefix + "self_attn.q_norm.weight",
      prefix + "self_attn.k_norm.weight",
      prefix + "input_layernorm.weight",
      prefix + "post_attention_layernorm.weight",
      prefix + "mlp.gate_proj.weight",
      prefix + "mlp.up_proj.weight",
      prefix + "mlp.down_proj.weight",
  };
}

std::vector<Tensor> Qwen3Text::forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) {
  const auto& token_ids = inputs[0];
  const auto& sin_emb = inputs[1];
  const auto& cos_emb = inputs[2];
  auto offset = args[0].get<int>();
  Tensor output = offset == 0 ? prefill_(token_ids, sin_emb, cos_emb) : decode_(token_ids, sin_emb, cos_emb, offset);
  return {output};
}

Tensor Qwen3Text::prefill_(const Tensor& token_ids, const Tensor& sin_emb, const Tensor& cos_emb) {
  auto seq_len = token_ids.shape()[1];
  auto num_chunks = static_cast<int>((seq_len + chunk_size_ - 1) / chunk_size_);

  RuntimeContext ctx(state_, parameter_loader_);

  std::vector<LayerContext> layers;
  layers.reserve(num_layers_);
  for (int i = 0; i < num_layers_; ++i) {
    layers.push_back({i, &h2kv_[i], &kv2h_[i]});
  }

  std::vector<ChunkContext> chunks;
  for (int i = 0; i < num_chunks; ++i) {
    int chunk_start = i * chunk_size_;
    int chunk_end = std::min(chunk_start + chunk_size_, seq_len);
    auto sin_chunk = sin_emb[{kAll, {chunk_start, chunk_end}, kAll}];
    auto cos_chunk = cos_emb[{kAll, {chunk_start, chunk_end}, kAll}];
    chunks.emplace_back(i, chunk_start, chunk_end, sin_chunk, cos_chunk);

    if (state_.getMinWatermark(chunk_start, chunk_end - chunk_start) < 0) {
      auto h0 = embedding_(token_ids[{kAll, {chunk_start, chunk_end}}]);
      state_.updateH({0, chunk_start, chunk_end - chunk_start}, h0);
    }
  }


  auto scheduler = std::make_unique<SimpleGridScheduler>(layers, chunks, ctx);

  scheduler->initTasks();

  scheduler->run();

  return norm_(state_.getH({num_layers_, 0, seq_len}));
}

Tensor Qwen3Text::decode_(const Tensor& token_ids, const Tensor& sin_emb, const Tensor& cos_emb, int token_idx) {
  if (state_.isPositionComplete(token_idx)) { return norm_(state_.getH({static_cast<int>(num_layers_), token_idx, 1})); }

  MLLM_RT_ASSERT_EQ(token_ids.shape()[1], 1);
  auto x = embedding_(token_ids);
  state_.updateH({0, token_idx, 1}, x);
  for (size_t j = 0; j < num_layers_; ++j) {
    recordEvent<LayerBeginEvent>(j, 1, token_idx);
    auto h2kv_result = h2kv_[j](x, sin_emb, cos_emb);
    auto& h = h2kv_result[0];
    auto& q = h2kv_result[1];
    auto& k = h2kv_result[2];
    auto& v = h2kv_result[3];
    state_.updateKV({static_cast<int>(j), token_idx, 1}, k, v);
    x = kv2h_[j](h, q, k, AnyValue(token_idx))[0];
    recordEvent<LayerCompleteEvent>(j, 1, token_idx);
    state_.updateH({static_cast<int>(j + 1), token_idx, 1}, x);
  }
  auto output = norm_(x);
  state_.checkpoint();
  return output;
}

/* ARGeneration Implementation */

Qwen3IntermittentForCausalLM::Qwen3IntermittentForCausalLM(const Qwen3Config& cfg,
                                                           GenerationState& state,
                                                           ParameterLoader& parameter_loader,
                                                           int chunk_size)
    : cfg_(cfg),
      state_(state),
      parameter_loader_(parameter_loader),
      llm_(reg<Qwen3Text>("model", cfg_, state_, parameter_loader_, chunk_size)),
      lm_head_(reg<nn::Linear>("lm_head_out", cfg_.hidden_size, cfg_.vocab_size, false,
                               cfg_.linear_impl_type)) {
  MLLM_INFO("Initializing intermittent Qwen3 model");

  eos_token_id_ = cfg.end_of_text_token_id;
  max_length_ = cfg.max_cache_length;
  tie_word_embeddings_ = cfg.tie_word_embeddings;

  MLLM_RT_ASSERT(tie_word_embeddings_);  //  "For simplicity, tie_word_embeddings_ must be true"
  registerBuffer("inv_freq", makeRoPEInvFreq(cfg.head_dim, cfg.rope_theta));
}

void Qwen3IntermittentForCausalLM::loadMinimalParams() {
  // Load embedding, norm, and lm_head at startup (layers are loaded on-demand)
  llm_.loadMiscParams();
  parameter_loader_.loadTensor("lm_head_out.weight");
  lm_head_.impl()->load(parameter_loader_.getParameterFile());
}


void Qwen3IntermittentForCausalLM::loadAllParams() {
  load(parameter_loader_.getParameterFile());
}

ARGenerationOutputPast Qwen3IntermittentForCausalLM::forward(const ARGenerationOutputPast& input,
                                                             const ARGenerationArgs& args) {
  auto sequence = input.at("sequence");
  auto position_ids = input.at("position_ids");

  MLLM_RT_ASSERT_EQ(sequence.shape()[0], 1); // currently only support batch size 1

  auto [llm_embedding_sin, llm_embedding_cos] =
      makeRotaryPosEmbedding(position_ids, getBuffer("inv_freq"), 1.0f);
  int offset = *position_ids.cptrAt<int64_t>({0, 0});
  sequence = llm_(sequence, llm_embedding_sin, llm_embedding_cos, AnyValue(offset))[0];

  // get output (last token)
  auto S = sequence.shape()[1];
  sequence = sequence[{kAll, {S - 1}, kAll}];
  sequence = lm_head_(sequence);

  return {
      {"sequence", sequence},
      {"position_ids", position_ids},
  };
}

void Qwen3IntermittentForCausalLM::streamGenerate(const ARGenerationOutputPast& input,
                                                  const ARGenerationArgs& args,
                                                  const std::function<void(int64_t)>& callback) {
  GenConfig gencfg = makeGenConfig(args);

  auto do_forward = [&, this](ARGenerationOutputPast& past) {
    past = forward(past, args);
    int64_t next_token = predictNextToken(past["sequence"], gencfg);
    callback(next_token);
    return next_token;
  };

  ARGenerationOutputPast prefill_input = input;
  int prefill_size = input.at("sequence").shape()[1];
  prefill_input["position_ids"] = makePositionIds(0, prefill_size);
  prefillEventStartTimePoint();
  int64_t next_token = do_forward(prefill_input);
  ar_prefill_tokens_ = prefill_size;
  prefillEventEndTimePoint();

  ARGenerationOutputPast decode_input = prefill_input;
  decodeEventStartTimePoint();
  for (int i = 0; i < gencfg.max_decode_tokens; ++i, ++ar_steps_) {
    if (!gencfg.ignore_eos && next_token == gencfg.eos_token_id) { break; }
    decode_input["sequence"] = makeDecodeSequence(next_token);
    decode_input["position_ids"] = makePositionIds(prefill_size + i, 1);
    next_token = do_forward(decode_input);
  }
  decodeEventEndTimePoint();
}

Qwen3IntermittentForCausalLM::GenConfig Qwen3IntermittentForCausalLM::makeGenConfig(
    const ARGenerationArgs& args) {
  auto cfg = GenConfig{
      .temperature = args.count("temperature") ? args.at("temperature").get<float>() : 1.0f,
      .top_k = args.count("top_k") ? args.at("top_k").get<int>() : 0,
      .top_p = args.count("top_p") ? args.at("top_p").get<float>() : 0.0f,
      .max_length = args.count("max_length") ? args.at("max_length").get<int>() : max_length_,
      .eos_token_id =
          args.count("eos_token_id") ? args.at("eos_token_id").get<int>() : eos_token_id_,
      .do_sample = args.count("do_sample") ? args.at("do_sample").get<bool>() : do_sample_,
      .max_decode_tokens =
          args.count("max_decode_tokens") && args.at("max_decode_tokens").get<int>() > 0
              ? args.at("max_decode_tokens").get<int>()
              : max_length_,
      .ignore_eos = args.count("ignore_eos") ? args.at("ignore_eos").get<bool>() : false,
  };
  cfg.use_sampling =
      cfg.do_sample || (cfg.temperature != 1.0f) || (cfg.top_k > 0) || (cfg.top_p > 0.0f);
  return cfg;
}

int64_t Qwen3IntermittentForCausalLM::predictNextToken(Tensor& logits, GenConfig& cfg) {
  if (!cfg.use_sampling) { return sampleGreedy(logits); }
  if (cfg.top_k > 0) {
    return sampleTopK(logits, cfg.top_k, cfg.temperature);
  } else if (cfg.top_p > 0.0f) {
    return sampleTopP(logits, cfg.top_p, cfg.temperature);
  } else {
    return sampleTemperature(logits, cfg.temperature);
  }
}

Tensor Qwen3IntermittentForCausalLM::makePositionIds(int start_pos, int count) {
  auto position_ids = Tensor::empty({1, count}, kInt64, kCPU).alloc();
  for (int s = 0; s < count; ++s) { *position_ids.ptrAt<int64_t>({0, s}) = start_pos + s; }
  return position_ids;
}

Tensor Qwen3IntermittentForCausalLM::makeDecodeSequence(int64_t token) {
  auto sequence = Tensor::empty({1, 1}, kInt64, kCPU).alloc();
  *sequence.ptrAt<int64_t>({0, 0}) = token;
  return sequence;
}

}  // namespace mllm::models::qwen3_i
