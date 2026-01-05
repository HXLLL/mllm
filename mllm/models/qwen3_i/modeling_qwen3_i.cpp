// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/models/qwen3_i/modeling_qwen3_i.hpp"
#include "mllm/models/qwen3_i/generation_state.hpp"
#include "mllm/models/qwen3_i/qwen3_events.hpp"
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

Qwen3MLP::Qwen3MLP(const std::string& name, const Qwen3Config& cfg) : nn::Module(name) {
  gate_proj_ = reg<nn::Linear>("gate_proj", cfg.hidden_size, cfg.intermediate_size, false, cfg.linear_impl_type);
  up_proj_ = reg<nn::Linear>("up_proj", cfg.hidden_size, cfg.intermediate_size, false, cfg.linear_impl_type);
  down_proj_ = reg<nn::Linear>("down_proj", cfg.intermediate_size, cfg.hidden_size, false, cfg.linear_impl_type);
  silu_ = reg<nn::SiLU>("act");
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

Qwen3Decoder::Qwen3Decoder(const std::string& name, const Qwen3Config& cfg, GenerationState& state, int idx)
    : nn::Module(name),
      layer_idx_(idx),
      hidden_size_(cfg.hidden_size),
      head_dim_(cfg.head_dim),
      num_attention_heads_(cfg.num_attention_heads),
      num_key_value_heads_(cfg.num_key_value_heads),
      num_key_value_groups_(cfg.num_attention_heads / cfg.num_key_value_heads),
      q_proj_(reg<nn::Linear>("self_attn.q_proj", cfg.hidden_size, cfg.head_dim * cfg.num_attention_heads, cfg.attention_bias, cfg.linear_impl_type)),
      k_proj_(reg<nn::Linear>("self_attn.k_proj", cfg.hidden_size, cfg.head_dim * cfg.num_key_value_heads, cfg.attention_bias, cfg.linear_impl_type)),
      v_proj_(reg<nn::Linear>("self_attn.v_proj", cfg.hidden_size, cfg.head_dim * cfg.num_key_value_heads, cfg.attention_bias, cfg.linear_impl_type)),
      o_proj_(reg<nn::Linear>("self_attn.o_proj", cfg.head_dim * cfg.num_attention_heads, cfg.hidden_size, cfg.attention_bias, cfg.linear_impl_type)),
      rms_norm_q_(reg<nn::RMSNorm>("self_attn.q_norm", cfg.rms_norm_eps)),
      rms_norm_k_(reg<nn::RMSNorm>("self_attn.k_norm", cfg.rms_norm_eps)),
      q_rope_(reg<nn::RoPE>("self_attn.q_rope", cfg.rope_theta, cfg.max_position_embeddings)),
      k_rope_(reg<nn::RoPE>("self_attn.k_rope", cfg.rope_theta, cfg.max_position_embeddings)),
      mask_(reg<nn::CausalMask>("self_attn.mask")),
      softmax_(reg<nn::Softmax>("self_attn.softmax", -1)),
      mlp_(reg<Qwen3MLP>("mlp", cfg)),
      input_layer_norm_(reg<nn::RMSNorm>("input_layernorm", cfg.rms_norm_eps)),
      post_attention_layer_norm_(reg<nn::RMSNorm>("post_attention_layernorm", cfg.rms_norm_eps)),
      state_(state) {}

std::vector<Tensor> Qwen3Decoder::forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) { return {}; }

/** Layer Tasks Implementation */

H2KV::H2KV(Qwen3Decoder& decoder, GenerationState& state)
    : layer_idx_(decoder.layer_idx_),
      hidden_size_(decoder.hidden_size_),
      head_dim_(decoder.head_dim_),
      num_attention_heads_(decoder.num_attention_heads_),
      num_key_value_heads_(decoder.num_key_value_heads_),
      num_key_value_groups_(decoder.num_key_value_groups_),
      input_layer_norm_(decoder.input_layer_norm_),
      q_proj_(decoder.q_proj_),
      k_proj_(decoder.k_proj_),
      v_proj_(decoder.v_proj_),
      rms_norm_q_(decoder.rms_norm_q_),
      rms_norm_k_(decoder.rms_norm_k_),
      q_rope_(decoder.q_rope_),
      k_rope_(decoder.k_rope_),
      state_(state) {}

std::vector<Tensor> H2KV::forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) {
  const auto& hidden_states = inputs[0];
  const auto& sin_emb = inputs[1];
  const auto& cos_emb = inputs[2];
  assert(hidden_states.shape()[0] == 1);
  const int count = hidden_states.shape()[1];

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

  return { hidden_states, query, key, value };
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
      state_(state) {}

std::vector<Tensor> KV2H::forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) {
  const auto& hidden_states = inputs[0];
  auto query = inputs[1].clone();
  const auto& key = inputs[2];
  int64_t offset = args[0].get<int64_t>();
  const int count = hidden_states.shape()[1];

  auto [cached_key, cached_value] = state_.getKV(layer_idx_, 0, offset + count);

  // Compute attention scores with scaling
  const float scale = 1.f / sqrtf(static_cast<float>(head_dim_));
  Tensor attn = (cached_key.dtype() == kFloat32)
      ? softmax_(mask_(nn::functional::matmul(query, cached_key, false, true) * scale))
      : softmax_(mask_(nn::functional::matmul(query.to(kFloat32), cached_key.to(kFloat32), false, true) * scale)).to(kFloat16);
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

Qwen3Text::Qwen3Text(const std::string& name, const Qwen3Config& cfg, GenerationState& state, int chunk_size)
    : nn::Module(name), chunk_size_(chunk_size), num_layers_(cfg.num_hidden_layers), state_(state) {
  decode_blocks_ = reg<nn::ModuleListWithIdx<Qwen3Decoder>>("layers", num_layers_, cfg, state);
  for (auto& block : decode_blocks_.list()) {
    h2kv_.emplace_back(block, state_);
    kv2h_.emplace_back(block, state_);
  }
  norm_ = reg<nn::RMSNorm>("norm", cfg.rms_norm_eps);
  embedding_ = reg<nn::Embedding>("embed_tokens", cfg.vocab_size, cfg.hidden_size);
}

std::vector<Tensor> Qwen3Text::forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) {
  const auto& token_ids = inputs[0];
  const auto& sin_emb = inputs[1];
  const auto& cos_emb = inputs[2];
  auto offset = args[0].get<int64_t>();
  Tensor output = offset == 0 ? prefill_(token_ids, sin_emb, cos_emb) : decode_(token_ids, sin_emb, cos_emb, offset);
  return {output};
}

Tensor Qwen3Text::prefill_(const Tensor& token_ids, const Tensor& sin_emb, const Tensor& cos_emb) {
  auto seq_len = token_ids.shape()[1];
  if (state_.prefill_done()) { return norm_(state_.getH(0, 0, seq_len)); }

  int64_t offset = 0;
  auto num_chunks = (seq_len + chunk_size_ - 1) / chunk_size_;
  std::vector<Tensor> chunk_outputs;
  chunk_outputs.reserve(num_chunks);

  MLLM_INFO("Prefilling [{} / {}]", offset, seq_len);
  for (int i = 0; i < num_chunks; ++i) {
    const int chunk_start = i * chunk_size_;
    const int chunk_end = std::min(chunk_start + chunk_size_, seq_len);
    int len = chunk_end - chunk_start;

    auto x = embedding_(token_ids[{kAll, {chunk_start, chunk_end}}]);
    auto sin_chunk = sin_emb[{kAll, {chunk_start, chunk_end}, kAll}];
    auto cos_chunk = cos_emb[{kAll, {chunk_start, chunk_end}, kAll}];

    state_.updateH(0, offset, len, x);
    for (size_t j = 0; j < decode_blocks_.list().size(); ++j) {
      recordEvent<LayerBeginEvent>(j, len, offset);
      auto h2kv_result = h2kv_[j](x, sin_chunk, cos_chunk);
      auto& h = h2kv_result[0];
      auto& q = h2kv_result[1];
      auto& k = h2kv_result[2];
      auto& v = h2kv_result[3];
      state_.updateKV(j, offset, len, k, v);
      x = kv2h_[j](h, q, k, AnyValue(offset))[0];
      recordEvent<LayerCompleteEvent>(j, len, offset);
      state_.updateH(j + 1, offset, len, x);
    }

    state_.set_prefill_done();
    state_.save();
    chunk_outputs.push_back(x);
    offset += len;
    MLLM_INFO("Prefilling [{} / {}]", offset, seq_len);
  }
  auto output = nn::functional::concat(chunk_outputs, 1);

  return norm_(output);
}

Tensor Qwen3Text::decode_(const Tensor& token_ids, const Tensor& sin_emb, const Tensor& cos_emb, int64_t token_idx) {
  MLLM_RT_ASSERT_EQ(token_ids.shape()[1], 1);
  MLLM_INFO("Decoding, token_idx: {}", token_idx);
  auto x = embedding_(token_ids);
  for (size_t j = 0; j < decode_blocks_.list().size(); ++j) {
    recordEvent<LayerBeginEvent>(j, 1, token_idx);
    auto h2kv_result = h2kv_[j](x, sin_emb, cos_emb);
    auto& h = h2kv_result[0];
    auto& q = h2kv_result[1];
    auto& k = h2kv_result[2];
    auto& v = h2kv_result[3];
    state_.updateKV(j, token_idx, 1, k, v);
    x = kv2h_[j](h, q, k, AnyValue(token_idx))[0];
    recordEvent<LayerCompleteEvent>(j, 1, token_idx);
  }
  state_.save();
  return norm_(x);
}

/* ARGeneration Implementation */

Qwen3IntermittentForCausalLM::Qwen3IntermittentForCausalLM(const Qwen3Config& cfg, GenerationState& state, int chunk_size)
    : cfg_(cfg), state_(state), llm_(reg<Qwen3Text>("model", cfg_, state_, chunk_size)) {
  MLLM_INFO("Initializing intermittent Qwen3 model");

  eos_token_id_ = cfg_.end_of_text_token_id;
  max_length_ = cfg_.max_cache_length;
  tie_word_embeddings_ = cfg_.tie_word_embeddings;

  if (tie_word_embeddings_) {
    lm_head_ = reg<nn::Linear>("lm_head_out", cfg_.hidden_size, cfg_.vocab_size, false, cfg_.linear_impl_type);
  }
  registerBuffer("inv_freq", makeRoPEInvFreq(cfg_.head_dim, cfg_.rope_theta));
}

ARGenerationOutputPast Qwen3IntermittentForCausalLM::forward(const ARGenerationOutputPast& input,
                                                             const ARGenerationArgs& args) {
  auto sequence = input.at("sequence");
  auto batch_size = sequence.shape()[0];
  MLLM_RT_ASSERT_EQ(batch_size, 1);
  auto seq_len = sequence.shape()[1];

  Tensor position_ids;
  if (input.count("position_ids")) {  // decode phase
    position_ids = input.at("position_ids");
    MLLM_RT_ASSERT_EQ(seq_len, 1);
    auto last_pos = *position_ids.cptrAt<int64_t>({0, position_ids.shape()[1] - 1});
    position_ids = Tensor::empty({batch_size, 1}, kInt64, kCPU).alloc();
    *position_ids.ptrAt<int64_t>({0, 0}) = last_pos + 1;
    state_.start_decode(sequence);
  } else {  // prefill phase
    position_ids = Tensor::empty({batch_size, seq_len}, kInt64, kCPU).alloc();
    for (int s = 0; s < seq_len; ++s) { *position_ids.ptrAt<int64_t>({0, s}) = s; }
    state_.start(sequence);
  }

  auto [llm_embedding_sin, llm_embedding_cos] = makeRotaryPosEmbedding(position_ids, getBuffer("inv_freq"), 1.0f);
  int64_t offset = *position_ids.cptrAt<int64_t>({0, 0});
  sequence = llm_(sequence, llm_embedding_sin, llm_embedding_cos, AnyValue(offset))[0];
  {
    auto S = sequence.shape()[1];
    auto D = sequence.shape()[2];
    sequence = sequence[{kAll, {S - 1}, kAll}];
    state_.appendOutputToken(sequence);
  }
  if (tie_word_embeddings_) { sequence = lm_head_(sequence); }

  return {
      {"sequence", sequence},
      {"position_ids", position_ids},
  };
}

void Qwen3IntermittentForCausalLM::streamGenerate(const ARGenerationOutputPast& input, const ARGenerationArgs& args,
                                                  const std::function<void(int64_t)>& callback) {
  float temperature = args.count("temperature") ? args.at("temperature").get<float>() : 1.0f;
  int top_k = args.count("top_k") ? args.at("top_k").get<int>() : 0;
  float top_p = args.count("top_p") ? args.at("top_p").get<float>() : 0.0f;
  int max_length = args.count("max_length") ? args.at("max_length").get<int>() : max_length_;
  int eos_token_id = args.count("eos_token_id") ? args.at("eos_token_id").get<int>() : eos_token_id_;
  bool do_sample = args.count("do_sample") ? args.at("do_sample").get<bool>() : do_sample_;
  bool use_sampling = do_sample || (temperature != 1.0f) || (top_k > 0) || (top_p > 0.0f);

  auto predict_next_token = [&, this](Tensor& logits) {
    if (use_sampling) {
      if (top_k > 0) {
        return sampleTopK(logits, top_k, temperature);
      } else if (top_p > 0.0f) {
        return sampleTopP(logits, top_p, temperature);
      } else {
        return sampleTemperature(logits, temperature);
      }
    } else {
      return sampleGreedy(logits);
    }
  };

  auto do_forward = [&, this](ARGenerationOutputPast& past) {
    past = forward(past, args);
    int64_t next_token = predict_next_token(past["sequence"]);
    callback(next_token);
    return next_token;
  };

  ARGenerationOutputPast prefill_input = input;
  prefillEventStartTimePoint();
  int64_t next_token = do_forward(prefill_input);
  ar_prefill_tokens_ = prefill_input["sequence"].shape()[1];
  prefillEventEndTimePoint();

  ARGenerationOutputPast decode_input = prefill_input;
  decodeEventStartTimePoint();
  for (int i = 0; i < max_length && next_token != eos_token_id; ++i, ++ar_steps_) {
    decode_input["sequence"] = Tensor::empty({1, 1}, kInt64, prefill_input["sequence"].device()).alloc();
    decode_input["sequence"].at<mllm_int64_t>({0, 0}) = next_token;
    next_token = do_forward(decode_input);
  }
  decodeEventEndTimePoint();
}

}  // namespace mllm::models::qwen3_i
