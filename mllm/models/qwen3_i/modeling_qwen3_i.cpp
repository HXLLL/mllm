// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/models/qwen3_i/modeling_qwen3_i.hpp"
#include "mllm/models/qwen3_i/generation_state.hpp"
#include "mllm/models/qwen3_i/qwen3_events.hpp"
#include "mllm/nn/Functional.hpp"
#include "mllm/nn/Module.hpp"
#include "mllm/utils/AnyValue.hpp"
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
      num_attention_heads_(cfg.num_attention_heads),
      num_key_value_heads_(cfg.num_key_value_heads),
      head_dim_(cfg.head_dim),
      num_key_value_groups_(num_attention_heads_ / num_key_value_heads_),
      state_(state) {
  // Register attention components
  q_proj_ = reg<nn::Linear>("self_attn.q_proj", hidden_size_, head_dim_ * num_attention_heads_, cfg.attention_bias, cfg.linear_impl_type);
  k_proj_ = reg<nn::Linear>("self_attn.k_proj", hidden_size_, head_dim_ * num_key_value_heads_, cfg.attention_bias, cfg.linear_impl_type);
  v_proj_ = reg<nn::Linear>("self_attn.v_proj", hidden_size_, head_dim_ * num_key_value_heads_, cfg.attention_bias, cfg.linear_impl_type);
  o_proj_ = reg<nn::Linear>("self_attn.o_proj", head_dim_ * num_attention_heads_, hidden_size_, cfg.attention_bias, cfg.linear_impl_type);
  rms_norm_q_ = reg<nn::RMSNorm>("self_attn.q_norm", cfg.rms_norm_eps);
  rms_norm_k_ = reg<nn::RMSNorm>("self_attn.k_norm", cfg.rms_norm_eps);
  q_rope_ = reg<nn::RoPE>("self_attn.q_rope", cfg.rope_theta, cfg.max_position_embeddings);
  k_rope_ = reg<nn::RoPE>("self_attn.k_rope", cfg.rope_theta, cfg.max_position_embeddings);
  mask_ = reg<nn::CausalMask>("self_attn.mask");
  softmax_ = reg<nn::Softmax>("self_attn.softmax", -1);

  // Register MLP and layer norms
  mlp_ = reg<Qwen3MLP>("mlp", cfg);
  input_layer_norm_ = reg<nn::RMSNorm>("input_layernorm", cfg.rms_norm_eps);
  post_attention_layer_norm_ = reg<nn::RMSNorm>("post_attention_layernorm", cfg.rms_norm_eps);
}

std::vector<Tensor> Qwen3Decoder::forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) {
  return {};
}

/** Qwen3Text Implementation **/

H2KV::H2KV(Qwen3Decoder& decoder, GenerationState &state)
    : input_layer_norm_(decoder.input_layer_norm_),
      q_proj_(decoder.q_proj_),
      k_proj_(decoder.k_proj_),
      v_proj_(decoder.v_proj_),
      rms_norm_q_(decoder.rms_norm_q_),
      rms_norm_k_(decoder.rms_norm_k_),
      q_rope_(decoder.q_rope_),
      k_rope_(decoder.k_rope_),
      layer_idx_(decoder.layer_idx_),
      hidden_size_(decoder.hidden_size_),
      head_dim_(decoder.head_dim_),
      num_attention_heads_(decoder.num_attention_heads_),
      num_key_value_heads_(decoder.num_key_value_heads_),
      num_key_value_groups_(decoder.num_key_value_groups_),
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

KV2H::KV2H(Qwen3Decoder& decoder, GenerationState &state)
    : o_proj_(decoder.o_proj_),
      mask_(decoder.mask_),
      softmax_(decoder.softmax_),
      mlp_(decoder.mlp_),
      post_attention_layer_norm_(decoder.post_attention_layer_norm_),
      layer_idx_(decoder.layer_idx_),
      head_dim_(decoder.head_dim_),
      num_attention_heads_(decoder.num_attention_heads_),
      state_(state)
       {}

std::vector<Tensor> KV2H::forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) {
  const auto& hidden_states = inputs[0];
  auto query = inputs[1].clone();
  const auto& key = inputs[2];
  const int offset = args[0].get<int>();
  const int count = hidden_states.shape()[1];

  auto [cached_key, cached_value] = *state_.get_kv(layer_idx_, 0, offset + count);

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

Qwen3Text::Qwen3Text(const std::string& name, const Qwen3Config& cfg, GenerationState& state)
    : nn::Module(name), num_layers_(cfg.num_hidden_layers), state_(state) {
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
  const auto& position_ids = inputs[3];

  seq_len_ = token_ids.shape()[1];
  num_chunks_ = (seq_len_ + chunksize_ - 1) / chunksize_;
  
  std::vector<Tensor> chunk_outputs;
  chunk_outputs.reserve(num_chunks_);

  // Chunked prefill
  int offset = *position_ids.cptrAt<int64_t>({0, 0});
  for (int i = 0; i < num_chunks_; i++) {
    const int chunk_start = i * chunksize_;
    const int chunk_end = std::min(chunk_start + chunksize_, seq_len_);
    int len = chunk_end - chunk_start;

    // Embed and slice position encodings for this chunk
    auto x = embedding_(token_ids[{kAll, {chunk_start, chunk_end}}]);
    auto sin_chunk = sin_emb[{kAll, {chunk_start, chunk_end}, kAll}];
    auto cos_chunk = cos_emb[{kAll, {chunk_start, chunk_end}, kAll}];

    // Process through all decoder layers
    for (int j = 0; j < static_cast<int>(decode_blocks_.list().size()); j++) {
      recordEvent<LayerBeginEvent>(j, len, offset);
      auto h2kv_result = h2kv_[j](x, sin_chunk, cos_chunk);
      auto &h = h2kv_result[0];
      auto &q = h2kv_result[1];
      auto &k = h2kv_result[2];
      auto &v = h2kv_result[3];
      state_.update_kv(j, offset, len, k, v);
      x = kv2h_[j](h, q, k, AnyValue(offset))[0];
      recordEvent<LayerCompleteEvent>(j, len, offset);
    }

    chunk_outputs.push_back(x);
    offset += len;
    state_.sync_cache(); // TODO: FIX THIS
  }

  auto output = nn::functional::concat(chunk_outputs, 1);
  return {norm_(output)};
}

Qwen3IntermittentForCausalLM::Qwen3IntermittentForCausalLM(const Qwen3Config& cfg, const std::filesystem::path& state_dir)
    : cfg_(cfg), state_(GenerationState::create_or_recover(cfg, state_dir)), llm_(reg<Qwen3Text>("model", cfg_, *state_)) {
  MLLM_INFO("Initializing intermittent Qwen3 model");

  eos_token_id_ = cfg_.end_of_text_token_id;
  max_length_ = cfg_.max_cache_length;
  tie_word_embeddings_ = cfg_.tie_word_embeddings;

  if (tie_word_embeddings_) {
    lm_head_ = reg<nn::Linear>("lm_head_out", cfg_.hidden_size, cfg_.vocab_size, false, cfg_.linear_impl_type);
  }

  // Initialize RoPE inverse frequencies
  registerBuffer("inv_freq", makeRoPEInvFreq(cfg_.head_dim, cfg_.rope_theta));
}

ARGenerationOutputPast Qwen3IntermittentForCausalLM::forward(const ARGenerationOutputPast& input,
                                                             const ARGenerationArgs& args) {
  auto sequence = input.at("sequence");
  auto batch_size = sequence.shape()[0];
  assert(batch_size == 1);
  auto seq_len = sequence.shape()[1];

  Tensor position_ids = Tensor::nil();
  if (input.count("position_ids")) { // decode phase
    position_ids = input.at("position_ids");
    if (seq_len == 1) {
      auto last_pos = *position_ids.offsettedPtr<int64_t>({0, position_ids.shape()[1] - 1});
      position_ids = Tensor::empty({batch_size, 1}, kInt64, kCPU).alloc();
      *position_ids.offsettedPtr<int64_t>({0, 0}) = last_pos + 1;
    }
  } else { // prefill phase
    position_ids = Tensor::empty({batch_size, seq_len}, kInt64, kCPU).alloc();
    auto position_ids_ptr = position_ids.ptr<int64_t>();
    for (int b = 0; b < batch_size; ++b) {
      for (int s = 0; s < seq_len; ++s) { position_ids_ptr[b * seq_len + s] = s; }
    }
    state_->start_generation(sequence);
  }

  auto [llm_embedding_sin, llm_embedding_cos] = makeRotaryPosEmbedding(position_ids, getBuffer("inv_freq"), 1.0f);
  sequence = llm_(sequence, llm_embedding_sin, llm_embedding_cos, position_ids, AnyValue(state_))[0];
  {
    auto S = sequence.shape()[1];
    auto D = sequence.shape()[2];
    sequence = sequence[{kAll, {S - 1}, kAll}];
    state_->append_output_token(sequence);
  }
  if (tie_word_embeddings_) { sequence = lm_head_(sequence); }

  return {
      {"sequence", sequence},
      {"position_ids", position_ids},
  };
}

}  // namespace mllm::models::qwen3_i
