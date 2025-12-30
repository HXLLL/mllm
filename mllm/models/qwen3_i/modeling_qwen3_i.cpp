// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/models/qwen3_i/modeling_qwen3_i.hpp"
#include "mllm/models/qwen3_i/qwen3_events.hpp"
#include "mllm/nn/Functional.hpp"
#include "mllm/utils/AnyValue.hpp"
#include "mllm/utils/Enumerate.hpp"
#include <algorithm>
#include <cmath>
#include <nlohmann/json.hpp>

namespace mllm::models::qwen3_i {

/** RoPE utilities **/

static Tensor makeRoPEInvFreq(int output_dim, float rope_theta) {
  auto inv_freq = Tensor::empty({output_dim / 2}, kFloat32, kCPU).alloc();
  auto* ptr = inv_freq.ptr<float>();
  for (int i = 0; i < output_dim / 2; i++) {
    ptr[i] = 1.0f / std::pow(rope_theta, 2.0f * i / output_dim);
  }
  return inv_freq;
}

static std::pair<Tensor, Tensor> makeRotaryPosEmbedding(const Tensor& position_ids, const Tensor& inv_freq,
                                                        float attention_scaling) {
  const auto batch_size = position_ids.shape()[0];
  const auto seq_len = position_ids.shape()[1];
  const auto inv_freq_len = inv_freq.shape()[0];
  const auto dim = inv_freq_len * 2;

  // Create sin/cos embeddings with [freqs, freqs] pattern
  auto sin_emb = Tensor::empty({batch_size, seq_len, dim}, kFloat32, kCPU).alloc();
  auto cos_emb = Tensor::empty({batch_size, seq_len, dim}, kFloat32, kCPU).alloc();
  auto* sin_ptr = sin_emb.ptr<float>();
  auto* cos_ptr = cos_emb.ptr<float>();
  const auto* position_ids_ptr = position_ids.ptr<int64_t>();
  const auto* inv_freq_ptr = inv_freq.ptr<float>();

  // Compute frequencies and sin/cos in a single pass
  for (int b = 0; b < batch_size; ++b) {
    for (int s = 0; s < seq_len; ++s) {
      const auto pos = static_cast<float>(position_ids_ptr[b * seq_len + s]);
      const int base_idx = b * seq_len * dim + s * dim;
      
      for (int d = 0; d < inv_freq_len; ++d) {
        const auto freq = pos * inv_freq_ptr[d];
        const auto sin_val = std::sin(freq) * attention_scaling;
        const auto cos_val = std::cos(freq) * attention_scaling;

        // Store to both halves (RoPE [freqs, freqs] pattern)
        sin_ptr[base_idx + d] = sin_val;
        sin_ptr[base_idx + d + inv_freq_len] = sin_val;
        cos_ptr[base_idx + d] = cos_val;
        cos_ptr[base_idx + d + inv_freq_len] = cos_val;
      }
    }
  }

  return {sin_emb, cos_emb};
}

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

Qwen3Decoder::Qwen3Decoder(const std::string& name, const Qwen3Config& cfg) : nn::Module(name) {
  // Initialize attention configuration
  hidden_size_ = cfg.hidden_size;
  num_attention_heads_ = cfg.num_attention_heads;
  num_key_value_heads_ = cfg.num_key_value_heads;
  head_dim_ = cfg.head_dim;
  num_key_value_groups_ = num_attention_heads_ / num_key_value_heads_;

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
  const auto& hidden_states = inputs[0];
  const auto& sin_emb = inputs[1];
  const auto& cos_emb = inputs[2];
  const auto& position_ids = inputs[3];
  auto state = args[0].get<GenerationState::ptr>();

  const int token_cnt = position_ids.shape()[1];
  const int token_offset = *position_ids.cptrAt<int64_t>({0, 0});

  const int B = hidden_states.shape()[0];
  const int S = hidden_states.shape()[1];

  // Apply input layer norm
  auto x = input_layer_norm_(hidden_states);
  // Project to Q, K, V
  auto query = q_proj_(x).view({B, S, num_attention_heads_, head_dim_});
  auto key = k_proj_(x).view({B, S, num_key_value_heads_, head_dim_});
  auto value = v_proj_(x).view({B, S, num_key_value_heads_, head_dim_});
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

  // TODO: consider task split point
  state->update_kv(layer_idx_, token_offset, token_cnt, key, value);
  Context::instance().tracer()->record<KVCacheCompleteEvent>(layer_idx_, S, 0);

  auto kv_cache = state->get_kv(layer_idx_, 0, token_cnt + token_offset);
  MLLM_RT_ASSERT(kv_cache);
  auto [cached_key, cached_value] = *kv_cache;

  // Compute attention scores with scaling
  const float scale = 1.f / sqrtf(static_cast<float>(head_dim_));
  Tensor attn = (cached_key.dtype() == kFloat32)
      ? softmax_(mask_(nn::functional::matmul(query, cached_key, false, true) * scale))
      : softmax_(mask_(nn::functional::matmul(query.to(kFloat32), cached_key.to(kFloat32), false, true) * scale)).to(kFloat16);
  auto output = nn::functional::matmul(attn, cached_value);
  output = output.transpose(1, 2).view({B, S, num_attention_heads_ * head_dim_});
  auto attn_output = o_proj_(output);
  recordEvent<SelfAttentionCompleteEvent>(layer_idx_, S, 0);
  auto residual = attn_output + hidden_states;
  // MLP with residual
  recordEvent<MLPBeginEvent>(layer_idx_, S, 0);
  auto mlp_output = mlp_(post_attention_layer_norm_(residual))[0];
  recordEvent<MLPCompleteEvent>(layer_idx_, S, 0);

  return {mlp_output + residual};
}

/** Qwen3Text Implementation **/

Qwen3Text::Qwen3Text(const std::string& name, const Qwen3Config& cfg) : nn::Module(name) {
  decode_blocks_ = reg<nn::ModuleList<Qwen3Decoder>>("layers", cfg.num_hidden_layers, cfg);
  for (auto [idx, block] : enumerate(decode_blocks_.list())) {
    block.layer_idx_ = idx;
  }
  norm_ = reg<nn::RMSNorm>("norm", cfg.rms_norm_eps);
  embedding_ = reg<nn::Embedding>("embed_tokens", cfg.vocab_size, cfg.hidden_size);
}

class Task {

};

class H2KVTask : public Task {

};

class KV2HTask : public Task {
  Qwen3MLP &mlp_;

};

class ChunkLayerTask : public Task {
};


std::vector<Tensor> Qwen3Text::forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) {
  const auto& token_ids = inputs[0];
  const auto& sin_emb = inputs[1];
  const auto& cos_emb = inputs[2];
  const auto& position_ids = inputs[3];
  auto state = args[0].get<GenerationState::ptr>();
  auto token_idx = *position_ids.cptrAt<int64_t>({0, 0});

  const int seq_len = token_ids.shape()[1];
  const int num_chunks = (seq_len + chunksize_ - 1) / chunksize_;
  
  std::vector<Tensor> chunk_outputs;
  chunk_outputs.reserve(num_chunks);

  // Chunked prefill
  for (int chunk_idx = 0; chunk_idx < num_chunks; chunk_idx++) {
    const int chunk_start = chunk_idx * chunksize_;
    const int chunk_end = std::min(chunk_start + chunksize_, seq_len);
    const int chunk_len = chunk_end - chunk_start;

    // Embed and slice position encodings for this chunk
    auto x = embedding_(token_ids[{kAll, {chunk_start, chunk_end}}]);
    auto sin_chunk = sin_emb[{kAll, {chunk_start, chunk_end}, kAll}];
    auto cos_chunk = cos_emb[{kAll, {chunk_start, chunk_end}, kAll}];

    // Process through all decoder layers
    for (int i = 0; i < static_cast<int>(decode_blocks_.list().size()); i++) {
      recordEvent<LayerBeginEvent>(i, chunk_len, token_idx);
      x = decode_blocks_.list()[i](x, sin_chunk, cos_chunk, position_ids, AnyValue(state))[0];
      recordEvent<LayerCompleteEvent>(i, chunk_len, token_idx);
    }

    chunk_outputs.push_back(x);
    token_idx += chunk_len;

    // Sync KV cache to disk after each chunk
    recordEvent<SyncStartEvent>(0, chunk_len, token_idx);
    state->sync_cache(); // TODO: FIX THIS
    recordEvent<SyncCompleteEvent>(0, chunk_len, token_idx);
  }

  auto output = nn::functional::concat(chunk_outputs, 1);
  return {norm_(output)};
}

// ============================================================================
// Qwen3IntermittentForCausalLM Implementation
// ============================================================================

Qwen3IntermittentForCausalLM::Qwen3IntermittentForCausalLM(const Qwen3Config& cfg, const std::filesystem::path& state_dir)
    : cfg_(cfg) {
  MLLM_INFO("Initializing intermittent Qwen3 model");
  state_ = GenerationState::create_or_recover(cfg, state_dir);

  // Initialize model components
  eos_token_id_ = cfg_.end_of_text_token_id;
  max_length_ = cfg_.max_cache_length;
  tie_word_embeddings_ = cfg_.tie_word_embeddings;

  llm_ = reg<Qwen3Text>("model", cfg_);

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

  // ========== 3. 生成位置编码 (Position IDs) ==========
  // 根据是预填充阶段还是解码阶段，生成或更新位置编码
  Tensor position_ids = Tensor::nil();
  if (input.count("position_ids")) {
    // ========== 3.1 解码阶段 ==========
    // 如果输入中已有 position_ids，说明是增量解码阶段
    // 需要基于之前的位置继续递增
    position_ids = input.at("position_ids");

    // 对于单 token 解码（seq_len == 1），将最后一个位置加 1
    if (seq_len == 1) {
      auto last_pos = *position_ids.offsettedPtr<int64_t>({0, position_ids.shape()[1] - 1});
      position_ids = Tensor::empty({batch_size, 1}, kInt64, kCPU).alloc();
      *position_ids.offsettedPtr<int64_t>({0, 0}) = last_pos + 1;
    }
  } else {
    // ========== 3.2 预填充阶段 ==========
    // 如果是首次输入（预填充阶段），生成从 0 开始的位置编码
    // 位置编码: [0, 1, 2, ..., seq_len-1]
    position_ids = Tensor::empty({batch_size, seq_len}, kInt64, kCPU).alloc();
    auto position_ids_ptr = position_ids.ptr<int64_t>();
    for (int b = 0; b < batch_size; ++b) {
      for (int s = 0; s < seq_len; ++s) { position_ids_ptr[b * seq_len + s] = s; }
    }
  }

  // ========== 4. 生成 RoPE 位置编码 ==========
  // 基于位置编码和预计算的逆频率，生成旋转位置编码的正弦和余弦值
  // 这些值将用于在注意力计算中注入位置信息
  // 输出形状: [B, S, head_dim] (sin 和 cos 各一个)
  auto [llm_embedding_sin, llm_embedding_cos] = makeRotaryPosEmbedding(position_ids, getBuffer("inv_freq"), 1.0f);

  // ========== 5. 通过 Transformer 模型前向传播 ==========
  // 将 token 序列、RoPE 编码和 KV cache 传入模型
  // 同时传递 token_counter 用于记录每层完成时间
  // 模型会依次通过：嵌入层 -> 多个解码器层 -> 最终归一化
  // 输出形状: [B, S, hidden_size]
  sequence = llm_(sequence, llm_embedding_sin, llm_embedding_cos, position_ids, AnyValue(state_))[0];

  // ========== 6. 提取最后一个 token 的表示 ==========
  // 对于自回归生成，通常只需要最后一个位置的隐藏状态
  // 这样可以减少计算量，因为后续的 LM head 只需要预测下一个 token
  // 输出形状: [B, 1, hidden_size]
  {
    auto S = sequence.shape()[1];
    sequence = sequence[{kAll, {S - 1}, kAll}];
  }

  // ========== 7. 语言模型头投影 ==========
  // 如果启用了词嵌入共享（tie_word_embeddings），将隐藏状态投影到词汇表大小
  // 输出形状: [B, 1, vocab_size]
  // 注意：如果未启用共享，则需要在外部调用 lm_head
  if (tie_word_embeddings_) { sequence = lm_head_(sequence); }

  // ========== 8. 返回输出 ==========
  // 返回更新后的序列表示和位置编码，供下一轮生成使用
  return {
      {"sequence", sequence},
      {"position_ids", position_ids},
  };
}

}  // namespace mllm::models::qwen3_i
