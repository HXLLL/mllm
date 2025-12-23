// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/models/qwen3_i/modeling_qwen3_i.hpp"
#include "mllm/nn/Functional.hpp"
#include "mllm/nn/lmcache/PersistentCache.hpp"
#include "mllm/utils/Enumerate.hpp"
#include "mllm/utils/Tracing.hpp"
#include <algorithm>
#include <cmath>

namespace mllm::models::qwen3_i {

template<typename Derived>
class LayerEvent : public Event {
 public:
  LayerEvent(int layer_idx, int seq_len, int token_idx) : layer_idx_(layer_idx), seq_len_(seq_len), token_idx_(token_idx) {}
  [[nodiscard]] std::map<std::string, std::string> toData() const override {
    return {{"layer_idx", std::to_string(layer_idx_)},
            {"seq_len", std::to_string(seq_len_)},
            {"token_idx", std::to_string(token_idx_)}};
  }
  [[nodiscard]] const char* typeName() const noexcept override {
    return Derived::kTypeName;
  }

 private:
  int layer_idx_;
  int seq_len_;
  int token_idx_;
};

struct LayerBeginEvent final : public LayerEvent<LayerBeginEvent> {
  using LayerEvent::LayerEvent;
  static constexpr const char* kTypeName = "LayerBegin";
};
struct LayerCompleteEvent final : public LayerEvent<LayerCompleteEvent> {
  using LayerEvent::LayerEvent;
  static constexpr const char* kTypeName = "LayerComplete";
};
struct KVCacheCompleteEvent final : public LayerEvent<KVCacheCompleteEvent> {
  using LayerEvent::LayerEvent;
  static constexpr const char* kTypeName = "KVCacheComplete";
};
struct SelfAttentionCompleteEvent final : public LayerEvent<SelfAttentionCompleteEvent> {
  using LayerEvent::LayerEvent;
  static constexpr const char* kTypeName = "SelfAttentionComplete";
};
struct MLPBeginEvent final : public LayerEvent<MLPBeginEvent> {
  using LayerEvent::LayerEvent;
  static constexpr const char* kTypeName = "MLPBegin";
};
struct MLPCompleteEvent final : public LayerEvent<MLPCompleteEvent> {
  using LayerEvent::LayerEvent;
  static constexpr const char* kTypeName = "MLPComplete";
};
struct SyncStartEvent final : public LayerEvent<SyncStartEvent> {
  using LayerEvent::LayerEvent;
  static constexpr const char* kTypeName = "SyncStart";
};
struct SyncCompleteEvent final : public LayerEvent<SyncCompleteEvent> {
  using LayerEvent::LayerEvent;
  static constexpr const char* kTypeName = "SyncComplete";
};

template< typename EventType>
static inline void record_event(int layer_idx, int seq_len, int token_idx) {
  Context::instance().tracer()->record<EventType>(layer_idx, seq_len, token_idx);
}

static Tensor makeRoPEInvFreq(int output_dim, float rope_theta) {
  auto inv_freq = Tensor::empty({output_dim / 2}, kFloat32, kCPU).alloc();
  auto inv_freq_ptr = inv_freq.ptr<float>();
  for (int i = 0; i < output_dim / 2; i++) { inv_freq_ptr[i] = 1.0 / std::pow(rope_theta, 2.0 * i / output_dim); }
  return inv_freq;
}

static std::pair<Tensor, Tensor> makeRotaryPosEmbedding(Tensor& position_ids, const Tensor& inv_freq, float attention_scaling) {
  // ========== 1. 获取维度信息 ==========
  auto batch_size = position_ids.shape()[0];  // 批次大小
  auto seq_len = position_ids.shape()[1];     // 序列长度
  auto inv_freq_len = inv_freq.shape()[0];    // 逆频率向量长度（通常是 dim/2）
  auto dim = inv_freq_len * 2;                // 完整维度（head_dim）

  // ========== 2. 计算频率矩阵 ==========
  // 将位置编码与逆频率向量相乘，得到每个位置、每个维度的频率
  // 计算公式: freqs[b, s, d] = position_ids[b, s] * inv_freq[d]
  // 输出形状: [B, S, inv_freq_len]
  auto freqs = Tensor::empty({batch_size, seq_len, inv_freq_len}, kFloat32, kCPU).alloc();
  auto freqs_ptr = freqs.ptr<float>();
  auto position_ids_ptr = position_ids.ptr<int64_t>();
  auto inv_freq_ptr = inv_freq.ptr<float>();

  // 计算频率矩阵（相当于广播乘法）
  for (int b = 0; b < batch_size; ++b) {
    for (int s = 0; s < seq_len; ++s) {
      auto pos = position_ids_ptr[b * seq_len + s];
      for (int d = 0; d < inv_freq_len; ++d) {
        freqs_ptr[b * seq_len * inv_freq_len + s * inv_freq_len + d] = static_cast<float>(pos) * inv_freq_ptr[d];
      }
    }
  }

  // ========== 3. 创建正弦和余弦嵌入张量 ==========
  // 为每个位置、每个维度计算 sin 和 cos 值
  // 输出形状: [B, S, dim]
  auto sin_emb = Tensor::empty({batch_size, seq_len, dim}, kFloat32, kCPU).alloc();
  auto cos_emb = Tensor::empty({batch_size, seq_len, dim}, kFloat32, kCPU).alloc();
  auto sin_ptr = sin_emb.ptr<float>();
  auto cos_ptr = cos_emb.ptr<float>();

  // ========== 4. 计算正弦和余弦值 ==========
  // 对频率矩阵应用 sin 和 cos 函数，得到旋转角度
  // 由于 RoPE 使用成对的维度进行旋转，所以将频率值复制到两个维度
  // 模式: [freqs, freqs] - 前半部分和后半部分使用相同的频率值
  for (int b = 0; b < batch_size; ++b) {
    for (int s = 0; s < seq_len; ++s) {
      for (int d = 0; d < inv_freq_len; ++d) {
        auto freq = freqs_ptr[b * seq_len * inv_freq_len + s * inv_freq_len + d];
        // 应用缩放因子并计算 sin/cos
        auto sin_val = std::sin(freq) * attention_scaling;
        auto cos_val = std::cos(freq) * attention_scaling;

        // 将相同的值存储到两个对应的维度位置（实现 [freqs, freqs] 模式）
        sin_ptr[b * seq_len * dim + s * dim + d] = sin_val;
        sin_ptr[b * seq_len * dim + s * dim + d + inv_freq_len] = sin_val;
        cos_ptr[b * seq_len * dim + s * dim + d] = cos_val;
        cos_ptr[b * seq_len * dim + s * dim + d + inv_freq_len] = cos_val;
      }
    }
  }

  return {sin_emb, cos_emb};
}

Qwen3MLP::Qwen3MLP(const std::string& name, const Qwen3Config& cfg) : nn::Module(name) {
  gate_proj_ = reg<nn::Linear>("gate_proj", cfg.hidden_size, cfg.intermediate_size, false, cfg.linear_impl_type);
  silu_ = reg<nn::SiLU>("act");
  up_proj_ = reg<nn::Linear>("up_proj", cfg.hidden_size, cfg.intermediate_size, false, cfg.linear_impl_type);
  down_proj_ = reg<nn::Linear>("down_proj", cfg.intermediate_size, cfg.hidden_size, false, cfg.linear_impl_type);
}

std::vector<Tensor> Qwen3MLP::forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) {
  auto x = gate_proj_(inputs[0]);
  x = silu_(x);
  auto y = up_proj_(inputs[0]);
  x = x * y;
  x = down_proj_(x);
  return {x};
}

Qwen3Attention::Qwen3Attention(const std::string& name, const Qwen3Config& cfg) : nn::Module(name) {
  hidden_size_ = cfg.hidden_size;
  num_attention_heads_ = cfg.num_attention_heads;
  num_key_value_heads_ = cfg.num_key_value_heads;
  head_dim_ = cfg.head_dim;
  num_key_value_groups_ = num_attention_heads_ / num_key_value_heads_;

  q_proj_ = reg<nn::Linear>("q_proj", hidden_size_, head_dim_ * num_attention_heads_, cfg.attention_bias, cfg.linear_impl_type);
  k_proj_ = reg<nn::Linear>("k_proj", hidden_size_, head_dim_ * num_key_value_heads_, cfg.attention_bias, cfg.linear_impl_type);
  v_proj_ = reg<nn::Linear>("v_proj", hidden_size_, head_dim_ * num_key_value_heads_, cfg.attention_bias, cfg.linear_impl_type);
  o_proj_ = reg<nn::Linear>("o_proj", head_dim_ * num_attention_heads_, hidden_size_, cfg.attention_bias, cfg.linear_impl_type);

  rms_norm_q_ = reg<nn::RMSNorm>("q_norm", cfg.rms_norm_eps);
  rms_norm_k_ = reg<nn::RMSNorm>("k_norm", cfg.rms_norm_eps);

  q_rope_ = reg<nn::RoPE>("q_rope", cfg.rope_theta, cfg.max_position_embeddings);
  k_rope_ = reg<nn::RoPE>("k_rope", cfg.rope_theta, cfg.max_position_embeddings);

  mask_ = reg<nn::CausalMask>("mask");
  softmax_ = reg<nn::Softmax>("softmax", -1);
}
std::vector<Tensor> Qwen3Attention::forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) {
  auto x = inputs[0];
  auto llm_embedding_sin = inputs[1];
  auto llm_embedding_cos = inputs[2];
  auto past_kv_cache = args[0].get<nn::PersistentCache*>();

  // ========== 2. 线性投影生成 Q、K、V ==========
  // 通过三个独立的线性层将输入投影为 Query、Key、Value
  // 输出形状: [B, S, H * D] (对于 Q) 或 [B, S, KV_H * D] (对于 K、V)
  auto query_states = q_proj_(x);
  auto key_states = k_proj_(x);
  auto value_states = v_proj_(x);

  int B = inputs[0].shape()[0];  // Batch size: 批次大小
  int S = inputs[0].shape()[1];  // Sequence length: 序列长度

  // ========== 4. Reshape 为多头格式 ==========
  // 将 Q、K、V 从 [B, S, H*D] 重塑为 [B, S, H, D] 的多头格式
  // 这样每个头可以独立计算注意力
  // 注意：K 和 V 使用 num_key_value_heads_（可能小于 num_attention_heads_，支持分组查询注意力）
  query_states = query_states.view({B, S, num_attention_heads_, head_dim_});
  key_states = key_states.view({B, S, num_key_value_heads_, head_dim_});
  value_states = value_states.view({B, S, num_key_value_heads_, head_dim_});

  // [B, S, H, D]
  query_states = rms_norm_q_(query_states);
  key_states = rms_norm_k_(key_states);

  // ========== 6. 转置维度，将头维度提前 ==========
  // 从 [B, S, H, D] 转置为 [B, H, S, D]
  // 这样便于后续的矩阵乘法操作（每个头独立计算）
  query_states = query_states.transpose(1, 2);
  key_states = key_states.transpose(1, 2);
  value_states = value_states.transpose(1, 2);

  // ========== 7. 应用 RoPE (Rotary Position Embedding) 位置编码 ==========
  // 对 Query 和 Key 应用旋转位置编码，将位置信息注入到注意力计算中
  // RoPE 通过旋转矩阵的方式编码位置，相比绝对位置编码更优雅
  // 输出形状保持不变: [B, H, S, D]
  query_states = q_rope_(query_states, llm_embedding_sin, llm_embedding_cos);
  key_states = k_rope_(key_states, llm_embedding_sin, llm_embedding_cos);

  // ========== 8. 更新 KV Cache ==========
  // 将当前的 K、V 状态与历史缓存合并（用于增量解码）
  // 在预填充阶段，直接存储；在解码阶段，追加到缓存末尾
  // 输出形状: [B, H, S_cache, D]，其中 S_cache 是累积的序列长度
  auto [key_states_new, value_states_new] = past_kv_cache->updateKVCache(layer_idx_, key_states, value_states);
  key_states = key_states_new;
  value_states = value_states_new;

  Context::instance().tracer()->record<KVCacheCompleteEvent>(layer_idx_, S, 0);

  // ========== 9. 计算注意力分数 ==========
  // 计算 Q @ K^T，得到注意力权重矩阵
  // 然后应用缩放因子 (1/sqrt(head_dim))，防止点积过大导致梯度消失
  // 应用因果掩码（causal mask），确保只能看到当前位置及之前的信息
  // 最后通过 softmax 归一化，得到注意力权重
  // 输出形状: [B, H, S, S_cache] (S 是当前序列长度，S_cache 是累积长度)
  Tensor attn;
  if (key_states.dtype() == kFloat32) {
    // Float32 路径：直接计算
    attn = nn::functional::matmul(query_states, key_states, false, true) * (1.f / sqrtf(head_dim_));
    attn = mask_(attn);      // 应用因果掩码
    attn = softmax_(attn);   // Softmax 归一化
  } else if (key_states.dtype() == kFloat16) {
    // Float16 路径：先转换为 Float32 计算（提高精度），再转回 Float16
    attn = nn::functional::matmul(query_states.to(kFloat32), key_states.to(kFloat32), false, true) * (1.f / sqrtf(head_dim_));
    attn = mask_(attn);
    attn = softmax_(attn);
    attn = attn.to(kFloat16);
  }

  // ========== 10. 计算注意力输出 ==========
  // 使用注意力权重对 Value 进行加权求和: attn @ V
  // 输出形状: [B, H, S, D]
  auto output = nn::functional::matmul(attn, value_states);

  // ========== 11. 重塑并输出投影 ==========
  // 将多头输出合并：先转置回 [B, S, H, D]，再 reshape 为 [B, S, H*D]
  // 最后通过输出投影层 o_proj_，将维度映射回 hidden_size
  // 输出形状: [B, S, hidden_size]
  output = output.transpose(1, 2).view({B, S, num_attention_heads_ * head_dim_});
  output = o_proj_(output);

  return {output};
}

Qwen3Decoder::Qwen3Decoder(const std::string& name, const Qwen3Config& cfg) : nn::Module(name) {
  self_attn_ = reg<Qwen3Attention>("self_attn", cfg);
  mlp_ = reg<Qwen3MLP>("mlp", cfg);
  input_layer_norm_ = reg<nn::RMSNorm>("input_layernorm", cfg.rms_norm_eps);
  post_attention_layer_norm_ = reg<nn::RMSNorm>("post_attention_layernorm", cfg.rms_norm_eps);
}

std::vector<Tensor> Qwen3Decoder::forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) {
  // ========== 1. 获取输入参数 ==========
  // inputs[0]: 输入张量，形状为 [B, S, hidden_size]
  // inputs[1]: RoPE 位置编码的正弦值
  // inputs[2]: RoPE 位置编码的余弦值
  // args[0]: KV cache，用于存储历史注意力状态
  auto llm_embedding_sin = inputs[1];
  auto llm_embedding_cos = inputs[2];
  auto& kv_cache = args[0];

  // ========== 2. Pre-Attention 归一化 ==========
  // 在自注意力计算之前，对输入进行 RMSNorm 归一化
  // 这是 Pre-Norm 架构（与 Post-Norm 相对），有助于训练稳定性
  // 输入形状: [B, S, hidden_size]
  // 输出形状: [B, S, hidden_size]
  auto x = input_layer_norm_(inputs[0]);

  // ========== 3. 自注意力计算 ==========
  // 计算多头自注意力，包括 Q、K、V 投影、RoPE 位置编码、注意力计算等
  // 输出形状: [B, S, hidden_size]
  x = self_attn_(x, llm_embedding_sin, llm_embedding_cos, kv_cache)[0];

  record_event<SelfAttentionCompleteEvent>(self_attn_.layer_idx_, inputs[0].shape()[1], 0);

  // ========== 4. 残差连接（注意力分支） ==========
  // 将注意力输出与原始输入相加，实现残差连接
  // 这有助于梯度流动和模型训练
  // 输出形状: [B, S, hidden_size]
  auto tmp = x + inputs[0];

  // ========== 5. Pre-MLP 归一化 ==========
  // 在 MLP 计算之前，对注意力输出进行归一化
  // 输入形状: [B, S, hidden_size]
  // 输出形状: [B, S, hidden_size]
  x = post_attention_layer_norm_(tmp);

  record_event<MLPBeginEvent>(self_attn_.layer_idx_, inputs[0].shape()[1], 0);
  // ========== 6. MLP 前向传播 ==========
  // 通过 MLP（多层感知机）进行非线性变换
  // MLP 包含 gate_proj、SiLU 激活、up_proj、down_proj 等操作
  // 输出形状: [B, S, hidden_size]
  x = mlp_(x)[0];

  record_event<MLPCompleteEvent>(self_attn_.layer_idx_, inputs[0].shape()[1], 0);

  // ========== 7. 残差连接（MLP 分支） ==========
  // 将 MLP 输出与注意力分支的输出相加，完成第二个残差连接
  // 最终输出形状: [B, S, hidden_size]
  x = x + tmp;

  return {x};
}

Qwen3Text::Qwen3Text(const std::string& name, const Qwen3Config& cfg) : nn::Module(name), chunksize_(1) {
  decode_blocks_ = reg<nn::ModuleList<Qwen3Decoder>>("layers", cfg.num_hidden_layers, cfg);
  for (auto [idx, b] : enumerate(decode_blocks_.list())) { b.self_attn_.layer_idx_ = idx; }
  norm_ = reg<nn::RMSNorm>("norm", cfg.rms_norm_eps);
  embedding_ = reg<nn::Embedding>("embed_tokens", cfg.vocab_size, cfg.hidden_size);
}

std::vector<Tensor> Qwen3Text::forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) {
  auto& blocks = decode_blocks_.list();

  // ========== 获取输入参数 ==========
  // inputs[0]: token IDs，形状为 [B, S]，每个元素是词汇表中的索引
  // inputs[1]: RoPE 位置编码的正弦值
  // inputs[2]: RoPE 位置编码的余弦值
  // args[0]: KV cache，用于存储所有层的注意力状态
  // args[1]: 当前 token 序号（可选）
  const auto& token_ids = inputs[0];
  const auto& llm_embedding_sin = inputs[1];
  const auto& llm_embedding_cos = inputs[2];
  auto& kv_cache = args[0];
  int token_idx = (args.size() > 1) ? args[1].get<int>() : 0;
  int seq_len = token_ids.shape()[1];

  int num_chunks = (seq_len + chunksize_ - 1) / chunksize_;
  std::vector<Tensor> chunk_outputs;
  chunk_outputs.reserve(num_chunks);

  for (int chunk_idx = 0; chunk_idx < num_chunks; chunk_idx++) {
    int chunk_start = chunk_idx * chunksize_;
    int chunk_end = std::min(chunk_start + chunksize_, seq_len);
    int chunk_len = chunk_end - chunk_start;

    auto x_chunk = embedding_(token_ids[{kAll, {chunk_start, chunk_end}}]);

    auto sin_chunk = llm_embedding_sin[{kAll, {chunk_start, chunk_end}, kAll}];
    auto cos_chunk = llm_embedding_cos[{kAll, {chunk_start, chunk_end}, kAll}];

    for (int i = 0; i < blocks.size(); i++) {
      auto& block = blocks[i];
      record_event<LayerBeginEvent>(i, chunk_len, token_idx);
      x_chunk = block(x_chunk, sin_chunk, cos_chunk, kv_cache)[0];
      record_event<LayerCompleteEvent>(i, chunk_len, token_idx);
    }

    chunk_outputs.push_back(x_chunk);
    token_idx += chunk_len;

    record_event<SyncStartEvent>(0, chunk_len, token_idx);
    if (!kv_cache.get<nn::PersistentCache*>()->sync()) {
      MLLM_ERROR_EXIT(ExitCode::kIOError, "Failed to sync kv cache");
    }
    record_event<SyncCompleteEvent>(0, chunk_len, token_idx);
  }

  auto x = nn::functional::concat(chunk_outputs, 1);

  x = norm_(x);

  return {x};
}

Qwen3IntermittentForCausalLM::Qwen3IntermittentForCausalLM(const Qwen3Config& cfg, const std::filesystem::path& cache_dir)
    : cfg(cfg) {
  MLLM_INFO("Initializing intermittent version of qwen3")
  
  // 根据目录是否存在来选择初始化 kv_cache_ 的方式
  std::filesystem::path metadata_path = cache_dir / "metadata.json";
  
  if (std::filesystem::exists(metadata_path)) {
    // 目录存在，尝试恢复缓存
    MLLM_INFO("Recovering PersistentCache from: {}", cache_dir.string());
    auto recovered_cache = nn::PersistentCache::recover(cache_dir);
    if (recovered_cache) {
      kv_cache_ = recovered_cache;
      MLLM_INFO("Successfully recovered PersistentCache");
    } else {
      MLLM_ERROR_EXIT(ExitCode::kIOError, "Failed to recover cache from {}", cache_dir.string());
    }
  } else {
    // 目录不存在，创建新的缓存
    MLLM_INFO("Creating new PersistentCache at: {}", cache_dir.string());
    kv_cache_ = std::make_shared<nn::PersistentCache>(
        cache_dir,                      // working_dir
        cfg.max_cache_length,           // max_cache_length
        cfg.num_hidden_layers,          // layer_nums
        cfg.num_attention_heads,        // q_heads
        cfg.num_key_value_heads,        // kv_heads
        cfg.head_dim,                   // kv_dims
        kFloat32,                       // k_dtype
        kFloat32,                       // v_dtype
        kCPU                            // device_type
    );
  }
  
  eos_token_id_ = cfg.end_of_text_token_id;
  max_length_ = cfg.max_cache_length;
  tie_word_embeddings_ = cfg.tie_word_embeddings;

  llm = reg<Qwen3Text>("model", cfg);

  if (cfg.tie_word_embeddings) {
    // NOTE:
    // model.lm_head.weight is quantization weights of model.embed_tokens.weight
    lm_head_ = reg<nn::Linear>("lm_head_out", cfg.hidden_size, cfg.vocab_size, false, cfg.linear_impl_type);
  }

  // Init inv freq
  auto inv = makeRoPEInvFreq(cfg.head_dim, cfg.rope_theta);
  registerBuffer("inv_freq", inv);
}

ARGenerationOutputPast Qwen3IntermittentForCausalLM::forward(const ARGenerationOutputPast& input, const ARGenerationArgs& args) {
  // ========== 1. 获取输入序列 ==========
  // 从输入字典中获取 token 序列
  // sequence 形状: [B, S]，其中每个元素是 token ID
  auto sequence = input.at("sequence");

  // ========== 2. 获取批次和序列维度信息 ==========
  // 提取批次大小和序列长度，用于后续的位置编码生成
  auto batch_size = sequence.shape()[0];
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
  sequence = llm(sequence, llm_embedding_sin, llm_embedding_cos, 
                 AnyValue(kv_cache_.get()), AnyValue(token_counter_))[0];
  token_counter_ += seq_len;

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

nn::PersistentCache& Qwen3IntermittentForCausalLM::kvCache() { return *kv_cache_; }

}  // namespace mllm::models::qwen3_i

