// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/mllm.hpp"
#include "mllm/nn/Module.hpp"
#include "mllm/nn/Nn.hpp"
#include "mllm/nn/lmcache/StaticCache.hpp"
#include "mllm/models/qwen3/configuration_qwen3.hpp"
#include "mllm/utils/Enumerate.hpp"
#include "mllm/models/ARGeneration.hpp"
#include "mllm/utils/Tracing.hpp"

namespace mllm::models::qwen3 {

// ========== 层事件基类 ==========
// 所有层相关事件的基类，包含通用的字段和方法
class LayerEventBase : public Event {
public:
  LayerEventBase(int layer_idx, int seq_len, int token_idx)
    : layer_idx_(layer_idx), seq_len_(seq_len), token_idx_(token_idx) {}

  [[nodiscard]] std::map<std::string, std::string> toData() const override {
    return {
      {"layer_idx", std::to_string(layer_idx_)},
      {"seq_len", std::to_string(seq_len_)},
      {"token_idx", std::to_string(token_idx_)}
    };
  }

protected:
  int layer_idx_;
  int seq_len_;
  int token_idx_;  // 当前是第几个 token 的推理
};

// ========== 层开始事件 ==========
class LayerBeginEvent : public LayerEventBase {
public:
  LayerBeginEvent(int layer_idx, int seq_len, int token_idx)
    : LayerEventBase(layer_idx, seq_len, token_idx) {}
  [[nodiscard]] constexpr const char* typeName() const noexcept override { return "LayerBegin"; }
};

// ========== 层完成事件 ==========
class LayerCompleteEvent : public LayerEventBase {
public:
  LayerCompleteEvent(int layer_idx, int seq_len, int token_idx)
    : LayerEventBase(layer_idx, seq_len, token_idx) {}
  [[nodiscard]] constexpr const char* typeName() const noexcept override { return "LayerComplete"; }
};

// ========== KV Cache 完成事件 ==========
class KVCacheCompleteEvent : public LayerEventBase {
public:
  KVCacheCompleteEvent(int layer_idx, int seq_len, int token_idx)
    : LayerEventBase(layer_idx, seq_len, token_idx) {}
  [[nodiscard]] constexpr const char* typeName() const noexcept override { return "KVCacheComplete"; }
};

// ========== 自注意力完成事件 ==========
class SelfAttentionCompleteEvent : public LayerEventBase {
public:
  SelfAttentionCompleteEvent(int layer_idx, int seq_len, int token_idx)
    : LayerEventBase(layer_idx, seq_len, token_idx) {}
  [[nodiscard]] constexpr const char* typeName() const noexcept override { return "SelfAttentionComplete"; }
};

// ========== MLP 开始事件 ==========
class MLPBeginEvent : public LayerEventBase {
public:
  MLPBeginEvent(int layer_idx, int seq_len, int token_idx)
    : LayerEventBase(layer_idx, seq_len, token_idx) {}
  [[nodiscard]] constexpr const char* typeName() const noexcept override { return "MLPBegin"; }
};

// ========== MLP 完成事件 ==========
class MLPCompleteEvent : public LayerEventBase {
public:
  MLPCompleteEvent(int layer_idx, int seq_len, int token_idx)
    : LayerEventBase(layer_idx, seq_len, token_idx) {}
  [[nodiscard]] constexpr const char* typeName() const noexcept override { return "MLPComplete"; }
};

// ========== 生成 RoPE 逆频率向量 ==========
// 该函数用于生成旋转位置编码（RoPE）所需的逆频率向量
// 逆频率用于计算不同维度的旋转角度，实现位置信息的编码
//
// 参数:
//   output_dim: 输出维度（通常是 head_dim）
//   rope_theta: RoPE 的基础频率参数，控制位置编码的周期
//
// 返回:
//   逆频率向量，形状为 [output_dim / 2]
//   每个元素的计算公式: 1.0 / (rope_theta ^ (2*i / output_dim))
//   其中 i 是维度索引，范围 [0, output_dim/2)
inline auto makeRoPEInvFreq(int output_dim, float rope_theta) -> Tensor {
  // 创建逆频率张量，大小为 output_dim / 2
  // 因为 RoPE 使用成对的维度进行旋转，所以只需要一半的维度
  auto inv_freq = Tensor::empty({output_dim / 2}, kFloat32, kCPU).alloc();
  auto inv_freq_ptr = inv_freq.ptr<float>();
  
  // 计算每个维度的逆频率
  // 频率随维度索引递减，使得不同维度有不同的旋转周期
  for (int i = 0; i < output_dim / 2; i++) { 
    inv_freq_ptr[i] = 1.0 / std::pow(rope_theta, 2.0 * i / output_dim); 
  }
  
  return inv_freq;
}

// ========== 生成旋转位置编码 (RoPE) ==========
// 该函数基于位置编码和逆频率向量，生成 RoPE 所需的正弦和余弦嵌入
// RoPE 通过旋转矩阵的方式将位置信息编码到 Query 和 Key 向量中
//
// 参数:
//   position_ids: 位置编码，形状为 [B, S]，每个元素是 token 的位置索引
//   inv_freq: 逆频率向量，形状为 [dim/2]，由 makeRoPEInvFreq 生成
//   attention_scaling: 注意力缩放因子，默认为 1.0
//
// 返回:
//   一对张量 (sin_emb, cos_emb)，形状均为 [B, S, dim]
//   用于后续的旋转位置编码计算
inline auto makeRotaryPosEmbedding(Tensor& position_ids, const Tensor& inv_freq,
                             float attention_scaling = 1.0) -> std::pair<Tensor, Tensor> {
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

class Qwen3MLP final : public nn::Module {
  nn::Linear gate_proj_;
  nn::Linear up_proj_;
  nn::Linear down_proj_;
  nn::SiLU silu_;

 public:
  Qwen3MLP() = default;
  Qwen3MLP(const std::string& name, const Qwen3Config& cfg) : nn::Module(name) {
    gate_proj_ = reg<nn::Linear>("gate_proj", cfg.hidden_size, cfg.intermediate_size, false, cfg.linear_impl_type);
    silu_ = reg<nn::SiLU>("act");
    up_proj_ = reg<nn::Linear>("up_proj", cfg.hidden_size, cfg.intermediate_size, false, cfg.linear_impl_type);
    down_proj_ = reg<nn::Linear>("down_proj", cfg.intermediate_size, cfg.hidden_size, false, cfg.linear_impl_type);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override;
};

class Qwen3Attention final : public nn::Module {
  nn::Linear q_proj_;
  nn::Linear k_proj_;
  nn::Linear v_proj_;
  nn::Linear o_proj_;
  nn::RMSNorm rms_norm_q_;
  nn::RMSNorm rms_norm_k_;
  nn::RoPE q_rope_;
  nn::RoPE k_rope_;
  nn::CausalMask mask_;
  nn::Softmax softmax_;

  int hidden_size_;
  int head_dim_;
  int num_attention_heads_;
  int num_key_value_heads_;
  int num_key_value_groups_;

 public:
  Qwen3Attention() = default;

  Qwen3Attention(const std::string& name, const Qwen3Config& cfg) : nn::Module(name) {
    hidden_size_ = cfg.hidden_size;
    num_attention_heads_ = cfg.num_attention_heads;
    num_key_value_heads_ = cfg.num_key_value_heads;
    head_dim_ = cfg.head_dim;
    num_key_value_groups_ = num_attention_heads_ / num_key_value_heads_;

    q_proj_ =
        reg<nn::Linear>("q_proj", hidden_size_, head_dim_ * num_attention_heads_, cfg.attention_bias, cfg.linear_impl_type);
    k_proj_ =
        reg<nn::Linear>("k_proj", hidden_size_, head_dim_ * num_key_value_heads_, cfg.attention_bias, cfg.linear_impl_type);
    v_proj_ =
        reg<nn::Linear>("v_proj", hidden_size_, head_dim_ * num_key_value_heads_, cfg.attention_bias, cfg.linear_impl_type);
    o_proj_ =
        reg<nn::Linear>("o_proj", head_dim_ * num_attention_heads_, hidden_size_, cfg.attention_bias, cfg.linear_impl_type);

    rms_norm_q_ = reg<nn::RMSNorm>("q_norm", cfg.rms_norm_eps);
    rms_norm_k_ = reg<nn::RMSNorm>("k_norm", cfg.rms_norm_eps);

    q_rope_ = reg<nn::RoPE>("q_rope", cfg.rope_theta, cfg.max_position_embeddings);
    k_rope_ = reg<nn::RoPE>("k_rope", cfg.rope_theta, cfg.max_position_embeddings);

    mask_ = reg<nn::CausalMask>("mask");
    softmax_ = reg<nn::Softmax>("softmax", -1);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override;

  int layer_idx_;
};

class Qwen3Decoder final : public nn::Module {
 public:
  Qwen3Attention self_attn_;
  Qwen3MLP mlp_;
  nn::RMSNorm input_layer_norm_;
  nn::RMSNorm post_attention_layer_norm_;

  Qwen3Decoder() = default;

  Qwen3Decoder(const std::string& name, const Qwen3Config& cfg) : nn::Module(name) {
    self_attn_ = reg<Qwen3Attention>("self_attn", cfg);
    mlp_ = reg<Qwen3MLP>("mlp", cfg);
    input_layer_norm_ = reg<nn::RMSNorm>("input_layernorm", cfg.rms_norm_eps);
    post_attention_layer_norm_ = reg<nn::RMSNorm>("post_attention_layernorm", cfg.rms_norm_eps);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override;
};

class Qwen3Text final : public nn::Module {
  nn::ModuleList<Qwen3Decoder> decode_blocks_;
  nn::RMSNorm norm_;
  nn::Embedding embedding_;
  int chunksize_;

 public:
  Qwen3Text() : chunksize_(1) {}

  Qwen3Text(const std::string& name, const Qwen3Config& cfg) : nn::Module(name), chunksize_(1) {
    decode_blocks_ = reg<nn::ModuleList<Qwen3Decoder>>("layers", cfg.num_hidden_layers, cfg);
    for (auto [idx, b] : enumerate(decode_blocks_.list())) { b.self_attn_.layer_idx_ = idx; }
    norm_ = reg<nn::RMSNorm>("norm", cfg.rms_norm_eps);
    embedding_ = reg<nn::Embedding>("embed_tokens", cfg.vocab_size, cfg.hidden_size);
  }

  void setChunkSize(int chunksize) { chunksize_ = chunksize; }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override;
};

class Qwen3ForCausalLM : public ARGeneration, public nn::Module {
 public:
  explicit Qwen3ForCausalLM(const Qwen3Config& cfg) : cfg(cfg) {
    kv_cache_ = nn::StaticCache(cfg.max_cache_length, cfg.num_hidden_layers,
                                cfg.num_attention_heads,  // q_heads
                                cfg.num_key_value_heads,  // kv_heads
                                cfg.head_dim,             // kv_dim
                                kFloat32,                 // k_dtype
                                kFloat32,                 // v_dtype
                                kCPU,                     // device_type
                                false                     // use_fa2
    );
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

  ARGenerationOutputPast forward(const ARGenerationOutputPast& input, const ARGenerationArgs& args) override;

  nn::StaticCache& kvCache();

  // 重置 token 计数器（在 clear/reset 时调用）
  void resetTokenCounter() { token_counter_ = 0; }

  // 设置 chunksize（用于控制序列分块处理的大小）
  void setChunkSize(int chunksize) { llm.setChunkSize(chunksize); }

 private:
  const Qwen3Config& cfg;
  Qwen3Text llm;
  nn::Linear lm_head_;
  bool tie_word_embeddings_;
  nn::StaticCache kv_cache_;
  int token_counter_ = 0;
};

}  // namespace mllm::models::qwen3
