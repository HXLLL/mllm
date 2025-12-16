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

// ========== 层完成事件 ==========
// 用于记录每层完成的时间戳，便于性能分析
class LayerCompleteEvent : public Event {
public:
  LayerCompleteEvent(int layer_idx, int seq_len, int token_idx)
    : layer_idx_(layer_idx), seq_len_(seq_len), token_idx_(token_idx) {}

  [[nodiscard]] std::map<std::string, std::string> toData() const override {
    return {
      {"layer_idx", std::to_string(layer_idx_)},
      {"seq_len", std::to_string(seq_len_)},
      {"token_idx", std::to_string(token_idx_)}
    };
  }

  [[nodiscard]] std::string typeName() const override { return "LayerComplete"; }

private:
  int layer_idx_;
  int seq_len_;
  int token_idx_;  // 当前是第几个 token 的推理
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
auto makeRoPEInvFreq(int output_dim, float rope_theta) -> Tensor;

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
auto makeRotaryPosEmbedding(Tensor& position_ids, const Tensor& inv_freq,
                             float attention_scaling = 1.0f) -> std::pair<Tensor, Tensor>;

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

 public:
  Qwen3Text() = default;

  Qwen3Text(const std::string& name, const Qwen3Config& cfg) : nn::Module(name) {
    decode_blocks_ = reg<nn::ModuleList<Qwen3Decoder>>("layers", cfg.num_hidden_layers, cfg);
    for (auto [idx, b] : enumerate(decode_blocks_.list())) { b.self_attn_.layer_idx_ = idx; }
    norm_ = reg<nn::RMSNorm>("norm", cfg.rms_norm_eps);
    embedding_ = reg<nn::Embedding>("embed_tokens", cfg.vocab_size, cfg.hidden_size);
  }

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

 private:
  const Qwen3Config& cfg;
  Qwen3Text llm;
  nn::Linear lm_head_;
  bool tie_word_embeddings_;
  nn::StaticCache kv_cache_;
  int token_counter_ = 0;
};

}  // namespace mllm::models::qwen3
