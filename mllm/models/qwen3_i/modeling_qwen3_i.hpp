// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/mllm.hpp"
#include "mllm/nn/Module.hpp"
#include "mllm/nn/Nn.hpp"
#include "mllm/nn/lmcache/PersistentCache.hpp"
#include "mllm/models/qwen3/configuration_qwen3.hpp"
#include "mllm/models/ARGeneration.hpp"

namespace mllm::models::qwen3_i {

using Qwen3Config = mllm::models::qwen3::Qwen3Config;


class Qwen3MLP final : public nn::Module {
  nn::Linear gate_proj_;
  nn::Linear up_proj_;
  nn::Linear down_proj_;
  nn::SiLU silu_;

 public:
  Qwen3MLP() = default;
  Qwen3MLP(const std::string& name, const Qwen3Config& cfg);

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
  Qwen3Attention(const std::string& name, const Qwen3Config& cfg);

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
  Qwen3Decoder(const std::string& name, const Qwen3Config& cfg);

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override;
};

class Qwen3Text final : public nn::Module {
  nn::ModuleList<Qwen3Decoder> decode_blocks_;
  nn::RMSNorm norm_;
  nn::Embedding embedding_;
  int chunksize_;

 public:
  Qwen3Text() : chunksize_(1) {}
  Qwen3Text(const std::string& name, const Qwen3Config& cfg);

  void setChunkSize(int chunksize) { chunksize_ = chunksize; }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override;
};

class Qwen3IntermittentForCausalLM : public ARGeneration, public nn::Module {
 public:
  explicit Qwen3IntermittentForCausalLM(const Qwen3Config& cfg, const std::filesystem::path& cache_dir = std::filesystem::path("./data/qwen3_i_kvcache"));

  ARGenerationOutputPast forward(const ARGenerationOutputPast& input, const ARGenerationArgs& args) override;
  nn::PersistentCache& kvCache();

  // 重置 token 计数器（在 clear/reset 时调用）
  void resetTokenCounter() { token_counter_ = 0; }

  // 设置 chunksize（用于控制序列分块处理的大小）
  void setChunkSize(int chunksize) { llm.setChunkSize(chunksize); }

 private:
  const Qwen3Config& cfg;
  Qwen3Text llm;
  nn::Linear lm_head_;
  bool tie_word_embeddings_;
  std::shared_ptr<nn::PersistentCache> kv_cache_;
  int token_counter_ = 0;
};

}  // namespace mllm::models::qwen3_i
