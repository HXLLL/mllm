// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/mllm.hpp"
#include "mllm/nn/Module.hpp"
#include "mllm/nn/Nn.hpp"
#include "mllm/models/qwen3_i/generation_state.hpp"
#include "mllm/models/qwen3/configuration_qwen3.hpp"
#include "mllm/models/ARGeneration.hpp"

namespace mllm::models::qwen3_i {

using Qwen3Config = mllm::models::qwen3::Qwen3Config;

class Qwen3MLP final : public nn::Module {
 public:
  Qwen3MLP() = default;
  Qwen3MLP(const std::string& name, const Qwen3Config& cfg);
  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override;

 private:
  nn::Linear gate_proj_;
  nn::Linear up_proj_;
  nn::Linear down_proj_;
  nn::SiLU silu_;
};

class Qwen3Attention final : public nn::Module {
 public:
  Qwen3Attention() = default;
  Qwen3Attention(const std::string& name, const Qwen3Config& cfg);
  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override;

  int layer_idx_ = 0;

 private:
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

  int hidden_size_ = 0;
  int head_dim_ = 0;
  int num_attention_heads_ = 0;
  int num_key_value_heads_ = 0;
  int num_key_value_groups_ = 0;
};

class Qwen3Decoder final : public nn::Module {
 public:
  Qwen3Decoder() = default;
  Qwen3Decoder(const std::string& name, const Qwen3Config& cfg);
  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override;

  Qwen3Attention self_attn_;  // Public for layer_idx access

 private:
  Qwen3MLP mlp_;
  nn::RMSNorm input_layer_norm_;
  nn::RMSNorm post_attention_layer_norm_;
};

class Qwen3Text final : public nn::Module {
 public:
  Qwen3Text() = default;
  Qwen3Text(const std::string& name, const Qwen3Config& cfg);
  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override;

  void setChunkSize(int chunksize) { chunksize_ = chunksize; }

 private:
  nn::ModuleList<Qwen3Decoder> decode_blocks_;
  nn::RMSNorm norm_;
  nn::Embedding embedding_;
  int chunksize_ = 1;
};

class Qwen3IntermittentForCausalLM : public ARGeneration, public nn::Module {
 public:
  explicit Qwen3IntermittentForCausalLM(const Qwen3Config& cfg, const std::filesystem::path& state_dir);

  ARGenerationOutputPast forward(const ARGenerationOutputPast& input, const ARGenerationArgs& args) override;
  void streamGenerate(const ARGenerationOutputPast& input, const ARGenerationArgs& args,
                      const std::function<void(int64_t)>& callback) override;

  void sync_state();
  void setChunkSize(int chunksize) { llm_.setChunkSize(chunksize); }

 private:
  static Tensor createPositionIds(int batch_size, int seq_len, int offset = 0);
  static Tensor updatePositionIdsForDecode(const Tensor& prev_position_ids, int batch_size);

  const Qwen3Config& cfg_;
  Qwen3Text llm_;
  nn::Linear lm_head_;
  GenerationState::ptr state_;

  bool tie_word_embeddings_ = false;
};

}  // namespace mllm::models::qwen3_i
