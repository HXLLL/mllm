// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/mllm.hpp"
#include "mllm/nn/Module.hpp"
#include "mllm/nn/Nn.hpp"
#include "mllm/models/qwen3_i/generation_state.hpp"
#include "mllm/models/qwen3_i/parameter_loader.hpp"
#include "mllm/models/qwen3/configuration_qwen3.hpp"
#include "mllm/models/ARGeneration.hpp"

namespace mllm::models::qwen3_i {

using Qwen3Config = mllm::models::qwen3::Qwen3Config;

class Qwen3MLP final : public nn::Module {
 public:
  Qwen3MLP(const std::string& name, const Qwen3Config& cfg, ParameterLoader& parameter_loader);
  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override;

  void loadFromDisk();

 private:
  ParameterLoader &parameter_loader_;
  nn::Linear gate_proj_;
  nn::Linear up_proj_;
  nn::Linear down_proj_;
  nn::SiLU silu_;
};

class Qwen3Decoder final : public nn::Module {
 public:
  Qwen3Decoder(const std::string& name, const Qwen3Config& cfg, GenerationState& state, ParameterLoader& parameter_loader, int idx);
  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override;

  void loadFromDisk();

  friend class H2KV;
  friend class KV2H;

 private:
  ParameterLoader& parameter_loader_;
  int layer_idx_;
  int hidden_size_;
  int head_dim_;
  int num_attention_heads_;
  int num_key_value_heads_;

  // attention
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

  // MLP and layer norms
  Qwen3MLP mlp_;
  nn::RMSNorm input_layer_norm_;
  nn::RMSNorm post_attention_layer_norm_;
  GenerationState& state_;
};

class Task : public nn::Module {
 public:
  Task() = default;
  explicit Task(const std::string& name) : nn::Module(name) {}
};

class H2KV : public Task {
 public:
  explicit H2KV(Qwen3Decoder& decoder, GenerationState& state);
  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override;

 private:
  int layer_idx_;
  int hidden_size_;
  int head_dim_;
  int num_attention_heads_;
  int num_key_value_heads_;

  nn::RMSNorm& input_layer_norm_;
  nn::Linear& q_proj_;
  nn::Linear& k_proj_;
  nn::Linear& v_proj_;
  nn::RMSNorm& rms_norm_q_;
  nn::RMSNorm& rms_norm_k_;
  nn::RoPE& q_rope_;
  nn::RoPE& k_rope_;
  GenerationState& state_;
};

class KV2H : public Task {
 public:
  explicit KV2H(Qwen3Decoder& decoder, GenerationState& state);
  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override;

 private:
  int layer_idx_;
  int head_dim_;
  int num_attention_heads_;

  nn::Linear& o_proj_;
  nn::CausalMask& mask_;
  nn::Softmax& softmax_;
  Qwen3MLP& mlp_;
  nn::RMSNorm& post_attention_layer_norm_;
  GenerationState& state_;
};

class Qwen3Text final : public nn::Module {
 public:
  Qwen3Text(const std::string& name, const Qwen3Config& cfg, GenerationState& state, ParameterLoader& parameter_loader, int chunk_size);
  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override;

  void loadFromDisk();

  [[nodiscard]] std::vector<std::string> collectLayerParamNames(int layer) const;

  [[nodiscard]] H2KV& getH2KV(int layer) { return h2kv_[layer]; }
  [[nodiscard]] KV2H& getKV2H(int layer) { return kv2h_[layer]; }

 private:
  [[nodiscard]] Tensor prefill_(const Tensor& token_ids, const Tensor& sin_emb, const Tensor& cos_emb);
  [[nodiscard]] Tensor decode_(const Tensor& token_ids, const Tensor& sin_emb, const Tensor& cos_emb, int token_idx);

  ParameterLoader& parameter_loader_;
  int chunk_size_;
  int num_layers_;

  nn::ModuleListWithIdx<Qwen3Decoder> decode_blocks_;
  std::vector<H2KV> h2kv_;
  std::vector<KV2H> kv2h_;
  nn::RMSNorm norm_;
  nn::Embedding embedding_;
  GenerationState& state_;
};

class Qwen3IntermittentForCausalLM : public ARGeneration, public nn::Module {
 public:
  explicit Qwen3IntermittentForCausalLM(const Qwen3Config& cfg, GenerationState& state, ParameterLoader& parameter_loader, int chunk_size);

  ARGenerationOutputPast forward(const ARGenerationOutputPast& input, const ARGenerationArgs& args) override;
  void streamGenerate(const ARGenerationOutputPast& input, const ARGenerationArgs& args,
                      const std::function<void(int64_t)>& callback) override;

  void loadFromDisk();

 private:
  ParameterLoader& parameter_loader_;
  const Qwen3Config& cfg_;
  GenerationState& state_;
  Qwen3Text llm_;
  nn::Linear lm_head_;

  bool tie_word_embeddings_ = false;
};

}  // namespace mllm::models::qwen3_i
