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

class Qwen3Decoder final : public nn::Module {
 public:
  Qwen3Decoder(const std::string& name, const Qwen3Config& cfg, GenerationState &state, int idx);
  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override;

  friend class H2KV;
  friend class KV2H;
 private:
  int layer_idx_;
  int hidden_size_;
  int head_dim_;
  int num_attention_heads_;
  int num_key_value_heads_;
  int num_key_value_groups_;

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
  GenerationState &state_;
};

class Task: public nn::Module {
 public:
  Task() = default;
  explicit Task(const std::string& name) : nn::Module(name) {}
};

class H2KV : public Task {
 public:
  explicit H2KV(Qwen3Decoder& decoder, GenerationState &state);
  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override;
 private:
  int layer_idx_;
  int hidden_size_;
  int head_dim_;
  int num_attention_heads_;
  int num_key_value_heads_;
  int num_key_value_groups_;

  nn::RMSNorm &input_layer_norm_;
  nn::Linear &q_proj_;
  nn::Linear &k_proj_;
  nn::Linear &v_proj_;
  nn::RMSNorm &rms_norm_q_;
  nn::RMSNorm &rms_norm_k_;
  nn::RoPE &q_rope_;
  nn::RoPE &k_rope_;
  GenerationState &state_;
};

class KV2H : public Task {
 public:
  explicit KV2H(Qwen3Decoder& decoder, GenerationState &state);
  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override;
 private:
  int layer_idx_;
  int head_dim_;
  int num_attention_heads_;

  nn::Linear &o_proj_;
  nn::CausalMask &mask_;
  nn::Softmax &softmax_;
  Qwen3MLP &mlp_;
  nn::RMSNorm &post_attention_layer_norm_;
  GenerationState &state_;
};


class Qwen3Text final : public nn::Module {
 public:
  Qwen3Text(const std::string& name, const Qwen3Config& cfg, GenerationState &state);
  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override;
  void setChunkSize(int chunksize) { chunksize_ = chunksize; }
 private:
  Tensor forward_chunk(int chunk_id, const Tensor& token_ids, const Tensor& sin_emb, const Tensor& cos_enb,
                       const Tensor& position_ids);

  int chunksize_ = 1;
  int num_layers_;
  int num_chunks_;
  int seq_len_;

  nn::ModuleListWithIdx<Qwen3Decoder> decode_blocks_;
  std::vector<H2KV> h2kv_;
  std::vector<KV2H> kv2h_;
  nn::RMSNorm norm_;
  nn::Embedding embedding_;
  GenerationState &state_;
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
  const Qwen3Config& cfg_;
  GenerationState::ptr state_;
  Qwen3Text llm_;
  nn::Linear lm_head_;

  bool tie_word_embeddings_ = false;
};

}  // namespace mllm::models::qwen3_i
