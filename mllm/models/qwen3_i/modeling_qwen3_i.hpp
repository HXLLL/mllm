// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/mllm.hpp"
#include "mllm/nn/Module.hpp"
#include "mllm/nn/Nn.hpp"
#include "mllm/nn/lmcache/PersistentCache.hpp"
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

// ============================================================================
// Generation State (for resumable generation)
// ============================================================================

/**
 * @brief Manages input/output token state for resumable generation.
 * 
 * This struct is separate from PersistentCache to maintain clean separation:
 * - PersistentCache: KV cache storage only
 * - GenerationState: token tracking and resume logic
 */
struct GenerationState {
  std::vector<int64_t> input_tokens;
  std::vector<int64_t> output_tokens;

  void save(const std::filesystem::path& path) const;
  static GenerationState load(const std::filesystem::path& path);
  void clear() { input_tokens.clear(); output_tokens.clear(); }

  /**
   * @brief Determine resume offset based on input and KV cache state.
   * @param input The input tensor to check against cached input_tokens
   * @param kv_seq_cnt Current sequence count in KV cache
   * @return -1 if input doesn't match (need fresh start), otherwise the resume offset
   */
  [[nodiscard]] int getResumeOffset(const Tensor& input, int kv_seq_cnt) const;
};

// ============================================================================
// Intermittent (Resumable) Causal LM
// ============================================================================

/**
 * @brief Qwen3 model with support for interruption and resumption.
 *
 * This model uses PersistentCache to save KV cache and generation state to disk.
 * If the process is interrupted during prefill, the next run can resume from
 * where it left off instead of recomputing from the beginning.
 *
 * Key features:
 * - Automatic state persistence via mmap-backed KV cache
 * - Resume detection based on matching input tokens
 * - Clean state reset for new generation sessions
 */
class Qwen3IntermittentForCausalLM : public ARGeneration, public nn::Module {
 public:
  explicit Qwen3IntermittentForCausalLM(
      const Qwen3Config& cfg, const std::filesystem::path& cache_dir = std::filesystem::path("./data/qwen3_i_kvcache"));

  // ARGeneration interface
  ARGenerationOutputPast forward(const ARGenerationOutputPast& input, const ARGenerationArgs& args) override;
  void streamGenerate(const ARGenerationOutputPast& input, const ARGenerationArgs& args,
                      const std::function<void(int64_t)>& callback) override;

  // Cache access
  nn::PersistentCache& kvCache() { return *kv_cache_; }

  // State management
  void clearAllState();
  void saveAllStates();
  [[nodiscard]] bool hasPendingWork() const;
  [[nodiscard]] int getProcessedTokens() const { return kv_cache_->getCurrentSeqCnt(0); }
  [[nodiscard]] const GenerationState& state() const { return state_; }

  // Configuration
  void setChunkSize(int chunksize) { llm_.setChunkSize(chunksize); }

 private:
  // Initialization
  void initializeCache(const std::filesystem::path& cache_dir);

  // Position encoding
  static Tensor createPositionIds(int batch_size, int seq_len, int offset = 0);
  static Tensor updatePositionIdsForDecode(const Tensor& prev_position_ids, int batch_size);

  // Member variables
  const Qwen3Config& cfg_;
  Qwen3Text llm_;
  nn::Linear lm_head_;
  std::shared_ptr<nn::PersistentCache> kv_cache_;
  GenerationState state_;
  std::filesystem::path state_path_;
  bool tie_word_embeddings_ = false;
};

}  // namespace mllm::models::qwen3_i
