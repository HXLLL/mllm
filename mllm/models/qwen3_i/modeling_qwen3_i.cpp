// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/models/qwen3_i/modeling_qwen3_i.hpp"
#include "mllm/models/qwen3_i/qwen3_events.hpp"
#include "mllm/nn/Functional.hpp"
#include "mllm/nn/lmcache/PersistentCache.hpp"
#include "mllm/utils/Enumerate.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <nlohmann/json.hpp>

namespace mllm::models::qwen3_i {

// ============================================================================
// RoPE Utilities
// ============================================================================

static Tensor makeRoPEInvFreq(int output_dim, float rope_theta) {
  auto inv_freq = Tensor::empty({output_dim / 2}, kFloat32, kCPU).alloc();
  auto* ptr = inv_freq.ptr<float>();
  for (int i = 0; i < output_dim / 2; i++) {
    ptr[i] = 1.0f / std::pow(rope_theta, 2.0f * i / output_dim);
  }
  return inv_freq;
}

static std::pair<Tensor, Tensor> makeRotaryPosEmbedding(
    const Tensor& position_ids, 
    const Tensor& inv_freq, 
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

// ============================================================================
// GenerationState Implementation
// ============================================================================

void GenerationState::save(const std::filesystem::path& path) const {
  nlohmann::json j;
  if (!input_tokens.empty()) j["input_tokens"] = input_tokens;
  if (!output_tokens.empty()) j["output_tokens"] = output_tokens;
  
  std::ofstream file(path);
  if (file) file << j.dump(2);
}

GenerationState GenerationState::load(const std::filesystem::path& path) {
  GenerationState state;
  std::ifstream file(path);
  if (!file) return state;
  
  try {
    nlohmann::json j;
    file >> j;
    if (j.contains("input_tokens")) state.input_tokens = j["input_tokens"].get<std::vector<int64_t>>();
    if (j.contains("output_tokens")) state.output_tokens = j["output_tokens"].get<std::vector<int64_t>>();
  } catch (...) {
    // Return empty state on parse error
  }
  return state;
}

int GenerationState::getResumeOffset(const Tensor& input, int kv_seq_cnt) const {
  const int input_len = input.shape()[1];
  
  // No cached input: fresh start
  if (input_tokens.empty()) return -1;
  
  // Check if input matches cached
  if (static_cast<int>(input_tokens.size()) != input_len) return -1;
  
  const auto* ptr = input.ptr<int64_t>();
  if (!std::equal(ptr, ptr + input_len, input_tokens.begin())) return -1;
  
  // Input matches: return KV cache progress (clamped to input length)
  return std::min(kv_seq_cnt, input_len);
}

// ============================================================================
// Qwen3MLP Implementation
// ============================================================================

Qwen3MLP::Qwen3MLP(const std::string& name, const Qwen3Config& cfg) : nn::Module(name) {
  gate_proj_ = reg<nn::Linear>("gate_proj", cfg.hidden_size, cfg.intermediate_size, false, cfg.linear_impl_type);
  up_proj_ = reg<nn::Linear>("up_proj", cfg.hidden_size, cfg.intermediate_size, false, cfg.linear_impl_type);
  down_proj_ = reg<nn::Linear>("down_proj", cfg.intermediate_size, cfg.hidden_size, false, cfg.linear_impl_type);
  silu_ = reg<nn::SiLU>("act");
}

std::vector<Tensor> Qwen3MLP::forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) {
  auto x = silu_(gate_proj_(inputs[0]));
  x = x * up_proj_(inputs[0]);
  return {down_proj_(x)};
}

// ============================================================================
// Qwen3Attention Implementation
// ============================================================================

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
  const auto& x = inputs[0];
  const auto& sin_emb = inputs[1];
  const auto& cos_emb = inputs[2];
  auto* kv_cache = args[0].get<nn::PersistentCache*>();

  const int B = x.shape()[0];
  const int S = x.shape()[1];

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

  // Update KV cache
  auto [cached_key, cached_value] = kv_cache->updateKVCache(layer_idx_, key, value);
  Context::instance().tracer()->record<KVCacheCompleteEvent>(layer_idx_, S, 0);

  // Compute attention scores with scaling
  const float scale = 1.f / sqrtf(static_cast<float>(head_dim_));
  Tensor attn = (cached_key.dtype() == kFloat32)
      ? softmax_(mask_(nn::functional::matmul(query, cached_key, false, true) * scale))
      : softmax_(mask_(nn::functional::matmul(query.to(kFloat32), cached_key.to(kFloat32), false, true) * scale)).to(kFloat16);

  // Compute output
  auto output = nn::functional::matmul(attn, cached_value);
  output = output.transpose(1, 2).view({B, S, num_attention_heads_ * head_dim_});
  
  return {o_proj_(output)};
}

// ============================================================================
// Qwen3Decoder Implementation
// ============================================================================

Qwen3Decoder::Qwen3Decoder(const std::string& name, const Qwen3Config& cfg) : nn::Module(name) {
  self_attn_ = reg<Qwen3Attention>("self_attn", cfg);
  mlp_ = reg<Qwen3MLP>("mlp", cfg);
  input_layer_norm_ = reg<nn::RMSNorm>("input_layernorm", cfg.rms_norm_eps);
  post_attention_layer_norm_ = reg<nn::RMSNorm>("post_attention_layernorm", cfg.rms_norm_eps);
}

std::vector<Tensor> Qwen3Decoder::forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) {
  const auto& hidden_states = inputs[0];
  const auto& sin_emb = inputs[1];
  const auto& cos_emb = inputs[2];
  const auto& kv_cache = args[0];
  const int seq_len = hidden_states.shape()[1];

  // Self-attention with residual
  auto attn_output = self_attn_(input_layer_norm_(hidden_states), sin_emb, cos_emb, kv_cache)[0];
  recordEvent<SelfAttentionCompleteEvent>(self_attn_.layer_idx_, seq_len, 0);
  auto residual = attn_output + hidden_states;

  // MLP with residual
  recordEvent<MLPBeginEvent>(self_attn_.layer_idx_, seq_len, 0);
  auto mlp_output = mlp_(post_attention_layer_norm_(residual))[0];
  recordEvent<MLPCompleteEvent>(self_attn_.layer_idx_, seq_len, 0);

  return {mlp_output + residual};
}

// ============================================================================
// Qwen3Text Implementation
// ============================================================================

Qwen3Text::Qwen3Text(const std::string& name, const Qwen3Config& cfg) : nn::Module(name) {
  decode_blocks_ = reg<nn::ModuleList<Qwen3Decoder>>("layers", cfg.num_hidden_layers, cfg);
  for (auto [idx, block] : enumerate(decode_blocks_.list())) {
    block.self_attn_.layer_idx_ = idx;
  }
  norm_ = reg<nn::RMSNorm>("norm", cfg.rms_norm_eps);
  embedding_ = reg<nn::Embedding>("embed_tokens", cfg.vocab_size, cfg.hidden_size);
}

std::vector<Tensor> Qwen3Text::forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) {
  const auto& token_ids = inputs[0];
  const auto& sin_emb = inputs[1];
  const auto& cos_emb = inputs[2];
  const auto& kv_cache = args[0];
  int token_idx = (args.size() > 1) ? args[1].get<int>() : 0;
  
  const int seq_len = token_ids.shape()[1];
  const int num_chunks = (seq_len + chunksize_ - 1) / chunksize_;
  
  std::vector<Tensor> chunk_outputs;
  chunk_outputs.reserve(num_chunks);

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
      x = decode_blocks_.list()[i](x, sin_chunk, cos_chunk, kv_cache)[0];
      recordEvent<LayerCompleteEvent>(i, chunk_len, token_idx);
    }

    chunk_outputs.push_back(x);
    token_idx += chunk_len;

    // Sync KV cache to disk after each chunk
    recordEvent<SyncStartEvent>(0, chunk_len, token_idx);
    if (!kv_cache.get<nn::PersistentCache*>()->sync()) {
      MLLM_ERROR_EXIT(ExitCode::kIOError, "Failed to sync KV cache");
    }
    recordEvent<SyncCompleteEvent>(0, chunk_len, token_idx);
  }

  auto output = nn::functional::concat(chunk_outputs, 1);
  return {norm_(output)};
}

// ============================================================================
// Qwen3IntermittentForCausalLM Implementation
// ============================================================================

Qwen3IntermittentForCausalLM::Qwen3IntermittentForCausalLM(
    const Qwen3Config& cfg, 
    const std::filesystem::path& cache_dir)
    : cfg_(cfg), state_path_(cache_dir / "generation_state.json") {
  
  MLLM_INFO("Initializing intermittent Qwen3 model");
  initializeCache(cache_dir);

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

void Qwen3IntermittentForCausalLM::initializeCache(const std::filesystem::path& cache_dir) {
  // Initialize or recover KV cache
  if (std::filesystem::exists(cache_dir / "metadata.json")) {
    MLLM_INFO("Recovering PersistentCache from: {}", cache_dir.string());
    kv_cache_ = nn::PersistentCache::recover(cache_dir);
    if (!kv_cache_) {
      MLLM_ERROR_EXIT(ExitCode::kIOError, "Failed to recover cache from {}", cache_dir.string());
    }
  } else {
    MLLM_INFO("Creating new PersistentCache at: {}", cache_dir.string());
    kv_cache_ = std::make_shared<nn::PersistentCache>(
        cache_dir, cfg_.max_cache_length, cfg_.num_hidden_layers,
        cfg_.num_attention_heads, cfg_.num_key_value_heads, cfg_.head_dim,
        kFloat32, kFloat32, kCPU);
  }

  // Load generation state
  state_ = GenerationState::load(state_path_);
  
  const int kv_seq_cnt = kv_cache_->getCurrentSeqCnt(0);
  if (hasPendingWork()) {
    MLLM_INFO("Recovered with pending work. KV seq: {}, input: {}", 
              kv_seq_cnt, state_.input_tokens.size());
  } else if (kv_seq_cnt > 0) {
    MLLM_INFO("Recovered cache. KV seq: {}", kv_seq_cnt);
  }
}

Tensor Qwen3IntermittentForCausalLM::createPositionIds(int batch_size, int seq_len, int offset) {
  auto pos_ids = Tensor::empty({batch_size, seq_len}, kInt64, kCPU).alloc();
  auto* ptr = pos_ids.ptr<int64_t>();
  
  for (int s = 0; s < seq_len; ++s) {
    const int64_t pos = s + offset;
    for (int b = 0; b < batch_size; ++b) {
      ptr[b * seq_len + s] = pos;
    }
  }
  return pos_ids;
}

Tensor Qwen3IntermittentForCausalLM::updatePositionIdsForDecode(
    const Tensor& prev_position_ids, int batch_size) {
  const int prev_len = prev_position_ids.shape()[1];
  const int64_t next_pos = prev_position_ids.ptr<int64_t>()[prev_len - 1] + 1;
  
  auto new_pos = Tensor::empty({batch_size, 1}, kInt64, kCPU).alloc();
  std::fill_n(new_pos.ptr<int64_t>(), batch_size, next_pos);
  return new_pos;
}

ARGenerationOutputPast Qwen3IntermittentForCausalLM::forward(
    const ARGenerationOutputPast& input, 
    const ARGenerationArgs& args) {
  
  auto sequence = input.at("sequence");
  const int batch_size = sequence.shape()[0];
  const int input_len = sequence.shape()[1];
  const bool is_prefill = !input.count("position_ids");
  const int kv_seq_cnt = kv_cache_->getCurrentSeqCnt(0);

  int resume_offset = 0;
  if (is_prefill) {
    resume_offset = state_.getResumeOffset(sequence, kv_seq_cnt);
    
    if (resume_offset < 0) {
      // Input changed: clear and start fresh
      MLLM_INFO("Input changed, clearing cache");
      clearAllState();
      resume_offset = 0;
    }
    
    // Save input tokens for future resume
    if (resume_offset == 0) {
      const auto* ptr = sequence.ptr<int64_t>();
      state_.input_tokens.assign(ptr, ptr + input_len);
    }
    
    // Fully cached: run last token only to get logits
    if (resume_offset >= input_len) {
      MLLM_INFO("Input fully cached ({} tokens)", input_len);
      kv_cache_->setCurrentSeqCnt(input_len - 1);
      
      auto last_token = sequence[{kAll, {input_len - 1, input_len}}];
      auto position_ids = createPositionIds(batch_size, 1, input_len - 1);
      auto [sin_emb, cos_emb] = makeRotaryPosEmbedding(position_ids, getBuffer("inv_freq"), 1.0f);
      
      sequence = llm_(last_token, sin_emb, cos_emb, AnyValue(kv_cache_.get()), AnyValue(input_len - 1))[0];
      sequence = sequence[{kAll, {sequence.shape()[1] - 1}, kAll}];
      
      if (tie_word_embeddings_) sequence = lm_head_(sequence);
      return {{"sequence", sequence}, {"position_ids", createPositionIds(batch_size, input_len, 0)}};
    }
    
    // Partial resume: skip already-processed tokens
    if (resume_offset > 0) {
      MLLM_INFO("Resuming prefill from token {} of {}", resume_offset, input_len);
      sequence = sequence[{kAll, {resume_offset, input_len}}];
    }
  }

  // Generate position IDs
  Tensor position_ids;
  const int seq_len = sequence.shape()[1];
  if (is_prefill) {
    position_ids = createPositionIds(batch_size, seq_len, resume_offset);
  } else {
    position_ids = updatePositionIdsForDecode(input.at("position_ids"), batch_size);
  }

  // Forward through transformer
  auto [sin_emb, cos_emb] = makeRotaryPosEmbedding(position_ids, getBuffer("inv_freq"), 1.0f);
  sequence = llm_(sequence, sin_emb, cos_emb, AnyValue(kv_cache_.get()), AnyValue(kv_cache_->getCurrentSeqCnt(0)))[0];

  // Extract last token's hidden states for LM head
  sequence = sequence[{kAll, {sequence.shape()[1] - 1}, kAll}];
  if (tie_word_embeddings_) sequence = lm_head_(sequence);

  // Build output position IDs
  Tensor output_position_ids = is_prefill 
      ? createPositionIds(batch_size, input_len, 0) 
      : position_ids;

  return {{"sequence", sequence}, {"position_ids", output_position_ids}};
}

void Qwen3IntermittentForCausalLM::clearAllState() {
  kv_cache_->clearCache();
  state_.clear();
  state_.save(state_path_);
  (void)kv_cache_->sync();
}

void Qwen3IntermittentForCausalLM::saveAllStates() {
  // Save generation state (input/output tokens)
  state_.save(state_path_);
  
  // Sync KV cache to disk
  if (!kv_cache_->sync()) {
    MLLM_ERROR_EXIT(ExitCode::kIOError, "Failed to sync KV cache to disk");
  }
}

bool Qwen3IntermittentForCausalLM::hasPendingWork() const {
  if (state_.input_tokens.empty()) return false;
  const int kv_seq_cnt = kv_cache_->getCurrentSeqCnt(0);
  return kv_seq_cnt > 0 && kv_seq_cnt < static_cast<int>(state_.input_tokens.size());
}

void Qwen3IntermittentForCausalLM::streamGenerate(
    const ARGenerationOutputPast& input,
    const ARGenerationArgs& args,
    const std::function<void(int64_t)>& callback) {
  
  auto sequence = input.at("sequence");
  const int input_len = sequence.shape()[1];
  const int max_length = args.count("max_length") ? args.at("max_length").get<int>() : max_length_;
  const int eos_token_id = args.count("eos_token_id") ? args.at("eos_token_id").get<int>() : eos_token_id_;
  const int kv_seq_cnt = kv_cache_->getCurrentSeqCnt(0);
  
  // Check resume possibility
  const int resume_offset = state_.getResumeOffset(sequence, kv_seq_cnt);
  const bool can_resume = (resume_offset >= 0);

  // Decode interrupted: replay cached outputs then continue
  if (can_resume && kv_seq_cnt >= input_len && !state_.output_tokens.empty()) {
    for (int64_t token : state_.output_tokens) {
      callback(token);
    }
    
    // If last token was EOS, we're done
    if (state_.output_tokens.back() == eos_token_id) {
      return;
    }
    
    // Continue from last output token
    ARGenerationOutputPast past;
    past["sequence"] = Tensor::empty({1, 1}, kInt64, kCPU).alloc();
    past["sequence"].at<mllm_int64_t>({0, 0}) = state_.output_tokens.back();
    past["position_ids"] = createPositionIds(1, kv_seq_cnt, 0);
    
    for (int i = static_cast<int>(state_.output_tokens.size()); i < max_length; ++i) {
      ARGenerationOutputPast output = forward(past, args);
      Tensor logits = output["sequence"];
      int64_t next_token_id = sampleGreedy(logits);
      
      state_.output_tokens.push_back(next_token_id);
      callback(next_token_id);
      
      // Save state after each token to enable resume on interruption
      state_.save(state_path_);
      
      if (next_token_id == eos_token_id) break;
      
      past = output;
      past["sequence"] = Tensor::empty({1, 1}, kInt64, logits.device()).alloc();
      past["sequence"].at<mllm_int64_t>({0, 0}) = next_token_id;
    }
    
    // Final sync of KV cache
    (void)kv_cache_->sync();
    return;
  }
  
  // Fresh start or input changed: clear output tokens
  if (!can_resume) {
    clearAllState();
  }
  state_.output_tokens.clear();
  
  // Normal generation flow (prefill resume handled by forward())
  ARGenerationOutputPast past = input;
  for (int i = 0; i < max_length; ++i) {
    ARGenerationOutputPast output = forward(past, args);
    Tensor logits = output["sequence"];
    int64_t next_token_id = sampleGreedy(logits);
    
    state_.output_tokens.push_back(next_token_id);
    callback(next_token_id);
    
    // Save state after each token to enable resume on interruption
    state_.save(state_path_);
    
    if (next_token_id == eos_token_id) break;
    
    past = output;
    past["sequence"] = Tensor::empty({1, 1}, kInt64, logits.device()).alloc();
    past["sequence"].at<mllm_int64_t>({0, 0}) = next_token_id;
  }

  // Final sync of KV cache
  (void)kv_cache_->sync();
}

}  // namespace mllm::models::qwen3_i
