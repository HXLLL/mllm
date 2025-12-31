#pragma once

#include "mllm/core/Tensor.hpp"

namespace mllm {

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

}  // namespace mllm
