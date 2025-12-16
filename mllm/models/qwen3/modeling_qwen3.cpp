// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/models/qwen3/modeling_qwen3.hpp"
#include <cmath>
#include <iostream>

namespace mllm::models::qwen3 {

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
auto makeRoPEInvFreq(int output_dim, float rope_theta) -> Tensor {
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
auto makeRotaryPosEmbedding(Tensor& position_ids, const Tensor& inv_freq,
                             float attention_scaling) -> std::pair<Tensor, Tensor> {
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

std::vector<Tensor> Qwen3MLP::forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) {
  // ========== 1. Gate 投影和激活 ==========
  // 通过 gate_proj 将输入投影到中间维度，然后应用 SiLU 激活函数
  // 输入形状: [B, S, hidden_size]
  // 输出形状: [B, S, intermediate_size]
  auto x = gate_proj_(inputs[0]);
  x = silu_(x);

  // ========== 2. Up 投影 ==========
  // 通过 up_proj 将输入投影到中间维度（与 gate 并行）
  // 输入形状: [B, S, hidden_size]
  // 输出形状: [B, S, intermediate_size]
  auto y = up_proj_(inputs[0]);

  // ========== 3. 门控机制 ==========
  // 将 gate 激活后的结果与 up 投影结果逐元素相乘
  // 这是 SwiGLU (Swish-Gated Linear Unit) 激活函数的核心
  // 输出形状: [B, S, intermediate_size]
  x = x * y;

  // ========== 4. 输出投影 ==========
  // 通过 down_proj 将中间维度投影回 hidden_size
  // 输入形状: [B, S, intermediate_size]
  // 输出形状: [B, S, hidden_size]
  x = down_proj_(x);

  return {x};
}

std::vector<Tensor> Qwen3Attention::forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) {
  // ========== 1. 获取输入参数 ==========
  // x: 输入张量，形状为 [B, S, hidden_size]
  // llm_embedding_sin/cos: RoPE 位置编码的正弦和余弦值
  // past_kv_cache: 用于存储历史 K、V 状态的缓存，支持增量解码
  auto x = inputs[0];
  auto llm_embedding_sin = inputs[1];
  auto llm_embedding_cos = inputs[2];
  auto past_kv_cache = args[0].get<nn::StaticCache*>();


  // ========== 2. 线性投影生成 Q、K、V ==========
  // 通过三个独立的线性层将输入投影为 Query、Key、Value
  // 输出形状: [B, S, H * D] (对于 Q) 或 [B, S, KV_H * D] (对于 K、V)
  // 其中 H 是注意力头数，KV_H 是键值头数（支持 GQA/MQA），D 是每个头的维度
  auto query_states = q_proj_(x);
  auto key_states = k_proj_(x);
  auto value_states = v_proj_(x);

  // ========== 3. 获取批次大小和序列长度 ==========
  // 从输入张量的形状中提取维度信息，用于后续的 reshape 操作
  int B = inputs[0].shape()[0];  // Batch size: 批次大小
  int S = inputs[0].shape()[1];  // Sequence length: 序列长度

  // ========== 4. Reshape 为多头格式 ==========
  // 将 Q、K、V 从 [B, S, H*D] 重塑为 [B, S, H, D] 的多头格式
  // 这样每个头可以独立计算注意力
  // 注意：K 和 V 使用 num_key_value_heads_（可能小于 num_attention_heads_，支持分组查询注意力）
  query_states = query_states.view({B, S, num_attention_heads_, head_dim_});
  key_states = key_states.view({B, S, num_key_value_heads_, head_dim_});
  value_states = value_states.view({B, S, num_key_value_heads_, head_dim_});

  // ========== 5. 对 Q 和 K 进行 RMSNorm 归一化 ==========
  // Qwen3 使用 RMSNorm 对 Query 和 Key 进行归一化，提高训练稳定性
  // 输出形状保持不变: [B, S, H, D]
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

  // ========== 6. MLP 前向传播 ==========
  // 通过 MLP（多层感知机）进行非线性变换
  // MLP 包含 gate_proj、SiLU 激活、up_proj、down_proj 等操作
  // 输出形状: [B, S, hidden_size]
  x = mlp_(x)[0];

  // ========== 7. 残差连接（MLP 分支） ==========
  // 将 MLP 输出与注意力分支的输出相加，完成第二个残差连接
  // 最终输出形状: [B, S, hidden_size]
  x = x + tmp;

  return {x};
}

std::vector<Tensor> Qwen3Text::forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) {
  // ========== 1. 获取解码器块列表 ==========
  // 获取所有 Transformer 解码器层的列表
  auto& blocks = decode_blocks_.list();

  // ========== 2. Token 嵌入 ==========
  // 将输入的 token IDs 转换为词嵌入向量
  // inputs[0]: token IDs，形状为 [B, S]，每个元素是词汇表中的索引
  // 输出形状: [B, S, hidden_size]
  auto x = embedding_(inputs[0]);

  // ========== 3. 获取位置编码和 KV Cache ==========
  // inputs[1]: RoPE 位置编码的正弦值
  // inputs[2]: RoPE 位置编码的余弦值
  // args[0]: KV cache，用于存储所有层的注意力状态
  // args[1]: 当前 token 序号（可选）
  auto llm_embedding_sin = inputs[1];
  auto llm_embedding_cos = inputs[2];
  auto& kv_cache = args[0];

  // 获取 token_idx（如果提供）
  int token_idx = (args.size() > 1) ? args[1].get<int>() : 0;
  int seq_len = x.shape()[1];

  // ========== 4. 逐层前向传播 ==========
  // 依次通过所有 Transformer 解码器层
  // 每一层包含：Pre-Norm、自注意力、残差连接、Pre-Norm、MLP、残差连接
  // 输出形状: [B, S, hidden_size]
  // for (auto& block : blocks) { 
  for (int i = 0; i < blocks.size(); i++) {
    auto& block = blocks[i];
    // 记录每层开始时间到全局 tracer
    globalTracer().record<LayerBeginEvent>(i, seq_len, token_idx);
    x = block(x, llm_embedding_sin, llm_embedding_cos, kv_cache)[0];
    // 记录每层完成时间到全局 tracer
    globalTracer().record<LayerCompleteEvent>(i, seq_len, token_idx);
  }

  // ========== 5. 最终归一化 ==========
  // 对所有解码器层的输出进行最终的 RMSNorm 归一化
  // 这是 Transformer 架构的标准做法，用于稳定输出
  // 输出形状: [B, S, hidden_size]
  x = norm_(x);

  return {x};
}

ARGenerationOutputPast Qwen3ForCausalLM::forward(const ARGenerationOutputPast& input, const ARGenerationArgs& args) {
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
                 AnyValue(&kv_cache_), AnyValue(token_counter_++))[0];

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

nn::StaticCache& Qwen3ForCausalLM::kvCache() { return kv_cache_; }

}  // namespace mllm::models::qwen3

