// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <mllm/mllm.hpp>
#include <mllm/nn/lmcache/PersistentCache.hpp>
#include <filesystem>
#include <cmath>
#include <thread>
#include <chrono>

using namespace mllm;  // NOLINT

int main() {
  mllm::initializeContext();

  constexpr int max_cache_length = 64;
  constexpr int layer_nums = 1;
  constexpr int q_heads = 1;
  constexpr int kv_heads = 1;
  constexpr int kv_dim = 1024;  // 调整使得一个token的KV占据一个page (4KB): 2 * (4 * 128 * 4) = 4096 bytes
  const std::filesystem::path cache_dir = "./data/test_cache";

  // 1. 启动时，如果没有目标文件夹，则创建，否则恢复
  nn::PersistentCache::ptr_t cache;
  if (std::filesystem::exists(cache_dir / "metadata.json")) {
    print("=== Recovering PersistentCache ===");
    cache = nn::PersistentCache::recover(cache_dir);
    print("Current seq count:", cache->getCurrentSeqCnt(0));
  } else {
    print("=== Creating PersistentCache ===");
    nn::PersistentCacheOptions options;
    options.working_dir = cache_dir;
    options.max_cache_length = max_cache_length;
    options.layer_nums = layer_nums;
    options.q_heads = q_heads;
    options.kv_heads = kv_heads;
    options.kv_dims = kv_dim;
    options.k_dtype = kFloat32;
    options.v_dtype = kFloat32;
    options.device_type = kCPU;
    cache = nn::PersistentCache::create(options);
    if (!cache) {
      print("Failed to create cache");
      return 1;
    }
    print("Cache created at:", cache->workingDir().string());
  }

  if (!cache) {
    print("Failed to initialize cache");
    return 1;
  }

  // 2. 循环处理
  while (true) {
    int32_t cur_seq = cache->getCurrentSeqCnt(0);
    
    // 获取当前缓存的view（不修改seq_cnt）
    auto [k_view, v_view] = cache->getKVCache(0);
    
    // 计算新值：查找第一个0，或者使用最后一个值+1
    float new_value = 1.0f;
    float prev_value = 0.0f;
    bool found_zero = false;
    int32_t update_token_pos = -1;  // 要更新的token位置（-1表示添加新token）
    
    if (k_view.numel() > 0) {
      auto* k_ptr = k_view.ptr<float>();
      const int32_t token_size = kv_heads * kv_dim;
      const int32_t num_tokens = cur_seq;
      
      // 查找第一个包含0的token
      for (int32_t token_idx = 0; token_idx < num_tokens; ++token_idx) {
        const int32_t token_offset = token_idx * token_size;
        bool token_has_zero = false;
        
        // 检查这个token是否包含0
        for (int32_t i = 0; i < token_size; ++i) {
          if (std::abs(k_ptr[token_offset + i]) < 1e-6) {
            token_has_zero = true;
            found_zero = true;
            update_token_pos = token_idx;
            
            // 获取上一个token的值来计算新值
            if (token_idx > 0) {
              prev_value = k_ptr[(token_idx - 1) * token_size];
              new_value = prev_value + 1.0f;
            } else {
              new_value = 1.0f;  // 第一个token为1
            }
            break;
          }
        }
        if (token_has_zero) break;
        
        // 记录最后一个token的值
        if (token_idx == num_tokens - 1) {
          prev_value = k_ptr[token_offset];
        }
      }
      
      // 如果没找到0，新值 = 最后一个token的值 + 1，添加新token
      if (!found_zero && num_tokens > 0) {
        prev_value = k_ptr[(num_tokens - 1) * token_size];
        new_value = prev_value + 1.0f;
      }
    }
    
    // 直接更新缓存中的数据
    if (found_zero && update_token_pos >= 0) {
      // 更新现有token的KV：直接修改view中的数据
      auto* k_ptr = k_view.ptr<float>();
      auto* v_ptr = v_view.ptr<float>();
      const int32_t token_size = kv_heads * kv_dim;
      const int32_t offset = update_token_pos * token_size;
      
      // 直接修改view中的数据
      for (int32_t i = 0; i < token_size; ++i) {
        k_ptr[offset + i] = new_value;
        v_ptr[offset + i] = new_value;
      }
      cache->markDirty();  // 标记为dirty
    } else {
      // 添加新token：先增加seq_cnt，然后获取包含新位置的view并直接修改
      cache->setCurrentSeqCnt(cur_seq + 1);
      auto [k_new_view, v_new_view] = cache->getKVCache(0);
      
      // 直接修改新token位置的数据
      auto* k_ptr = k_new_view.ptr<float>();
      auto* v_ptr = v_new_view.ptr<float>();
      const int32_t token_size = kv_heads * kv_dim;
      const int32_t new_token_offset = cur_seq * token_size;
      
      for (int32_t i = 0; i < token_size; ++i) {
        k_ptr[new_token_offset + i] = new_value;
        v_ptr[new_token_offset + i] = new_value;
      }
      cache->markDirty();  // 标记为dirty
    }
    
    // 4. 输出所有元素
    print("\n=== Iteration ===");
    print("Current seq count:", cache->getCurrentSeqCnt(0));
    
    // 获取更新后的view用于输出
    auto [k_cached, v_cached] = cache->getKVCache(0);
    if (k_cached.numel() > 0) {
      auto* k_ptr = k_cached.ptr<float>();
      std::string values_str = "Cache values:";
      for (int32_t i = 0; i < k_cached.numel(); ++i) {
        values_str += " " + std::to_string(k_ptr[i]);
      }
      print(values_str);
    }
    
    // 5. 调用sync
    if (!cache->sync()) {
      print("Failed to sync cache");
    } else {
      print("Cache synced");
    }
    
    // 短暂延迟，方便观察和中断
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
  }

  mllm::shutdownContext();
  return 0;
}
