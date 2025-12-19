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
  constexpr int kv_dim = 1;  // 一行，只有一个元素
  const std::filesystem::path cache_dir = "./example_cache";

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
    
    // 读取当前缓存的所有数据
    // 通过创建一个单元素tensor来获取view（这会追加一个0，但我们可以读取之前的数据）
    auto k_temp = Tensor::zeros({1, kv_heads, 1, kv_dim}, kFloat32, kCPU);
    auto v_temp = Tensor::zeros({1, kv_heads, 1, kv_dim}, kFloat32, kCPU);
    auto [k_view, v_view] = cache->updateKVCache(0, k_temp, v_temp);
    
    // 现在k_view包含了所有数据（包括刚追加的0）
    float new_value = 1.0f;
    float prev_value = 0.0f;
    bool found_zero = false;
    
    if (k_view.numel() > 1) {
      auto* k_ptr = k_view.ptr<float>();
      // 查找第一个0（不包括最后一个，因为那是我们刚追加的）
      for (int32_t i = 0; i < k_view.numel() - 1; ++i) {
        if (std::abs(k_ptr[i]) < 1e-6) {  // 找到第一个0
          found_zero = true;
          // 获取上一个元素的值
          if (i > 0) {
            prev_value = k_ptr[i - 1];
            new_value = prev_value + 1.0f;
          } else {
            new_value = 1.0f;  // 第一个元素为1
          }
          break;
        }
        prev_value = k_ptr[i];  // 更新prev_value
      }
      
      // 如果没找到0，新值 = 最后一个元素 + 1
      if (!found_zero && k_view.numel() > 1) {
        prev_value = k_ptr[k_view.numel() - 2];  // 倒数第二个（排除刚追加的0）
        new_value = prev_value + 1.0f;
      }
    }
    
    // 回退seq_cnt（因为我们刚才追加了一个临时值）
    cache->setCurrentSeqCnt(cur_seq);
    
    // 3. 创建tensor大小为一行，更新缓存
    auto k = Tensor::fromVector(std::vector<float>{new_value}, {1, kv_heads, 1, kv_dim}, kFloat32, kCPU);
    auto v = Tensor::fromVector(std::vector<float>{new_value}, {1, kv_heads, 1, kv_dim}, kFloat32, kCPU);
    
    auto [k_cached, v_cached] = cache->updateKVCache(0, k, v);
    
    // 4. 输出所有元素
    print("\n=== Iteration ===");
    print("Current seq count:", cache->getCurrentSeqCnt(0));
    
    // 读取并输出所有元素（通过返回的view）
    if (k_cached.numel() > 0) {
      auto* k_ptr = k_cached.ptr<float>();
      print("Cache values:");
      for (int32_t i = 0; i < k_cached.numel(); ++i) {
        if (i > 0) print(" ");
        print(k_ptr[i]);
      }
      print("");
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
