// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <mllm/mllm.hpp>
#include <mllm/nn/lmcache/PersistentCache.hpp>
#include <filesystem>
#include <cmath>
#include <thread>
#include <chrono>
#include <mllm/nn/lmcache/StaticCache.hpp>

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
  std::shared_ptr<nn::PersistentCache> cache;
  if (std::filesystem::exists(cache_dir / "metadata.json")) {
    print("=== Recovering PersistentCache ===");
    cache = nn::PersistentCache::recover(cache_dir);
    if (!cache) {
      print("Failed to recover cache");
      return 1;
    }
    print("Current seq count:", cache->getCurrentSeqCnt(0));
  } else {
    print("=== Creating PersistentCache ===");
    try {
      cache = std::make_shared<nn::PersistentCache>(cache_dir, max_cache_length, layer_nums, q_heads, kv_heads, kv_dim,
                                                    kFloat32, kFloat32, kCPU);
      print("Cache created at:", cache->workingDir().string());
    } catch (const std::exception& e) {
      print("Failed to create cache:", e.what());
      return 1;
    }
  }

  auto scache = std::make_shared<nn::StaticCache>(max_cache_length, layer_nums, q_heads, kv_heads, kv_dim, kFloat32, kFloat32, kCPU, false);

  while (true) {
    int32_t cur_seq = cache->getCurrentSeqCnt(0);
    
    auto k = Tensor::ones({1, 1, 1, kv_dim}, kFloat32) * cur_seq;
    auto v = Tensor::ones({1, 1, 1, kv_dim}, kFloat32) * cur_seq;
    auto [k_cached, v_cached] = cache->updateKVCache(0, k, v);

    // print(k.shape(), v.shape(), k_cached.shape(), v_cached.shape());
    auto* k_ptr = k_cached.ptr<float>();
    std::string values_str = "Cache values:";
    for (int32_t i = 0; i < k_cached.shape()[2]; ++i) {
      values_str += " " + std::to_string(k_ptr[i * k_cached.shape()[3]]);
    }
    print(values_str);
    
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
