#include <iostream>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <csignal>
#include <atomic>
#include <thread>
#include <chrono>
#include "mllm/mllm.hpp"
#include "mllm/engine/prefix_cache/ZenFS.hpp"

using mllm::prefix_cache::ZenFileSystem;
using mllm::prefix_cache::ZenFileSystemOptions;
using mllm::prefix_cache::ZenFSBlobMMapType;
using mllm::prefix_cache::vp_addr_t;

int main(int argc, char *argv[]) {
  // Initialize ZenFS
  ZenFileSystem zenfs;
  ZenFileSystemOptions options;
  options.record = true;  // Enable persistence
  options.working_dir = "./zenfs_working_dir";
  options.blob_bits_size = 20;
  options.page_bits = 7;  // 128 pages per blob
  options.lane_bits = 5;  // 32 tokens per page
  options.per_k_token_ele = 1024;
  options.per_v_token_ele = 1024;
  options.k_dtype = mllm::kFloat16;
  options.v_dtype = mllm::kFloat16;
  options.mmap_type = ZenFSBlobMMapType::kFile;

  mllm::initializeContext();
  
  // Try to recover previous data if index.json exists
  if (std::filesystem::exists(options.working_dir + "/index.json")) {
    try {
      zenfs.recover(options.working_dir);
      std::cout << "Recovered previous ZenFS state" << std::endl;
    } catch (...) {
      std::cout << "Could not recover previous state, starting fresh" << std::endl;
      zenfs.initialize(options);
    }
  } else {
    zenfs.initialize(options);
    std::cout << "ZenFS initialized successfully" << std::endl;
  }
  
  // Allocate a block
  vp_addr_t block_addr = zenfs.malloc();
  if (block_addr == INVALID_VP_ADDR) {
    std::cerr << "Failed to allocate block" << std::endl;
    return 1;
  }
  std::cout << "Allocated block at address: " << block_addr << std::endl;
  
  // Access the block to get a pointer
  char* block_ptr = zenfs.access(block_addr);
  if (!block_ptr) {
    std::cerr << "Failed to access block" << std::endl;
    return 1;
  }
  
  // Calculate block size (per_kv_token_mem_size)
  // This is calculated internally: bytesOfType(k_dtype) * per_k_token_ele / lanesOfType(k_dtype)
  // For Float16: 2 bytes * 1024 / 1 = 2048 bytes per token
  size_t block_size = 2 * 1024; // 2048 bytes for Float16 with 1024 elements
  
  // Continuously read an integer from the memory block and write back the value plus one
  int* int_ptr = reinterpret_cast<int*>(block_ptr);
  while (true) {
    int value = *int_ptr;
    std::cout << "Read value: " << value << std::endl;
    *int_ptr = value + 1;
    std::cout << "Wrote value: " << (value + 1) << std::endl;
    
    zenfs.finalize();

    // Sleep a little to avoid busy spinning
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
  }
  
  // Cleanup
  zenfs.free(block_addr);
  zenfs.finalize();
  std::cout << "ZenFS finalized" << std::endl;
  
  mllm::shutdownContext();
  return 0;
}
