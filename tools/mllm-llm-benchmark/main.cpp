// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <vector>
#include <sstream>
#include <thread>
#include <chrono>

#include <mllm/mllm.hpp>
#include <mllm/utils/Argparse.hpp>
#include <mllm/utils/CPUArchHelper.hpp>
#include <mllm/utils/PlatformRTHelper.hpp>
#include <mllm/utils/Tracing.hpp>

#define STRINGIFY_INTERNAL(x) #x
#define STRINGIFY(x) STRINGIFY_INTERNAL(x)

#include "models/All.hpp"
#include "models/Qwen3_W4A32_KAI_i.hpp"

static void print_device_info() {
  // Print Device Info
  mllm::print("ARCH               :", mllm::cpu::CURRENT_ARCH_STRING);
  mllm::print("FP16               :", mllm::cpu::hasFP16());
  mllm::print("BF16               :", mllm::cpu::hasBF16());
  mllm::print("SVE                :", mllm::cpu::hasSVE());
  mllm::print("SME                :", mllm::cpu::hasSME());
  mllm::print("Neon               :", mllm::cpu::hasNEON());
  mllm::print("DotProd            :", mllm::cpu::hasDotProd());
  mllm::print("SSE                :", mllm::cpu::hasSSE());
  mllm::print("SSE2               :", mllm::cpu::hasSSE2());
  mllm::print("SSE3               :", mllm::cpu::hasSSE3());
  mllm::print("SSSE3              :", mllm::cpu::hasSSSE3());
  mllm::print("SSE4_1             :", mllm::cpu::hasSSE4_1());
  mllm::print("SSE4_2             :", mllm::cpu::hasSSE4_2());
  mllm::print("AVX                :", mllm::cpu::hasAVX());
  mllm::print("AVX2               :", mllm::cpu::hasAVX2());
  mllm::print("AVX512F            :", mllm::cpu::hasAVX512F());
  mllm::print("AVX512BW           :", mllm::cpu::hasAVX512BW());
  mllm::print("AVX512CD           :", mllm::cpu::hasAVX512CD());
  mllm::print("AVX512DQ           :", mllm::cpu::hasAVX512DQ());
  mllm::print("AVX512VL           :", mllm::cpu::hasAVX512VL());
  mllm::print("FMA                :", mllm::cpu::hasFMA());
}

static void print_basic_info(auto &benchmark) {
  // Print Build Version
  mllm::print("MLLM Build Version :", STRINGIFY(MLLM_GIT_COMMIT_HASH));

  print_device_info();

  // Print Threading Implementation Info
  mllm::print("\n========== Threading Implementation ==========");
#ifdef MLLM_KERNEL_USE_THREADS
  #ifdef MLLM_KERNEL_THREADS_VENDOR_APPLE_GCD
    mllm::print("Threading Backend  : Apple Grand Central Dispatch (GCD)");
  #elif defined(MLLM_KERNEL_THREADS_VENDOR_OPENMP)
    mllm::print("Threading Backend  : OpenMP");
  #elif defined(MLLM_KERNEL_USE_THREADS_VENDOR_MLLM)
    mllm::print("Threading Backend  : MLLM Thread Pool");
  #else
    mllm::print("Threading Backend  : Unknown (threading enabled but no vendor specified)");
  #endif
#else
  mllm::print("Threading Backend  : Disabled (Sequential execution)");
#endif
  mllm::print("CPU Op Threads     :", mllm::Context::instance().getCpuOpThreads());
  mllm::print("===============================================\n");

  mllm::print("Model Info");
  benchmark->printModelInfo();
}

MLLM_MAIN({
  auto& help = mllm::Argparse::add<bool>("-h|--help").help("Show help message");
  auto& model_name = mllm::Argparse::add<std::string>("-n|--model_name").help("Model name");
  auto& model_path = mllm::Argparse::add<std::string>("-m|--model_path").help("Model path");
  auto& config_path = mllm::Argparse::add<std::string>("-c|--config_path").help("Config path");
  auto& num_threads = mllm::Argparse::add<int32_t>("-t|--threads").help("Number of threads");
  auto& pp = mllm::Argparse::add<std::string>("-pp|--prompt_length").help("Prompt length");
  auto& tg = mllm::Argparse::add<std::string>("-tg|--test_generation_length").help("Test Generation length");
  auto& cache_length = mllm::Argparse::add<int32_t>("-cl|--cache_length").help("Cache length");
  auto& trace_file = mllm::Argparse::add<std::string>("--trace_file").help("Trace file path (CSV). If not specified, tracing is disabled");
  auto& append_trace = mllm::Argparse::add<bool>("--append_trace").help("Append traces to existing trace file instead of overwriting (default: false)");
  auto& chunksize = mllm::Argparse::add<int32_t>("-cs|--chunksize").help("Chunk size for sequence processing (default: 1)");
  auto& intermittent = mllm::Argparse::add<bool>("-i|--intermittent").help("Intermittent mode (default: false)");
  auto& cache_dir = mllm::Argparse::add<std::string>("--cache_dir").help("Cache directory path for persistent KV cache (default: ./data/qwen3_i_kvcache)");
  mllm::Argparse::parse(argc, argv);

  // Set CPU operation threads if specified
  if (num_threads.isSet() && num_threads.get() > 0) {
    mllm::Context::instance().setCpuOpThreads(num_threads.get());
    mllm::print("CPU operation threads set to:", num_threads.get());
  } else {
    mllm::print("Using default CPU operation threads:", mllm::Context::instance().getCpuOpThreads());
  }

  // Create benchmark with all required parameters
  mllm::print("Create Benchmark: ", model_name.get());
  std::string cache_dir_path = cache_dir.isSet() ? cache_dir.get() : "";
  auto benchmark = createBenchmark(model_name.get(), config_path.get(), model_path.get(), cache_length.get(), cache_dir_path, intermittent.get());
  MLLM_RT_ASSERT(benchmark != nullptr);
  
  // Set chunksize if specified
  if (chunksize.isSet() && chunksize.get() > 0) {
    auto* qwen3_benchmark = dynamic_cast<Qwen3_W4A32_KAI_Benchmark_Intermittent*>(benchmark.get());
    if (qwen3_benchmark) {
      qwen3_benchmark->setChunkSize(chunksize.get());
    }
  }

  print_basic_info(benchmark);

  // Warmup run
  mllm::print("Warmup Run");
  benchmark->warmup();
  
  // Enable tracing after warmup if trace file is specified
  if (trace_file.isSet()) {
    mllm::Context::instance().tracer()->enable();
    mllm::print("Tracing enabled, output will be written to:", trace_file.get());
  }

  // Split pp and tg if they have multiple set.
  std::vector<std::pair<int32_t, int32_t>> pp_tg_pairs;
  {
    // pp and tg are strings with multiple values separated by comma. We need to split them.
    std::vector<int32_t> pp_values;
    std::vector<int32_t> tg_values;

    // Split pp string
    std::string pp_str = pp.get();
    std::string item;
    std::istringstream pp_stream(pp_str);
    while (std::getline(pp_stream, item, ',')) { pp_values.push_back(std::stoi(item)); }

    // Split tg string
    std::string tg_str = tg.get();
    std::istringstream tg_stream(tg_str);
    while (std::getline(tg_stream, item, ',')) { tg_values.push_back(std::stoi(item)); }

    // Check that pp and tg have the same number of items
    MLLM_RT_ASSERT_EQ(pp_values.size(), tg_values.size());

    // Create pairs
    for (size_t i = 0; i < pp_values.size(); ++i) { pp_tg_pairs.emplace_back(pp_values[i], tg_values[i]); }
  }

  // Actual run for 3 turns and gives avg results. Each turn will sleep for 5 seconds to let the SoC or GPU/NPU cool down.
  mllm::print("\n========================================");
  mllm::print("Starting Benchmark Tests");
  mllm::print("========================================\n");

  for (auto [pp, tg] : pp_tg_pairs) {
    mllm::print("----------------------------------------");
    mllm::print("Test Configuration:");
    mllm::print("  Prompt Length (PP)    :", pp);
    mllm::print("  Generation Length (TG):", tg);
    mllm::print("----------------------------------------");

    const int total_runs = 1;
    // Storage for results
    std::vector<BenchmarkTemplateResult> results;
    results.reserve(total_runs);

    for (int i = 0; i < total_runs; ++i) {
      mllm::print("  Run", i + 1, "of", total_runs, "...");

      // Clear cache before each run
      benchmark->clear();

      // Run benchmark
      auto result = benchmark->run(pp, tg);
      results.push_back(result);

      mllm::print("    TTFT         :", result.ttft, "ms");
      mllm::print("    Prefill Speed:", result.prefill_speed, "tokens/s");
      mllm::print("    Decode Speed :", result.decode_speed, "tokens/s");

      // Sleep for 5 seconds between runs to cool down
      if (i < total_runs - 1) {
        mllm::print("    Cooling down for 5 seconds...");
        std::this_thread::sleep_for(std::chrono::seconds(5));
      }
    }

    // Calculate average results
    float avg_ttft = 0.0f;
    float avg_prefill_speed = 0.0f;
    float avg_decode_speed = 0.0f;

    for (const auto& result : results) {
      avg_ttft += result.ttft;
      avg_prefill_speed += result.prefill_speed;
      avg_decode_speed += result.decode_speed;
    }

    avg_ttft /= total_runs;
    avg_prefill_speed /= total_runs;
    avg_decode_speed /= total_runs;

    // Print average results
    mllm::print("\n========== Average Results ==========");
    mllm::print("Configuration: PP=", pp, " TG=", tg);
    mllm::print("Average TTFT         :", avg_ttft, "ms");
    mllm::print("Average Prefill Speed:", avg_prefill_speed, "tokens/s");
    mllm::print("Average Decode Speed :", avg_decode_speed, "tokens/s");
    mllm::print("=====================================\n");
  }

  mllm::print("\n========================================");
  mllm::print("Benchmark Tests Completed");
  mllm::print("========================================");

  // 导出 tracing 数据到 CSV 文件（如果启用了 tracing）
  if (trace_file.isSet() && mllm::Context::instance().tracer()->size() > 0) {
    // Record StopTracing event before export if tracing is still enabled
    if (mllm::Context::instance().tracer()->isEnabled()) {
      mllm::Context::instance().tracer()->disable();
    }
    
    bool append = append_trace.isSet() && append_trace.get();
    if (mllm::Context::instance().tracer()->exportToCSV(trace_file.get(), append)) {
      mllm::print("Layer tracing data exported to:", trace_file.get());
      mllm::print("Total events recorded:", mllm::Context::instance().tracer()->size());
      if (append) {
        mllm::print("Traces appended to existing file");
      }
    } else {
      mllm::print("Failed to export tracing data to CSV");
    }
  }
})
