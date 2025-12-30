// Copyright (c) MLLM Team.
// Licensed under the MIT License.

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

static void print_basic_info(auto &benchmark, bool verbose = false) {
  mllm::print("Verbose level: {}", verbose);

  if (verbose) {
    print_device_info();
  }

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
  auto& pp = mllm::Argparse::add<int32_t>("-pp|--prompt_length").help("Prompt length");
  auto& tg = mllm::Argparse::add<int32_t>("-tg|--test_generation_length").help("Test Generation length");
  auto& cache_length = mllm::Argparse::add<int32_t>("-cl|--cache_length").help("Cache length");
  auto& trace_file = mllm::Argparse::add<std::string>("--trace_file").help("Trace file path (CSV). If not specified, tracing is disabled");
  auto& append_trace = mllm::Argparse::add<bool>("--append_trace").help("Append traces to existing trace file instead of overwriting (default: false)");
  auto& chunksize = mllm::Argparse::add<int32_t>("-cs|--chunksize").help("Chunk size for sequence processing (default: 1)");
  auto& intermittent = mllm::Argparse::add<bool>("-i|--intermittent").help("Intermittent mode (default: false)");
  auto& cache_dir = mllm::Argparse::add<std::string>("--cache_dir").help("Cache directory path for persistent KV cache (default: ./data/qwen3_i_kvcache)");
  auto& tokenizer_path = mllm::Argparse::add<std::string>("-t|--tokenizer_path").help("Tokenizer path");
  auto& verbose = mllm::Argparse::add<int32_t>("-v|--verbose").help("Verbose level");
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
  auto benchmark = createBenchmark(model_name.get(), config_path.get(), model_path.get(), cache_length.get(), cache_dir_path,
                                   intermittent.get());
  MLLM_RT_ASSERT(benchmark != nullptr);
  
  // Set chunksize if specified
  if (chunksize.isSet() && chunksize.get() > 0) {
    auto* qwen3_benchmark = dynamic_cast<Qwen3_W4A32_KAI_Benchmark_Intermittent*>(benchmark.get());
    if (qwen3_benchmark) {
      qwen3_benchmark->setChunkSize(chunksize.get());
    }
  }

  print_basic_info(benchmark);

  if (trace_file.isSet()) {
    mllm::Context::instance().tracer()->enable();
    mllm::print("Tracing enabled, output will be written to:", trace_file.get());
  }

  mllm::print("----------------------------------------");
  mllm::print("Test Configuration:");
  mllm::print("  Prompt Length (PP)    :", pp.get());
  mllm::print("  Generation Length (TG):", tg.get());
  mllm::print("----------------------------------------");

  // Run benchmark
  auto result = benchmark->run(pp.get(), tg.get());

  mllm::print("    TTFT         :", result.ttft, "ms");
  mllm::print("    Prefill Speed:", result.prefill_speed, "tokens/s");
  mllm::print("    Decode Speed :", result.decode_speed, "tokens/s");

  if (trace_file.isSet() && mllm::Context::instance().tracer()->size() > 0) {
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
