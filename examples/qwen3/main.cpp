#include <iostream>
#include <filesystem>
#include <fmt/core.h>
#include <mllm/mllm.hpp>
#include <mllm/utils/Timer.hpp>
#include <mllm/models/qwen3_i/modeling_qwen3_i.hpp>
#include <mllm/models/qwen3/tokenization_qwen3.hpp>
#include "mllm/models/qwen3_i/generation_state.hpp"


using mllm::Argparse;

class Qwen3Service {
 public:
  using Model = mllm::models::qwen3_i::Qwen3IntermittentForCausalLM;
  using GenerationState = mllm::models::qwen3_i::GenerationState;

  struct Config {
    std::filesystem::path model_path;
    std::filesystem::path tokenizer_path;
    std::filesystem::path config_path;
    std::filesystem::path state_path;
    mllm::ModelFileVersion file_version;
    int32_t chunk_size;
    bool use_mmap;
  };

  explicit Qwen3Service(Config&& config)
      : config_(config),
        qwen3_cfg_(config.config_path),
        qwen3_tokenizer_(config.tokenizer_path),
        state_(qwen3_cfg_, config.state_path) {}

  void load() {
    if (std::filesystem::exists(config_.state_path)) {
      mllm::Timer load_state_timer;
      state_.load();
      fmt::print("Generation state loaded in {}ms\n", load_state_timer.elapsed_ms());
    } else {
      state_.create();
      fmt::print("Generation state created\n");
    }
    model_ = std::make_unique<Model>(qwen3_cfg_, state_, config_.chunk_size);

    mllm::Timer load_model_timer;
    auto model_params = mllm::load(config_.model_path, config_.file_version, mllm::kCPU, config_.use_mmap);
    model_->load(model_params);
    fmt::print("Model loaded in {}ms\n", load_model_timer.elapsed_ms());
  }

  void run() {
    fmt::print("\n{:*^60}\n", " Qwen3 Interactive CLI ");
    fmt::print("Enter 'exit' or 'quit' to end the session\n\n");

    mllm::models::ARGenerationOutputPast inputs;

    fmt::print("💬 Prompt text (or 'exit/quit'): ");

    if (state_.hasStarted()) {
      auto old_input = state_.getInputTokens();
      inputs["sequence"] = buildSequenceFromIds(old_input);
      fmt::print("[Resuming] ");
      replayTokens(old_input);
    } else {
      std::string prompt_text;
      std::getline(std::cin, prompt_text);
      inputs = qwen3_tokenizer_.convertMessage({.prompt = prompt_text});
      state_.start(inputs["sequence"]);
    }

    model_->streamGenerate(inputs, {},
                           [&](int64_t token_id) { std::wcout << qwen3_tokenizer_.detokenize(token_id) << std::flush; });
    state_.checkpoint();

    fmt::print("\n{}\n", std::string(60, '-'));

    model_->perfSummary();
  }

 private:
  static mllm::Tensor buildSequenceFromIds(const std::vector<int64_t>& ids) {
    int seq_len = ids.size();
    mllm::Tensor sequence = mllm::Tensor::empty({1, seq_len}, mllm::kInt64, mllm::kCPU).alloc();
    auto* ptr = sequence.ptr<int64_t>();
    for (int i = 0; i < seq_len; ++i) { ptr[i] = ids[i]; }
    return sequence;
  };

  void replayTokens(const std::vector<int64_t>& tokens) {
    for (int64_t token_id : tokens) { std::wcout << qwen3_tokenizer_.detokenize(token_id) << std::flush; }
  };

  Config config_;
  mllm::models::qwen3::Qwen3Config qwen3_cfg_;
  mllm::models::qwen3::Qwen3Tokenizer qwen3_tokenizer_;
  std::unique_ptr<Model> model_;
  GenerationState state_;
};

MLLM_MAIN({
  auto& help = Argparse::add<bool>("-h|--help").help("Show help message");
  auto& model_path = Argparse::add<std::string>("-m|--model_path").help("Model path").required(true);
  auto& model_version = Argparse::add<std::string>("-mv|--model_version").help("Model version").required(true);
  auto& tokenizer_path = Argparse::add<std::string>("-t|--tokenizer_path").help("Tokenizer directory").required(true);
  auto& config_path = Argparse::add<std::string>("-c|--config_path").help("Config path").required(true);
  auto& state_path = Argparse::add<std::string>("-sp|--state_path").help("State path").required(true);
  auto& chunk_size = Argparse::add<int32_t>("-cs|--chunk_size").help("Chunk size").def(32);
  auto& use_mmap = Argparse::add<bool>("-um|--use_mmap").help("Use mmap").def(false);
  Argparse::parse(argc, argv);

  if (help.isSet()) {
    Argparse::printHelp();
    mllm::shutdownContext();
    return 0;
  }

#ifdef MLLM_PERFETTO_ENABLE
  mllm::perf::start();
#endif

  try {
    Qwen3Service qwen3_service(Qwen3Service::Config{
      .model_path = model_path.get(),
      .tokenizer_path = tokenizer_path.get(),
      .config_path = config_path.get(),
      .state_path = std::filesystem::path(state_path.get()),
      .file_version = model_version.get() == "v2" ? mllm::ModelFileVersion::kV2 : mllm::ModelFileVersion::kV1,
      .chunk_size = chunk_size.get(),
      .use_mmap = use_mmap.get(),
    });
    qwen3_service.load();
    qwen3_service.run();
  } catch (const std::exception& e) { 
    fmt::print("\n❌ Error: {}\n[Errno] {} ({})\n{}\n", e.what(), errno, std::strerror(errno), std::string(60, '-'));
  }

#ifdef MLLM_PERFETTO_ENABLE
  mllm::perf::stop();
  mllm::perf::saveReport("qwen3.perf");
#endif

  // mllm::print("\n");
  // mllm::memoryReport();
})
