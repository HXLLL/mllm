#include <iostream>
#include <filesystem>
#include <csignal>
#include <sstream>
#include <fmt/core.h>
#include <mllm/mllm.hpp>
#include <mllm/utils/Timer.hpp>
#include <mllm/models/qwen3_i/modeling_qwen3_i.hpp>
#include <mllm/models/qwen3_i/parameter_loader.hpp>
#include <mllm/models/qwen3/tokenization_qwen3.hpp>
#include "mllm/models/qwen3_i/generation_state.hpp"
#include "mllm/models/qwen3_i/qwen3_events.hpp"

static std::string g_trace_file;

static void exportTrace() {
  if (!g_trace_file.empty()) {
    mllm::Context::instance().tracer()->disable();
    if (mllm::Context::instance().tracer()->exportToCSV(g_trace_file, false)) {
      fmt::print("\nTrace data exported to: {}\n", g_trace_file);
      fmt::print("Total events recorded: {}\n", mllm::Context::instance().tracer()->size());
    } else {
      fmt::print("\nFailed to export tracing data to CSV: {}\n", g_trace_file);
    }
  }
}

static void signalHandler(int sig) {
  fmt::print("\nInterrupted (signal {})\n", sig);
  exportTrace();
  mllm::shutdownContext();
  std::exit(128 + sig);
}

using mllm::Argparse;

class Qwen3Service {
 public:
  using Model = mllm::models::qwen3_i::Qwen3IntermittentForCausalLM;
  using GenerationState = mllm::models::qwen3_i::GenerationState;
  using ParameterLoader = mllm::models::qwen3_i::ParameterLoader;

  struct Config {
    std::filesystem::path model_path;
    std::filesystem::path tokenizer_path;
    std::filesystem::path config_path;
    std::filesystem::path state_path;
    std::optional<std::filesystem::path> input_file;
    int32_t chunk_size;
    int32_t max_decode_tokens;
    bool ignore_eos;
    bool lazy_load;
  };

  explicit Qwen3Service(Config&& config)
      : config_(config),
        qwen3_cfg_(config.config_path),
        qwen3_tokenizer_(config.tokenizer_path),
        parameter_loader_(config.model_path),
        state_(config.state_path),
        model_(qwen3_cfg_, state_, parameter_loader_, config.chunk_size) {}

  void load() {
    if (std::filesystem::exists(config_.state_path)) {
      if (config_.lazy_load) {
        state_.lazyLoad();
      } else {
        mllm::Timer load_state_timer;
        state_.load();
        fmt::print("Generation state loaded in {}ms\n", load_state_timer.elapsed_ms());
      }
    } else {
      state_.create(qwen3_cfg_);
      fmt::print("Generation state created\n");
    }

    mllm::Context::instance().tracer()->record<mllm::models::qwen3_i::ModelLoadBeginEvent>();
    mllm::Timer load_model_timer;
    if (config_.lazy_load) {
      parameter_loader_.lazyLoad();
      parameter_loader_.dumpTensorStatus();
      model_.loadMinimalParams();
    } else {
      parameter_loader_.load();
      model_.loadAllParams();
    }
    parameter_loader_.dumpTensorStatus();
    mllm::Context::instance().tracer()->record<mllm::models::qwen3_i::ModelLoadCompleteEvent>();
    fmt::print("Model loaded in {}ms\n", load_model_timer.elapsed_ms());
  }

  void run() {
    fmt::print("\n{:*^60}\n", " Qwen3 Interactive CLI ");
    fmt::print("Enter 'exit' or 'quit' to end the session\n\n");

    mllm::models::ARGenerationOutputPast inputs;

    if (state_.hasStarted()) {
      fmt::print("💬 [Resuming] ");
      auto old_input = state_.getInputTokens();
      inputs["sequence"] = buildSequenceFromIds(old_input);
      replayTokens(old_input);
    } else {
      std::string prompt_text = getPrompt();
      inputs = qwen3_tokenizer_.convertMessage({.prompt = prompt_text});
      state_.startPrefill(inputs["sequence"]);
    }

    model_.streamGenerate(inputs, makeGenArgs(), [&](int64_t token_id) {
      std::wcout << qwen3_tokenizer_.detokenize(token_id) << std::flush;
    });
    state_.checkpoint();

    fmt::print("\n{}\n", std::string(60, '-'));
    model_.perfSummary();
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

  std::string getPrompt() {
    if (config_.input_file.has_value()) {
      fmt::print("💬 Reading from file: {}\n", config_.input_file->string());
      std::ifstream file(config_.input_file->string());
      if (!file.is_open()) {
        throw std::runtime_error(fmt::format("Failed to open input file: {}", config_.input_file->string()));
      }
      std::stringstream buffer;
      buffer << file.rdbuf();
      return buffer.str();
    } else {
      fmt::print("💬 Prompt text (or 'exit/quit'): ");
      std::string prompt_text;
      std::getline(std::cin, prompt_text);
      return prompt_text;
    }
  };

  mllm::models::ARGenerationArgs makeGenArgs() {
    return {
      {"max_decode_tokens", mllm::AnyValue(config_.max_decode_tokens)},
      {"ignore_eos", mllm::AnyValue(config_.ignore_eos)},
    };
  }

  Config config_;
  mllm::models::qwen3::Qwen3Config qwen3_cfg_;
  mllm::models::qwen3::Qwen3Tokenizer qwen3_tokenizer_;
  ParameterLoader parameter_loader_;
  GenerationState state_;
  Model model_;
};

MLLM_MAIN({
  auto& help = Argparse::add<bool>("-h|--help").help("Show help message");
  auto& model_path = Argparse::add<std::string>("-m|--model_path").help("Model path").required(true);
  auto& tokenizer_path = Argparse::add<std::string>("-t|--tokenizer_path").help("Tokenizer directory").required(true);
  auto& config_path = Argparse::add<std::string>("-c|--config_path").help("Config path").required(true);
  auto& state_path = Argparse::add<std::string>("-sp|--state_path").help("State path").required(true);
  auto& input_file = Argparse::add<std::string>("-i|--input").help("Input file path (read prompt from file)");
  auto& chunk_size = Argparse::add<int32_t>("-cs|--chunk_size").help("Chunk size").def(32);
  auto& max_decode_tokens =
      Argparse::add<int32_t>("-mdt|--max_decode_tokens").help("Maximum decode tokens (0=no limit)").def(0);
  auto& ignore_eos = Argparse::add<bool>("-ie|--ignore_eos").help("Ignore EOS token").def(false);
  auto& trace_file = Argparse::add<std::string>("-tf|--trace_file").help("Trace file path to dump trace data");
  auto& lazy_load = Argparse::add<bool>("-ll|--lazy_load").help("Lazy load (default: false)");
  Argparse::parse(argc, argv);

  if (help.isSet()) {
    Argparse::printHelp();
    mllm::shutdownContext();
    return 0;
  }

  if (trace_file.isSet()) {
    g_trace_file = trace_file.get();
    std::signal(SIGINT, signalHandler);
    mllm::Context::instance().tracer()->enable();
  }

  try {
    Qwen3Service::Config cfg{
        .model_path = model_path.get(),
        .tokenizer_path = tokenizer_path.get(),
        .config_path = config_path.get(),
        .state_path = std::filesystem::path(state_path.get()),
        .input_file = input_file.isSet() ? std::optional<std::filesystem::path>(input_file.get()) : std::nullopt,
        .chunk_size = chunk_size.get(),
        .max_decode_tokens = max_decode_tokens.get(),
        .ignore_eos = ignore_eos.get(),
        .lazy_load = lazy_load.get(),
    };
    Qwen3Service qwen3_service(std::move(cfg));
    qwen3_service.load();
    qwen3_service.run();
  } catch (const std::exception& e) {
    fmt::print("\n❌ Error: {}\n[Errno] {} ({})\n{}\n", e.what(), errno, std::strerror(errno), std::string(60, '-'));
  }

  exportTrace();
})
