#include <iostream>
#include <filesystem>
#include <fmt/core.h>
#include <mllm/mllm.hpp>
#include <mllm/models/qwen3/modeling_qwen3.hpp>
#include <mllm/models/qwen3/tokenization_qwen3.hpp>
#include <mllm/utils/AnyValue.hpp>

using Model = mllm::models::qwen3::Qwen3ForCausalLM;

using mllm::Argparse;

class Qwen3Service {
 public:
  Qwen3Service(const std::string& model_path, const std::string& tokenizer_path, const std::string& config_path,
               const std::string& cache_dir, mllm::ModelFileVersion file_version)
      : qwen3_cfg_(config_path), qwen3_tokenizer_(tokenizer_path), qwen3_(qwen3_cfg_) {
    auto param = mllm::load(model_path, file_version, mllm::kCPU, false);
    qwen3_.load(param);
  }

  void run() {
    fmt::print("\n{:*^60}\n", " Qwen3 Interactive CLI ");
    fmt::print("Enter 'exit' or 'quit' to end the session\n\n");

    std::string prompt_text;
    mllm::models::ARGenerationOutputPast inputs;

    // const auto& cached_state = qwen3_.state();
    // const bool has_archive = std::filesystem::exists(cache_path_ / "metadata.json")
    //                          && std::filesystem::exists(cache_path_ / "generation_state.json")
    //                          && !cached_state.input_tokens.empty();

    try {
      // if (has_archive) {
      //   fmt::print("🔁 Archive detected at: {}\n", cache_path_.string());
      //   fmt::print("📤 Replaying cached tokens...\n");

      //   inputs["sequence"] = buildSequenceFromIds(cached_state.input_tokens);
      //   replayTokens(cached_state.input_tokens);
      //   replayTokens(cached_state.output_tokens);
      // } else {
        fmt::print("💬 Prompt text (or 'exit/quit'): ");
        std::getline(std::cin, prompt_text);

        fmt::print("🔄 Processing...\n");
        inputs = qwen3_tokenizer_.convertMessage({.prompt = prompt_text});
        fmt::print("\n🤖 Response: ");
      // }

      // size_t skipped_cached_outputs = has_archive ? cached_state.output_tokens.size() : 0;
      size_t seen_cached = 0;

      qwen3_.streamGenerate(inputs, {}, [&](int64_t token_id) {
        // if (seen_cached < skipped_cached_outputs) {
        //   ++seen_cached;
        //   return;
        // }
        std::wcout << qwen3_tokenizer_.detokenize(token_id) << std::flush;
      });

      fmt::print("\n{}\n", std::string(60, '-'));
    } catch (const std::exception& e) { fmt::print("\n❌ Error: {}\n{}\n", e.what(), std::string(60, '-')); }

    qwen3_.perfSummary();
  }

 private:
  mllm::Tensor buildSequenceFromIds(const std::vector<int64_t>& ids) {
    mllm::Tensor sequence = mllm::Tensor::empty({1, static_cast<int32_t>(ids.size())}, mllm::kInt64, mllm::kCPU)
                          .setMemType(mllm::kNormal)
                          .setName("qwen3-resume-sequence")
                          .alloc();
    auto* ptr = sequence.ptr<int64_t>();
    for (size_t i = 0; i < ids.size(); ++i) { ptr[i] = ids[i]; }
    return sequence;
  };

  void replayTokens(const std::vector<int64_t>& tokens) {
    for (int64_t token_id : tokens) { std::wcout << qwen3_tokenizer_.detokenize(token_id) << std::flush; }
  };

  mllm::models::qwen3::Qwen3Config qwen3_cfg_;
  mllm::models::qwen3::Qwen3Tokenizer qwen3_tokenizer_;
  std::filesystem::path cache_path_;
  Model qwen3_;
};

MLLM_MAIN({
  auto& help = Argparse::add<bool>("-h|--help").help("Show help message");
  auto& model_path = Argparse::add<std::string>("-m|--model_path").help("Model path").required(true);
  auto& model_version = Argparse::add<std::string>("-mv|--model_version").help("Model version").required(true);
  auto& tokenizer_path = Argparse::add<std::string>("-t|--tokenizer_path").help("Tokenizer directory").required(true);
  auto& config_path = Argparse::add<std::string>("-c|--config_path").help("Config path").required(true);
  auto& cache_dir = Argparse::add<std::string>("-cd|--cache_dir").help("Cache directory").required(true);
  Argparse::parse(argc, argv);

  if (help.isSet()) {
    Argparse::printHelp();
    mllm::shutdownContext();
    return 0;
  }

#ifdef MLLM_PERFETTO_ENABLE
  mllm::perf::start();
#endif

  mllm::ModelFileVersion file_version = model_version.get() == "v2" ? mllm::ModelFileVersion::kV2 : mllm::ModelFileVersion::kV1;

  std::filesystem::path cache_path = std::filesystem::path(cache_dir.get());

  Qwen3Service qwen3_service(model_path.get(), tokenizer_path.get(), config_path.get(), cache_dir.get(), file_version);
  qwen3_service.run();

#ifdef MLLM_PERFETTO_ENABLE
  mllm::perf::stop();
  mllm::perf::saveReport("qwen3.perf");
#endif

  mllm::print("\n");
  mllm::memoryReport();
})
