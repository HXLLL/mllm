#include <iostream>
#include <filesystem>
#include <fmt/core.h>
#include <mllm/mllm.hpp>
#include <mllm/models/qwen3_i/modeling_qwen3_i.hpp>
#include <mllm/models/qwen3/tokenization_qwen3.hpp>
#include <mllm/utils/AnyValue.hpp>

using Model = mllm::models::qwen3_i::Qwen3IntermittentForCausalLM;

using mllm::Argparse;

class Qwen3Service {
 public:
  Qwen3Service(const std::string& model_path, const std::string& tokenizer_path, const std::string& config_path,
               const std::string& cache_dir, mllm::ModelFileVersion file_version, const int32_t chunk_size)
      : qwen3_cfg_(config_path), qwen3_tokenizer_(tokenizer_path), qwen3_(qwen3_cfg_, cache_dir) {
    auto param = mllm::load(model_path, file_version, mllm::kCPU, false);
    qwen3_.load(param);
    qwen3_.setChunkSize(chunk_size);
  }

  void run() {
    fmt::print("\n{:*^60}\n", " Qwen3 Interactive CLI ");
    fmt::print("Enter 'exit' or 'quit' to end the session\n\n");

    std::string prompt_text;
    mllm::models::ARGenerationOutputPast inputs;

    fmt::print("💬 Prompt text (or 'exit/quit'): ");
    std::getline(std::cin, prompt_text);

    fmt::print("🔄 Processing...\n");
    inputs = qwen3_tokenizer_.convertMessage({.prompt = prompt_text});
    fmt::print("\n🤖 Response: ");

    auto callback = [&](int64_t token_id) { std::wcout << qwen3_tokenizer_.detokenize(token_id) << std::flush; };
    qwen3_.streamGenerate(inputs, {}, callback);

    fmt::print("\n{}\n", std::string(60, '-'));

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
  auto& chunk_size = Argparse::add<int32_t>("-cs|--chunk_size").help("Chunk size").def(32);
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

  try {
    Qwen3Service qwen3_service(model_path.get(), tokenizer_path.get(), config_path.get(), cache_dir.get(), file_version, chunk_size.get());
    qwen3_service.run();
  } catch (const std::exception& e) { 
    fmt::print("\n❌ Error: {}\n[Errno] {} ({})\n{}\n", e.what(), errno, std::strerror(errno), std::string(60, '-'));
  }

#ifdef MLLM_PERFETTO_ENABLE
  mllm::perf::stop();
  mllm::perf::saveReport("qwen3.perf");
#endif

  mllm::print("\n");
  mllm::memoryReport();
})
