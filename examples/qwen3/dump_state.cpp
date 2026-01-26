#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

#include <fmt/core.h>
#include <nlohmann/json.hpp>

namespace fs = std::filesystem;

struct StateInfo {
  // From metadata.json
  int max_length;
  int layer_nums;
  int q_heads;
  int kv_heads;
  int kv_dim;
  int hidden_size;
  int phase;
  std::vector<int64_t> input_tokens;

  // Computed
  size_t expected_kv_size;
  size_t expected_h_size;
  size_t expected_watermark_size;
};

bool loadMetadata(const fs::path& path, StateInfo& info) {
  auto metadata_path = path / "metadata.json";
  if (!fs::exists(metadata_path)) {
    fmt::print("Error: metadata.json not found\n");
    return false;
  }

  std::ifstream file(metadata_path);
  nlohmann::json json_obj = nlohmann::json::parse(file);

  info.max_length = json_obj.at("max_length");
  info.layer_nums = json_obj.at("layer_nums");
  info.q_heads = json_obj.at("q_heads");
  info.kv_heads = json_obj.at("kv_heads");
  info.kv_dim = json_obj.at("kv_dim");
  info.hidden_size = json_obj.at("hidden_size");
  info.phase = json_obj.at("phase");
  info.input_tokens = json_obj.at("input_tokens").get<std::vector<int64_t>>();

  // Compute expected sizes (float32 = 4 bytes)
  info.expected_kv_size = static_cast<size_t>(info.layer_nums) * info.q_heads * info.max_length * info.kv_dim * 4;
  info.expected_h_size = static_cast<size_t>(info.layer_nums + 1) * info.max_length * info.hidden_size * 4;
  info.expected_watermark_size = info.max_length;

  return true;
}

std::string phaseName(int phase) {
  switch (phase) {
    case 0: return "Init";
    case 1: return "Prefill";
    case 2: return "Decode";
    default: return "Unknown";
  }
}

std::string formatSize(size_t bytes) {
  if (bytes >= 1024 * 1024 * 1024) {
    return fmt::format("{:.2f} GB", bytes / (1024.0 * 1024.0 * 1024.0));
  } else if (bytes >= 1024 * 1024) {
    return fmt::format("{:.2f} MB", bytes / (1024.0 * 1024.0));
  } else if (bytes >= 1024) {
    return fmt::format("{:.2f} KB", bytes / 1024.0);
  }
  return fmt::format("{} bytes", bytes);
}

void checkFile(const fs::path& path, const std::string& name, size_t expected_size) {
  auto file_path = path / name;
  if (!fs::exists(file_path)) {
    fmt::print("  {}: MISSING\n", name);
    return;
  }
  auto actual_size = fs::file_size(file_path);
  if (actual_size == expected_size) {
    fmt::print("  {}: OK ({})\n", name, formatSize(actual_size));
  } else {
    fmt::print("  {}: SIZE MISMATCH (expected {}, actual {})\n", name, formatSize(expected_size), formatSize(actual_size));
  }
}

int main(int argc, char* argv[]) {
  if (argc < 2) {
    fmt::print("Usage: {} <state_path>\n", argv[0]);
    return 1;
  }

  fs::path state_path = argv[1];
  if (!fs::exists(state_path)) {
    fmt::print("Error: State path does not exist: {}\n", state_path.string());
    return 1;
  }

  fmt::print("\n=== State Dump: {} ===\n\n", state_path.string());

  // Load metadata
  StateInfo info;
  if (!loadMetadata(state_path, info)) {
    return 1;
  }

  // Print metadata
  fmt::print("Metadata:\n");
  fmt::print("  max_length:   {}\n", info.max_length);
  fmt::print("  layer_nums:   {}\n", info.layer_nums);
  fmt::print("  q_heads:      {}\n", info.q_heads);
  fmt::print("  kv_heads:     {}\n", info.kv_heads);
  fmt::print("  kv_dim:       {}\n", info.kv_dim);
  fmt::print("  hidden_size:  {}\n", info.hidden_size);
  fmt::print("  phase:        {} ({})\n", info.phase, phaseName(info.phase));
  fmt::print("  input_tokens: {} tokens\n", info.input_tokens.size());

  // Print files
  fmt::print("\nFiles:\n");
  fmt::print("  metadata.json: OK\n");
  checkFile(state_path, "layer_watermark.bin", info.expected_watermark_size);
  checkFile(state_path, "k_cache.bin", info.expected_kv_size);
  checkFile(state_path, "v_cache.bin", info.expected_kv_size);
  checkFile(state_path, "h_cache.bin", info.expected_h_size);

  // Load and analyze watermark
  auto watermark_path = state_path / "layer_watermark.bin";
  if (!fs::exists(watermark_path)) {
    fmt::print("\nWatermark: MISSING\n");
    return 1;
  }

  std::vector<int8_t> watermark(info.max_length);
  std::ifstream wm_file(watermark_path, std::ios::binary);
  wm_file.read(reinterpret_cast<char*>(watermark.data()), info.max_length);

  // Analyze watermark
  int complete_count = 0;
  int partial_count = 0;
  int empty_count = 0;
  int last_complete_pos = -1;

  for (int i = 0; i < info.max_length; ++i) {
    if (watermark[i] == info.layer_nums) {
      complete_count++;
      last_complete_pos = i;
    } else if (watermark[i] >= 0) {
      partial_count++;
    } else {
      empty_count++;
    }
  }

  fmt::print("\nWatermark Analysis:\n");

  // Show watermark ranges
  int range_start = 0;
  int8_t range_val = watermark[0];
  for (int i = 1; i <= info.max_length; ++i) {
    if (i == info.max_length || watermark[i] != range_val) {
      std::string status;
      if (range_val == info.layer_nums) {
        status = "complete";
      } else if (range_val >= 0) {
        status = fmt::format("partial (layer {})", range_val);
      } else {
        status = "empty";
      }

      if (i - range_start == 1) {
        fmt::print("  Position {}: {} (watermark={})\n", range_start, status, range_val);
      } else {
        fmt::print("  Positions {}-{}: {} (watermark={})\n", range_start, i - 1, status, range_val);
      }

      if (i < info.max_length) {
        range_start = i;
        range_val = watermark[i];
      }
    }
  }

  // Summary
  fmt::print("\nSummary:\n");
  fmt::print("  Input tokens:    {}\n", info.input_tokens.size());
  fmt::print("  Complete:        {} ({:.1f}%)\n", complete_count, 100.0 * complete_count / info.max_length);
  fmt::print("  Partial:         {}\n", partial_count);
  fmt::print("  Empty:           {}\n", empty_count);

  if (last_complete_pos >= 0) {
    fmt::print("\n  Recovery will skip {} prefill tokens (positions 0-{}).\n", last_complete_pos + 1, last_complete_pos);
  } else {
    fmt::print("\n  No complete positions. Recovery will restart from beginning.\n");
  }

  fmt::print("\n");
  return 0;
}
