#include <fstream>
#include <filesystem>
#include <fcntl.h>
#include <unistd.h>
#include <nlohmann/json.hpp>
#include "mllm/utils/Common.hpp"
#include "mllm/utils/Log.hpp"

#include "mllm/models/qwen3_i/generation_state.hpp"
#include "mllm/models/qwen3_i/qwen3_events.hpp"

namespace mllm::models::qwen3_i {

namespace fs = std::filesystem;

template<typename StreamType>
static inline StreamType open_stream_impl(const fs::path& path, const char* op_desc, auto&&... args) {
  StreamType stream;
  stream.open(path, std::forward<decltype(args)>(args)...);
  if (!stream.is_open()) { MLLM_ERROR_EXIT(ExitCode::kIOError, "Failed to open file {} for {}", path.string(), op_desc); }
  return stream;
}

template<typename... Args>
static inline std::ifstream open_ifstream(const fs::path& path, Args&&... args) {
  return open_stream_impl<std::ifstream>(path, "reading", std::forward<Args>(args)...);
}

template<typename... Args>
static inline std::ofstream open_ofstream(const fs::path& path, Args&&... args) {
  return open_stream_impl<std::ofstream>(path, "writing", std::forward<Args>(args)...);
}

template<typename... Args>
static inline std::fstream open_fstream(const fs::path& path, Args&&... args) {
  return open_stream_impl<std::fstream>(path, "read-write", std::forward<Args>(args)...);
}

static inline void preallocate_cache_file(const fs::path& dir, const std::string& filename, size_t size) {
  if (!fs::exists(dir)) { fs::create_directories(dir); }

  auto filepath = dir / filename;
  int fd = open(filepath.c_str(), O_CREAT | O_WRONLY, 0644);
  if (fd < 0) { MLLM_ERROR_EXIT(ExitCode::kIOError, "Failed to create file {}", filepath.string()); }

  if (fallocate(fd, 0, 0, size) != 0) {
    close(fd);
    MLLM_ERROR_EXIT(ExitCode::kIOError, "Failed to preallocate {} bytes for {}", size, filepath.string());
  }

  close(fd);
}

static inline void fsync_path(const fs::path& path) {
  int fd = open(path.c_str(), O_WRONLY);
  if (fd >= 0) {
    fsync(fd);
    close(fd);
  }
}

GenerationState::GenerationState(const Qwen3Config& cfg, const fs::path& path)
    : path_(path),
      max_length_(cfg.max_cache_length),
      layer_nums_(cfg.num_hidden_layers),
      q_heads_(cfg.num_attention_heads),
      kv_heads_(cfg.num_key_value_heads),
      kv_dim_(cfg.head_dim),
      hidden_size_(cfg.hidden_size) {}

void GenerationState::allocateCaches() {
  k_cache_.reserve(layer_nums_);
  v_cache_.reserve(layer_nums_);
  h_cache_.reserve(layer_nums_ + 1);
  for (int i = 0; i < layer_nums_; ++i) {
    k_cache_.emplace_back(Tensor::empty({1, q_heads_, max_length_, kv_dim_}, kFloat32, kCPU).alloc());
    v_cache_.emplace_back(Tensor::empty({1, q_heads_, max_length_, kv_dim_}, kFloat32, kCPU).alloc());
    h_cache_.emplace_back(Tensor::empty({1, max_length_, hidden_size_}, kFloat32, kCPU).alloc());
  }
  h_cache_.emplace_back(Tensor::empty({1, max_length_, hidden_size_}, kFloat32, kCPU).alloc());
}

void GenerationState::initLoadedState() {
  k_loaded_.assign(layer_nums_, std::vector<bool>(max_length_, false));
  v_loaded_.assign(layer_nums_, std::vector<bool>(max_length_, false));
  h_loaded_.assign(layer_nums_ + 1, std::vector<bool>(max_length_, false));
}

void GenerationState::markLoaded(std::vector<std::vector<bool>>& loaded, int layer_idx, int offset, int count) {
  for (int i = 0; i < count; ++i) { loaded[layer_idx][offset + i] = true; }
}

void GenerationState::loadLayerKVCacheImpl(std::ifstream& file, std::vector<Tensor>& cache,
                                           std::vector<std::vector<bool>>& loaded, int layer_idx, int offset, int count) {
  MLLM_RT_ASSERT(layer_idx >= 0 && layer_idx < layer_nums_);
  MLLM_RT_ASSERT(offset >= 0 && offset + count <= max_length_);

  size_t entry_size = kv_dim_ * ELEMENT_SIZE;
  size_t head_stride = max_length_ * entry_size;
  size_t layer_stride = q_heads_ * head_stride;

  for (int h = 0; h < q_heads_; ++h) {
    size_t file_offset = layer_idx * layer_stride + h * head_stride + offset * entry_size;
    file.seekg(static_cast<std::streamoff>(file_offset));
    auto ptr = cache[layer_idx].ptrAt<char>({0, h, offset, 0});
    file.read(ptr, count * entry_size);
  }

  markLoaded(loaded, layer_idx, offset, count);
}

void GenerationState::prepareForLayerwiseLoad() {
  auto tracer = Context::instance().tracer();
  tracer->record<StateLoadBeginEvent>();
  MLLM_INFO("GenerationState: Preparing layerwise load from {}", path_.string());

  loadMetadata();
  tracer->record<StateLoadMetadataEvent>();

  layer_watermark_.resize(max_length_);
  {
    auto watermark_file = open_ifstream(path_ / "layer_watermark.bin", std::ios::binary);
    watermark_file.read(reinterpret_cast<char*>(layer_watermark_.data()), max_length_);
  }
  last_saved_watermark_ = layer_watermark_;
  tracer->record<StateLoadWatermarkEvent>();

  k_cache_file_ = open_ifstream(path_ / "k_cache.bin", std::ios::binary);
  v_cache_file_ = open_ifstream(path_ / "v_cache.bin", std::ios::binary);
  h_cache_file_ = open_ifstream(path_ / "h_cache.bin", std::ios::binary);

  allocateCaches();
  initLoadedState();
}

void GenerationState::loadLayerKCache(int layer_idx, int offset, int count) {
  loadLayerKVCacheImpl(k_cache_file_, k_cache_, k_loaded_, layer_idx, offset, count);
}

void GenerationState::loadLayerVCache(int layer_idx, int offset, int count) {
  loadLayerKVCacheImpl(v_cache_file_, v_cache_, v_loaded_, layer_idx, offset, count);
}

void GenerationState::loadLayerHCache(int layer_idx, int offset, int count) {
  MLLM_RT_ASSERT(layer_idx >= 0 && layer_idx <= layer_nums_);
  MLLM_RT_ASSERT(offset >= 0 && offset + count <= max_length_);

  size_t entry_size = hidden_size_ * ELEMENT_SIZE;
  size_t layer_stride = max_length_ * entry_size;
  size_t file_offset = layer_idx * layer_stride + offset * entry_size;

  h_cache_file_.seekg(static_cast<std::streamoff>(file_offset));
  auto ptr = h_cache_[layer_idx].ptrAt<char>({0, offset, 0});
  h_cache_file_.read(ptr, count * entry_size);

  markLoaded(h_loaded_, layer_idx, offset, count);
}

void GenerationState::load() {
  prepareForLayerwiseLoad();

  for (int layer = 0; layer < layer_nums_; ++layer) {
    loadLayerKCache(layer, 0, max_length_);
    loadLayerVCache(layer, 0, max_length_);
    loadLayerHCache(layer, 0, max_length_);
  }
  loadLayerHCache(layer_nums_, 0, max_length_);

  auto tracer = Context::instance().tracer();
  tracer->record<StateLoadKCacheEvent>();
  tracer->record<StateLoadVCacheEvent>();
  tracer->record<StateLoadHCacheEvent>();
  tracer->record<StateLoadCompleteEvent>();
}

void GenerationState::create() {
  input_tokens_ = {};
  layer_watermark_.assign(max_length_, -1);
  last_saved_watermark_.assign(max_length_, -1);

  allocateCaches();

  size_t kv_cache_size = static_cast<size_t>(layer_nums_) * max_length_ * q_heads_ * kv_dim_ * ELEMENT_SIZE;
  size_t h_cache_size = static_cast<size_t>(layer_nums_ + 1) * max_length_ * hidden_size_ * ELEMENT_SIZE;

  preallocate_cache_file(path_, "k_cache.bin", kv_cache_size);
  preallocate_cache_file(path_, "v_cache.bin", kv_cache_size);
  preallocate_cache_file(path_, "h_cache.bin", h_cache_size);

  initLoadedState();

  checkpoint();
}

// Returns (start_pos, count) for positions that need to be written.
// Returns count=0 if nothing needs to be written.
std::pair<int, int> GenerationState::findWriteRange(const std::function<bool(int)>& shouldWrite) const {
  int pos = 0;
  // Skip positions that don't need writing
  while (pos < max_length_ && !shouldWrite(pos)) { pos++; }

  int start_pos = pos;
  // Count consecutive positions that need writing
  while (pos < max_length_ && shouldWrite(pos)) { pos++; }
  return {start_pos, pos - start_pos};
}

int GenerationState::writeHCacheLayer(std::fstream& file, int layer) {
  auto [start_pos, count] =
      findWriteRange([&](int pos) { return layer_watermark_[pos] >= layer && last_saved_watermark_[pos] < layer; });
  if (count == 0) return 0;
  size_t h_entry_size = hidden_size_ * ELEMENT_SIZE;
  size_t layer_stride = static_cast<size_t>(layer) * max_length_ * h_entry_size;
  size_t offset = layer_stride + static_cast<size_t>(start_pos) * h_entry_size;
  file.seekp(offset);
  auto ptr = h_cache_[layer].cptrAt<char>({0, start_pos, 0});
  file.write(ptr, count * h_entry_size);
  return count;
}

void GenerationState::writeKVCacheLayer(std::fstream& file, const Tensor& cache, int layer) {
  auto [start_pos, count] =
      findWriteRange([&](int pos) { return layer_watermark_[pos] > layer && last_saved_watermark_[pos] <= layer; });
  if (count == 0) return;
  size_t kv_entry_size = kv_dim_ * ELEMENT_SIZE;
  size_t head_stride = max_length_ * kv_entry_size;
  size_t layer_stride = static_cast<size_t>(layer) * q_heads_ * head_stride;
  for (int h = 0; h < q_heads_; ++h) {
    size_t offset = layer_stride + static_cast<size_t>(h) * head_stride + static_cast<size_t>(start_pos) * kv_entry_size;
    file.seekp(offset);
    auto ptr = cache.cptrAt<char>({0, h, start_pos, 0});
    file.write(ptr, count * kv_entry_size);
  }
}

void GenerationState::checkpoint() {
  auto tracer = Context::instance().tracer();
  tracer->record<CheckpointBeginEvent>();

  if (!fs::exists(path_)) { fs::create_directories(path_); }

  saveMetadata();
  tracer->record<CheckpointMetadataEvent>();

  int count = 0;

  // Write data files with layer-major sequential writes
  {
    auto k_cache_file = open_fstream(path_ / "k_cache.bin", std::ios::binary | std::ios::in | std::ios::out);
    auto v_cache_file = open_fstream(path_ / "v_cache.bin", std::ios::binary | std::ios::in | std::ios::out);
    auto h_cache_file = open_fstream(path_ / "h_cache.bin", std::ios::binary | std::ios::in | std::ios::out);
    tracer->record<CheckpointFilesOpenEvent>();

    for (int layer = 0; layer <= layer_nums_; ++layer) { count += writeHCacheLayer(h_cache_file, layer); }
    tracer->record<CheckpointHCacheWriteEvent>();

    for (int layer = 0; layer < layer_nums_; ++layer) {
      writeKVCacheLayer(k_cache_file, k_cache_[layer], layer);
      writeKVCacheLayer(v_cache_file, v_cache_[layer], layer);
    }
    tracer->record<CheckpointKVCacheWriteEvent>();

    k_cache_file.flush();
    v_cache_file.flush();
    h_cache_file.flush();
    tracer->record<CheckpointFlushEvent>();
  }

  // Fsync data files before writing commit marker
  fsync_path(path_ / "k_cache.bin");
  fsync_path(path_ / "v_cache.bin");
  fsync_path(path_ / "h_cache.bin");
  tracer->record<CheckpointFsyncDataEvent>();

  // Write watermark (commit marker) only after data is durable
  {
    auto watermark_file = open_ofstream(path_ / "layer_watermark.bin", std::ios::binary);
    watermark_file.write(reinterpret_cast<const char*>(layer_watermark_.data()), max_length_);
    watermark_file.flush();
  }  // RAII: file closed here
  fsync_path(path_ / "layer_watermark.bin");
  tracer->record<CheckpointWatermarkEvent>();

  last_saved_watermark_ = layer_watermark_;

  tracer->record<CheckpointCompleteEvent>(count);
}

void GenerationState::start(const Tensor& token_ids) {
  MLLM_RT_ASSERT(token_ids.shape()[0] == 1 && token_ids.dtype() == kInt64);
  auto seq_len = token_ids.shape()[1];
  for (int i = 0; i < seq_len; ++i) { input_tokens_.push_back(*token_ids.cptrAt<int64_t>({0, i})); }
  started_ = 1;
}

int GenerationState::hasStarted() const { return started_; }

void GenerationState::start_decode(const Tensor& token_id) {
  MLLM_RT_ASSERT(token_id.shape() == std::vector<int32_t>({1, 1}) && token_id.dtype() == kInt64);
}

const std::vector<int64_t>& GenerationState::getInputTokens() const { return input_tokens_; }

int GenerationState::getMinWatermark(int offset, int count) const {
  int min_layer = layer_watermark_[offset];
  for (int i = 1; i < count; ++i) { min_layer = std::min(min_layer, static_cast<int>(layer_watermark_[offset + i])); }
  return min_layer;
}

bool GenerationState::isPositionComplete(int pos) const { return layer_watermark_[pos] == layer_nums_; }

void GenerationState::updateKV(int layer_idx, int offset, int count, const Tensor& k, const Tensor& v) {
  MLLM_RT_ASSERT(k.shape() == std::vector<int32_t>({1, kv_heads_, count, kv_dim_}));
  MLLM_RT_ASSERT(v.shape() == std::vector<int32_t>({1, kv_heads_, count, kv_dim_}));

  auto repeat_times = q_heads_ / kv_heads_;

  for (int h = 0; h < kv_heads_; ++h) {
    auto k_ptr = k.cptrAt<mllm_byte_t>({0, h, 0, 0});
    auto v_ptr = v.cptrAt<mllm_byte_t>({0, h, 0, 0});
    for (int r = 0; r < repeat_times; ++r) {
      auto k_cache_ptr = k_cache_[layer_idx].ptrAt<mllm_byte_t>({0, h * repeat_times + r, offset, 0});
      auto v_cache_ptr = v_cache_[layer_idx].ptrAt<mllm_byte_t>({0, h * repeat_times + r, offset, 0});
      std::memcpy(k_cache_ptr, k_ptr, count * kv_dim_ * ELEMENT_SIZE);
      std::memcpy(v_cache_ptr, v_ptr, count * kv_dim_ * ELEMENT_SIZE);
    }
  }

  markLoaded(k_loaded_, layer_idx, offset, count);
  markLoaded(v_loaded_, layer_idx, offset, count);
}

void GenerationState::updateH(int layer_idx, int offset, int count, const Tensor& h) {
  MLLM_RT_ASSERT(h.shape() == std::vector<int32_t>({1, count, hidden_size_}) && h.dtype() == kFloat32);
  auto h_ptr = h.cptrAt<mllm_byte_t>({0, 0, 0});
  auto h_cache_ptr = h_cache_[layer_idx].ptrAt<mllm_byte_t>({0, offset, 0});
  std::memcpy(h_cache_ptr, h_ptr, count * hidden_size_ * ELEMENT_SIZE);
  for (int i = 0; i < count; ++i) { layer_watermark_[offset + i] = layer_idx; }
  markLoaded(h_loaded_, layer_idx, offset, count);
}

Tensor GenerationState::getH(int layer_idx, int offset, int count) {
  for (int i = 0; i < count; ++i) { MLLM_RT_ASSERT_EQ(h_loaded_[layer_idx][offset + i], true); }
  return h_cache_[layer_idx][{0, {offset, offset + count}, kAll}];
}

std::array<Tensor, 2> GenerationState::getKV(int layer_idx) {
  MLLM_ERROR_EXIT(ExitCode::kCoreError, "to be implemented");
  return {k_cache_[layer_idx][{0, kAll, kAll, kAll}], v_cache_[layer_idx][{0, kAll, kAll, kAll}]};
}

std::array<Tensor, 2> GenerationState::getKV(int layer_idx, int offset, int count) {
  for (int i = 0; i < count; ++i) {
    MLLM_RT_ASSERT_EQ(k_loaded_[layer_idx][offset + i], true);
    MLLM_RT_ASSERT_EQ(v_loaded_[layer_idx][offset + i], true);
  }
  return {{
      k_cache_[layer_idx][{0, kAll, {offset, offset + count}, kAll}],
      v_cache_[layer_idx][{0, kAll, {offset, offset + count}, kAll}],
  }};
}

void GenerationState::loadMetadata() {
  nlohmann::json json_data;
  {
    auto metadata_file = open_ifstream(path_ / "metadata.json");
    metadata_file >> json_data;
  }

  max_length_ = json_data.at("max_length");
  layer_nums_ = json_data.at("layer_nums");
  q_heads_ = json_data.at("q_heads");
  kv_heads_ = json_data.at("kv_heads");
  kv_dim_ = json_data.at("kv_dim");
  hidden_size_ = json_data.at("hidden_size");
  started_ = json_data.at("started");
  input_tokens_ = json_data.at("input_tokens").get<std::vector<int64_t>>();
}

void GenerationState::saveMetadata() const {
  nlohmann::json json_data;
  json_data["max_length"] = max_length_;
  json_data["layer_nums"] = layer_nums_;
  json_data["q_heads"] = q_heads_;
  json_data["kv_heads"] = kv_heads_;
  json_data["kv_dim"] = kv_dim_;
  json_data["hidden_size"] = hidden_size_;
  json_data["input_tokens"] = input_tokens_;
  json_data["started"] = started_;
  {
    auto metadata_file = open_ofstream(path_ / "metadata.json");
    metadata_file << json_data.dump(2);
  }
}

}  // namespace mllm::models::qwen3_i
