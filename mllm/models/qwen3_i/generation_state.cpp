#include <algorithm>
#include <nlohmann/json.hpp>
#include "mllm/utils/Common.hpp"
#include "mllm/utils/Log.hpp"

#include "mllm/models/qwen3_i/generation_state.hpp"
#include "mllm/models/qwen3_i/qwen3_events.hpp"
#include "mllm/models/qwen3_i/fs.hpp"  // For FileDescriptor (metadata files only)

namespace mllm::models::qwen3_i {

namespace fs = std::filesystem;

GenerationState::GenerationState(const fs::path& path) : path_(path) {
  fsync_worker_ = std::thread(&GenerationState::fsyncWorkerLoop, this);
}

GenerationState::~GenerationState() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    stop_fsync_worker_ = true;
  }
  fsync_cv_.notify_one();
  if (fsync_worker_.joinable()) { fsync_worker_.join(); }
}

void GenerationState::create(const Qwen3Config& cfg) {
  initMetadata(cfg);

  createFiles();
  openFiles();

  initWatermark();
  initLoadedState();
  initCaches();

  checkpoint();
}

void GenerationState::load() {
  auto tracer = Context::instance().tracer();
  tracer->record<StateLoadBeginEvent>();
  MLLM_INFO("GenerationState: Preparing layerwise load from {}", path_.string());
  openFiles();

  loadMetadata();
  tracer->record<StateLoadMetadataEvent>();

  loadWatermark();
  tracer->record<StateLoadWatermarkEvent>();

  initLoadedState();
  initCaches();

  for (int layer = 0; layer < layer_nums_; ++layer) { loadLayerKCache({layer, 0, max_length_}); }
  tracer->record<StateLoadKCacheEvent>();

  for (int layer = 0; layer < layer_nums_; ++layer) { loadLayerVCache({layer, 0, max_length_}); }
  tracer->record<StateLoadVCacheEvent>();

  for (int layer = 0; layer <= layer_nums_; ++layer) { loadLayerHCache({layer, 0, max_length_}); }
  tracer->record<StateLoadHCacheEvent>();

  tracer->record<StateLoadCompleteEvent>();
}

void GenerationState::lazyLoad() {
  openFiles();
  loadMetadata();
  loadWatermark();
  initLoadedState();
  initCaches();
}

void GenerationState::checkpoint() {
  auto tracer = Context::instance().tracer();
  tracer->record<CheckpointBeginEvent>();

  saveMetadata();
  tracer->record<CheckpointMetadataEvent>();

  int count = 0;

  // Write data files with layer-major sequential writes
  tracer->record<CheckpointFilesOpenEvent>();

  for (int layer = 0; layer <= layer_nums_; ++layer) {
    count += writeHCacheLayer(h_cache_file_, layer);
  }
  tracer->record<CheckpointHCacheWriteEvent>();

  for (int layer = 0; layer < layer_nums_; ++layer) {
    writeKVCacheLayer(k_cache_file_, k_cache_[layer], layer);
    writeKVCacheLayer(v_cache_file_, v_cache_[layer], layer);
  }
  tracer->record<CheckpointKVCacheWriteEvent>();

  k_cache_file_.fsync();
  v_cache_file_.fsync();
  h_cache_file_.fsync();
  tracer->record<CheckpointFsyncDataEvent>();

  saveWatermark();
  tracer->record<CheckpointWatermarkEvent>();

  tracer->record<CheckpointCompleteEvent>(count);
}

void GenerationState::startPrefill(const Tensor& token_ids) {
  MLLM_RT_ASSERT(token_ids.shape()[0] == 1 && token_ids.dtype() == kInt64);
  auto seq_len = token_ids.shape()[1];
  auto* token_ids_ptr = token_ids.cptrAt<int64_t>({0, 0});
  input_tokens_.assign(token_ids_ptr, token_ids_ptr + seq_len);
  phase_ = GenerationPhase::kPrefill;
}

void GenerationState::startDecode() { phase_ = GenerationPhase::kDecode; }

int GenerationState::getMinWatermark(int offset, int count) const {
  int min_layer = layer_watermark_[offset];
  for (int i = 1; i < count; ++i) {
    min_layer = std::min(min_layer, static_cast<int>(layer_watermark_[offset + i]));
  }
  return min_layer;
}

bool GenerationState::isPositionComplete(int pos) const {
  return isHLoaded({layer_nums_, pos, 1});
  // return layer_watermark_[pos] == layer_nums_;
}

void GenerationState::loadLayerKCache(const CacheRange& range) {
  loadLayerKVCacheImpl(k_cache_file_, k_cache_, k_loaded_, range);
}

void GenerationState::loadLayerVCache(const CacheRange& range) {
  loadLayerKVCacheImpl(v_cache_file_, v_cache_, v_loaded_, range);
}

void GenerationState::loadLayerHCache(const CacheRange& range) {
  MLLM_RT_ASSERT(range.layer_idx >= 0 && range.layer_idx <= layer_nums_);
  MLLM_RT_ASSERT(range.offset >= 0 && range.end() <= max_length_);

  size_t entry_size = hidden_size_ * ELEMENT_SIZE;
  size_t layer_stride = max_length_ * entry_size;
  size_t file_offset = range.layer_idx * layer_stride + range.offset * entry_size;

  auto ptr = h_cache_[range.layer_idx].ptrAt<mllm_byte_t>({0, range.offset, 0});
  h_cache_file_.pread(ptr, range.count * entry_size, file_offset);

  markLoaded(h_loaded_, range);
}

std::array<Tensor, 2> GenerationState::getKV(const CacheRange& range) {
  assertRangeLoaded(k_loaded_, range);
  assertRangeLoaded(v_loaded_, range);
  return {{
      k_cache_[range.layer_idx][{0, kAll, {range.offset, range.end()}, kAll}],
      v_cache_[range.layer_idx][{0, kAll, {range.offset, range.end()}, kAll}],
  }};
}

Tensor GenerationState::getH(const CacheRange& range) {
  assertRangeLoaded(h_loaded_, range);
  return h_cache_[range.layer_idx][{0, {range.offset, range.end()}, kAll}];
}

void GenerationState::updateKV(const CacheRange& range, const Tensor& k, const Tensor& v) {
  MLLM_RT_ASSERT(k.shape() == std::vector<int32_t>({1, kv_heads_, range.count, kv_dim_}));
  MLLM_RT_ASSERT(v.shape() == std::vector<int32_t>({1, kv_heads_, range.count, kv_dim_}));

  auto repeat_times = q_heads_ / kv_heads_;

  for (int h = 0; h < kv_heads_; ++h) {
    auto k_ptr = k.cptrAt<mllm_byte_t>({0, h, 0, 0});
    auto v_ptr = v.cptrAt<mllm_byte_t>({0, h, 0, 0});
    for (int r = 0; r < repeat_times; ++r) {
      auto k_cache_ptr =
          k_cache_[range.layer_idx].ptrAt<mllm_byte_t>({0, h * repeat_times + r, range.offset, 0});
      auto v_cache_ptr =
          v_cache_[range.layer_idx].ptrAt<mllm_byte_t>({0, h * repeat_times + r, range.offset, 0});
      std::memcpy(k_cache_ptr, k_ptr, range.count * kv_dim_ * ELEMENT_SIZE);
      std::memcpy(v_cache_ptr, v_ptr, range.count * kv_dim_ * ELEMENT_SIZE);
    }
  }

  markLoaded(k_loaded_, range);
  markLoaded(v_loaded_, range);
}

void GenerationState::updateH(const CacheRange& range, const Tensor& h) {
  MLLM_RT_ASSERT(h.shape() == std::vector<int32_t>({1, range.count, hidden_size_})
                 && h.dtype() == kFloat32);
  auto h_ptr = h.cptrAt<mllm_byte_t>({0, 0, 0});
  auto h_cache_ptr = h_cache_[range.layer_idx].ptrAt<mllm_byte_t>({0, range.offset, 0});
  std::memcpy(h_cache_ptr, h_ptr, range.count * hidden_size_ * ELEMENT_SIZE);
  for (int i = 0; i < range.count; ++i) { layer_watermark_[range.offset + i] = range.layer_idx; }
  markLoaded(h_loaded_, range);
}

/* metadata management */

void GenerationState::initMetadata(const Qwen3Config& cfg) {
  max_length_ = cfg.max_cache_length;
  layer_nums_ = cfg.num_hidden_layers;
  q_heads_ = cfg.num_attention_heads;
  kv_heads_ = cfg.num_key_value_heads;
  kv_dim_ = cfg.head_dim;
  hidden_size_ = cfg.hidden_size;
}

void GenerationState::loadMetadata() {
  std::array<char, MAX_METADATA_FILE_SIZE> json_str;
  auto metadata_file = FileDescriptor(path_ / "metadata.json");
  metadata_file.read(json_str.data(), MAX_METADATA_FILE_SIZE);
  nlohmann::json json_obj = nlohmann::json::parse(json_str.data());

  max_length_ = json_obj.at("max_length");
  layer_nums_ = json_obj.at("layer_nums");
  q_heads_ = json_obj.at("q_heads");
  kv_heads_ = json_obj.at("kv_heads");
  kv_dim_ = json_obj.at("kv_dim");
  hidden_size_ = json_obj.at("hidden_size");
  phase_ = static_cast<GenerationPhase>(json_obj.at("phase"));
  input_tokens_ = json_obj.at("input_tokens").get<std::vector<int64_t>>();
}

void GenerationState::saveMetadata() const {
  nlohmann::json json_obj;
  json_obj["max_length"] = max_length_;
  json_obj["layer_nums"] = layer_nums_;
  json_obj["q_heads"] = q_heads_;
  json_obj["kv_heads"] = kv_heads_;
  json_obj["kv_dim"] = kv_dim_;
  json_obj["hidden_size"] = hidden_size_;
  json_obj["input_tokens"] = input_tokens_;
  json_obj["phase"] = static_cast<int>(phase_);

  auto metadata_file = FileDescriptor(path_ / "metadata.json");
  auto json_str = json_obj.dump(2);
  metadata_file.write(json_str.c_str(), json_str.size());
}

/* metadata management */

void GenerationState::initWatermark() {
  layer_watermark_.assign(max_length_, -1);
  last_saved_watermark_.assign(max_length_, -1);
}

void GenerationState::loadWatermark() {
  layer_watermark_.resize(max_length_);
  watermark_file_.pread(layer_watermark_.data(), max_length_, 0);
  last_saved_watermark_ = layer_watermark_;
}

void GenerationState::saveWatermark() {
  watermark_file_.pwrite(layer_watermark_.data(), max_length_, 0);
  // Debug: dump layer_watermark_ to log
  watermark_file_.fsync();
  last_saved_watermark_ = layer_watermark_;
}

/* loaded state management */

void GenerationState::initLoadedState() {
  k_loaded_.assign(layer_nums_ * max_length_, 0);
  v_loaded_.assign(layer_nums_ * max_length_, 0);
  h_loaded_.assign((layer_nums_ + 1) * max_length_, 0);
}

void GenerationState::markLoaded(std::vector<uint8_t>& loaded, const CacheRange& range) {
  size_t base_idx = range.layer_idx * max_length_;
  for (int i = 0; i < range.count; ++i) { loaded[base_idx + range.offset + i] = 1; }
}

void GenerationState::assertRangeLoaded(const std::vector<uint8_t>& loaded,
                                        const CacheRange& range) const {
  size_t base_idx = range.layer_idx * max_length_;
  for (int i = 0; i < range.count; ++i) {
    MLLM_RT_ASSERT_EQ(loaded[base_idx + range.offset + i], 1);
  }
}

bool GenerationState::isRangeLoaded(const std::vector<uint8_t>& loaded,
                                    const CacheRange& range) const {
  size_t base_idx = range.layer_idx * max_length_;
  auto begin = loaded.begin() + base_idx + range.offset;
  return std::all_of(begin, begin + range.count, [](auto val) { return val != 0; });
}

/* cache management */

void GenerationState::initCaches() {
  k_cache_.reserve(layer_nums_);
  v_cache_.reserve(layer_nums_);
  h_cache_.reserve(layer_nums_ + 1);
  for (int i = 0; i < layer_nums_; ++i) {
    k_cache_.emplace_back(
        Tensor::empty({1, q_heads_, max_length_, kv_dim_}, kFloat32, kCPU).alloc());
    v_cache_.emplace_back(
        Tensor::empty({1, q_heads_, max_length_, kv_dim_}, kFloat32, kCPU).alloc());
    h_cache_.emplace_back(Tensor::empty({1, max_length_, hidden_size_}, kFloat32, kCPU).alloc());
  }
  h_cache_.emplace_back(Tensor::empty({1, max_length_, hidden_size_}, kFloat32, kCPU).alloc());
}

// Returns (start_pos, count) for positions that need to be written.
// Returns count=0 if nothing needs to be written.
std::pair<int, int> GenerationState::findWriteRange(
    const std::function<bool(int)>& shouldWrite) const {
  int pos = 0;
  // Skip positions that don't need writing
  while (pos < max_length_ && !shouldWrite(pos)) { pos++; }

  int start_pos = pos;
  // Count consecutive positions that need writing
  while (pos < max_length_ && shouldWrite(pos)) { pos++; }
  return {start_pos, pos - start_pos};
}

int GenerationState::writeHCacheLayer(SyncFile& file, int layer) {
  auto [start_pos, count] = findWriteRange([&](int pos) {
    return layer_watermark_[pos] >= layer && last_saved_watermark_[pos] < layer;
  });
  if (count == 0) return 0;
  size_t h_entry_size = hidden_size_ * ELEMENT_SIZE;
  size_t layer_stride = static_cast<size_t>(layer) * max_length_ * h_entry_size;
  size_t offset = layer_stride + static_cast<size_t>(start_pos) * h_entry_size;
  auto ptr = h_cache_[layer].cptrAt<mllm_byte_t>({0, start_pos, 0});
  file.pwrite(ptr, count * h_entry_size, offset);
  return count;
}

int GenerationState::writeKVCacheLayer(SyncFile& file, const Tensor& cache, int layer) {
  auto [start_pos, count] = findWriteRange([&](int pos) {
    return layer_watermark_[pos] > layer && last_saved_watermark_[pos] <= layer;
  });
  if (count == 0) return 0;
  size_t kv_entry_size = kv_dim_ * ELEMENT_SIZE;
  size_t head_stride = max_length_ * kv_entry_size;
  size_t layer_stride = static_cast<size_t>(layer) * q_heads_ * head_stride;
  for (int h = 0; h < q_heads_; ++h) {
    size_t offset = layer_stride + static_cast<size_t>(h) * head_stride
                    + static_cast<size_t>(start_pos) * kv_entry_size;
    auto ptr = cache.cptrAt<mllm_byte_t>({0, h, start_pos, 0});
    file.pwrite(ptr, count * kv_entry_size, offset);
  }
  return count;
}

void GenerationState::loadLayerKVCacheImpl(SyncFile& file, std::vector<Tensor>& cache,
                                           std::vector<uint8_t>& loaded, const CacheRange& range) {
  MLLM_RT_ASSERT(range.layer_idx >= 0 && range.layer_idx < layer_nums_);
  MLLM_RT_ASSERT(range.offset >= 0 && range.end() <= max_length_);

  size_t entry_size = kv_dim_ * ELEMENT_SIZE;
  size_t head_stride = max_length_ * entry_size;
  size_t layer_stride = q_heads_ * head_stride;

  for (int h = 0; h < q_heads_; ++h) {
    size_t file_offset =
        range.layer_idx * layer_stride + h * head_stride + range.offset * entry_size;
    auto ptr = cache[range.layer_idx].ptrAt<mllm_byte_t>({0, h, range.offset, 0});
    file.pread(ptr, range.count * entry_size, file_offset);
  }

  markLoaded(loaded, range);
}

void GenerationState::createFiles() {
  if (!fs::exists(path_)) { fs::create_directories(path_); }

  size_t kv_cache_size =
      static_cast<size_t>(layer_nums_) * max_length_ * q_heads_ * kv_dim_ * ELEMENT_SIZE;
  size_t h_cache_size =
      static_cast<size_t>(layer_nums_ + 1) * max_length_ * hidden_size_ * ELEMENT_SIZE;

  preallocate_file(path_, "k_cache.bin", kv_cache_size);
  preallocate_file(path_, "v_cache.bin", kv_cache_size);
  preallocate_file(path_, "h_cache.bin", h_cache_size);
}

void GenerationState::openFiles() {
  // Calculate buffer size based on max single I/O operation
  // Largest single I/O is typically h_cache layer read: max_length * hidden_size * 4
  size_t max_io_size = static_cast<size_t>(max_length_) * hidden_size_ * ELEMENT_SIZE;
  size_t buffer_size = std::max(max_io_size, static_cast<size_t>(1024 * 1024));  // At least 1MB

  k_cache_file_ = SyncFile(path_ / "k_cache.bin", buffer_size);
  v_cache_file_ = SyncFile(path_ / "v_cache.bin", buffer_size);
  h_cache_file_ = SyncFile(path_ / "h_cache.bin", buffer_size);
  watermark_file_ =
      SyncFile(path_ / "layer_watermark.bin", alignUp(max_length_, kDirectIOAlignment));
}

void GenerationState::openAsyncFiles() {
  size_t max_io_size = static_cast<size_t>(max_length_) * hidden_size_ * ELEMENT_SIZE;
  size_t buffer_size = std::max(max_io_size, static_cast<size_t>(1024 * 1024));

  async_k_file_ =
      std::make_unique<AsyncFile>(path_ / "k_cache.bin", kDefaultMaxInflight, buffer_size);
  async_v_file_ =
      std::make_unique<AsyncFile>(path_ / "v_cache.bin", kDefaultMaxInflight, buffer_size);
  async_h_file_ =
      std::make_unique<AsyncFile>(path_ / "h_cache.bin", kDefaultMaxInflight, buffer_size);
}

void GenerationState::fsyncWorkerLoop() {
  while (true) {
    PendingFsync fsync_task;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      fsync_cv_.wait(lock, [this] { return stop_fsync_worker_ || !pending_fsyncs_.empty(); });
      if (stop_fsync_worker_ && pending_fsyncs_.empty()) { return; }
      fsync_task = std::move(pending_fsyncs_.back());
      pending_fsyncs_.pop_back();
    }

    bool success = true;
    switch (fsync_task.type) {
      case StateIOType::kFsyncK:
        if (async_k_file_) { async_k_file_->fdatasync(); }
        break;
      case StateIOType::kFsyncV:
        if (async_v_file_) { async_v_file_->fdatasync(); }
        break;
      case StateIOType::kFsyncH:
        if (async_h_file_) { async_h_file_->fdatasync(); }
        break;
      case StateIOType::kFsyncWatermark: watermark_file_.fsync(); break;
      default: success = false; break;
    }

    {
      std::lock_guard<std::mutex> lock(mutex_);
      completed_ios_.push_back({fsync_task.type, fsync_task.range, success});
      --inflight_count_;
    }

    if (fsync_task.callback) { fsync_task.callback(fsync_task.type, fsync_task.range, success); }
  }
}

bool GenerationState::submitLoadK(const CacheRange& range, StateIOCallback callback) {
  if (!async_k_file_ || !async_k_file_->hasCapacity()) { return false; }

  size_t entry_size = kv_dim_ * ELEMENT_SIZE;
  size_t head_stride = max_length_ * entry_size;
  size_t layer_stride = q_heads_ * head_stride;

  for (int h = 0; h < q_heads_; ++h) {
    size_t file_offset =
        range.layer_idx * layer_stride + h * head_stride + range.offset * entry_size;
    auto ptr = k_cache_[range.layer_idx].ptrAt<mllm_byte_t>({0, h, range.offset, 0});
    auto req_id = async_k_file_->submitRead(ptr, range.count * entry_size, file_offset);
    if (req_id == 0) { return false; }
  }

  ++inflight_count_;
  (void)callback;
  return true;
}

bool GenerationState::submitLoadV(const CacheRange& range, StateIOCallback callback) {
  (void)callback;
  if (!async_v_file_ || !async_v_file_->hasCapacity()) { return false; }

  size_t entry_size = kv_dim_ * ELEMENT_SIZE;
  size_t head_stride = max_length_ * entry_size;
  size_t layer_stride = q_heads_ * head_stride;

  for (int h = 0; h < q_heads_; ++h) {
    size_t file_offset =
        range.layer_idx * layer_stride + h * head_stride + range.offset * entry_size;
    auto ptr = v_cache_[range.layer_idx].ptrAt<mllm_byte_t>({0, h, range.offset, 0});
    auto req_id = async_v_file_->submitRead(ptr, range.count * entry_size, file_offset);
    if (req_id == 0) { return false; }
  }

  ++inflight_count_;
  return true;
}

bool GenerationState::submitLoadH(const CacheRange& range, StateIOCallback callback) {
  (void)callback;
  if (!async_h_file_ || !async_h_file_->hasCapacity()) { return false; }

  size_t entry_size = hidden_size_ * ELEMENT_SIZE;
  size_t layer_stride = max_length_ * entry_size;
  size_t file_offset = range.layer_idx * layer_stride + range.offset * entry_size;

  auto ptr = h_cache_[range.layer_idx].ptrAt<mllm_byte_t>({0, range.offset, 0});
  auto req_id = async_h_file_->submitRead(ptr, range.count * entry_size, file_offset);
  if (req_id == 0) { return false; }

  ++inflight_count_;
  return true;
}

bool GenerationState::submitWriteK(const CacheRange& range, StateIOCallback callback) {
  (void)callback;
  if (!async_k_file_ || !async_k_file_->hasCapacity()) { return false; }

  size_t entry_size = kv_dim_ * ELEMENT_SIZE;
  size_t head_stride = max_length_ * entry_size;
  size_t layer_stride = q_heads_ * head_stride;

  for (int h = 0; h < q_heads_; ++h) {
    size_t file_offset =
        range.layer_idx * layer_stride + h * head_stride + range.offset * entry_size;
    auto ptr = k_cache_[range.layer_idx].cptrAt<mllm_byte_t>({0, h, range.offset, 0});
    auto req_id = async_k_file_->submitWrite(ptr, range.count * entry_size, file_offset);
    if (req_id == 0) { return false; }
  }

  ++inflight_count_;
  return true;
}

bool GenerationState::submitWriteV(const CacheRange& range, StateIOCallback callback) {
  (void)callback;
  if (!async_v_file_ || !async_v_file_->hasCapacity()) { return false; }

  size_t entry_size = kv_dim_ * ELEMENT_SIZE;
  size_t head_stride = max_length_ * entry_size;
  size_t layer_stride = q_heads_ * head_stride;

  for (int h = 0; h < q_heads_; ++h) {
    size_t file_offset =
        range.layer_idx * layer_stride + h * head_stride + range.offset * entry_size;
    auto ptr = v_cache_[range.layer_idx].cptrAt<mllm_byte_t>({0, h, range.offset, 0});
    auto req_id = async_v_file_->submitWrite(ptr, range.count * entry_size, file_offset);
    if (req_id == 0) { return false; }
  }

  ++inflight_count_;
  return true;
}

bool GenerationState::submitWriteH(const CacheRange& range, StateIOCallback callback) {
  (void)callback;
  if (!async_h_file_ || !async_h_file_->hasCapacity()) { return false; }

  size_t entry_size = hidden_size_ * ELEMENT_SIZE;
  size_t layer_stride = max_length_ * entry_size;
  size_t file_offset = range.layer_idx * layer_stride + range.offset * entry_size;

  auto ptr = h_cache_[range.layer_idx].cptrAt<mllm_byte_t>({0, range.offset, 0});
  auto req_id = async_h_file_->submitWrite(ptr, range.count * entry_size, file_offset);
  if (req_id == 0) { return false; }

  ++inflight_count_;
  return true;
}

bool GenerationState::submitFsync(StateIOType type, StateIOCallback callback) {
  if (type != StateIOType::kFsyncK && type != StateIOType::kFsyncV && type != StateIOType::kFsyncH
      && type != StateIOType::kFsyncWatermark) {
    return false;
  }

  {
    std::lock_guard<std::mutex> lock(mutex_);
    pending_fsyncs_.push_back({type, {}, std::move(callback)});
    ++inflight_count_;
  }
  fsync_cv_.notify_one();
  return true;
}

bool GenerationState::submitWriteWatermark(StateIOCallback callback) {
  watermark_file_.pwrite(layer_watermark_.data(), max_length_, 0);
  last_saved_watermark_ = layer_watermark_;

  if (callback) { callback(StateIOType::kWriteWatermark, {}, true); }
  return true;
}

std::vector<StateIOCompletion> GenerationState::poll() {
  std::vector<StateIOCompletion> results;

  if (async_k_file_) {
    auto completions = async_k_file_->poll();
    for (const auto& c : completions) {
      if (c.status == AsyncIOStatus::kCompleted) { --inflight_count_; }
    }
  }
  if (async_v_file_) {
    auto completions = async_v_file_->poll();
    for (const auto& c : completions) {
      if (c.status == AsyncIOStatus::kCompleted) { --inflight_count_; }
    }
  }
  if (async_h_file_) {
    auto completions = async_h_file_->poll();
    for (const auto& c : completions) {
      if (c.status == AsyncIOStatus::kCompleted) { --inflight_count_; }
    }
  }

  {
    std::lock_guard<std::mutex> lock(mutex_);
    results = std::move(completed_ios_);
    completed_ios_.clear();
  }

  return results;
}

int32_t GenerationState::inflightCount() const { return inflight_count_.load(); }

bool GenerationState::hasInflight() const { return inflight_count_.load() > 0; }

bool GenerationState::hasAsyncCapacity() const {
  if (!async_k_file_ || !async_v_file_ || !async_h_file_) { return false; }
  return async_k_file_->hasCapacity() && async_v_file_->hasCapacity()
         && async_h_file_->hasCapacity();
}

}  // namespace mllm::models::qwen3_i
