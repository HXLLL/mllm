#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>

#include "mllm/core/Tensor.hpp"
#include "mllm/mllm.hpp"
#include "mllm/models/qwen3/configuration_qwen3.hpp"
#include "mllm/models/qwen3_i/async_file.hpp"
#include "mllm/models/qwen3_i/file_io.hpp"

namespace mllm::models::qwen3_i {

struct CacheRange {
  int layer_idx;
  int offset;
  int count;

  [[nodiscard]] int end() const { return offset + count; }
  [[nodiscard]] bool isEmpty() const { return count <= 0; }

  bool operator==(const CacheRange& other) const {
    return layer_idx == other.layer_idx && offset == other.offset && count == other.count;
  }
};

enum class GenerationPhase {
  kInit,
  kPrefill,
  kDecode,
};

enum class StateIOType {
  kLoadK,
  kLoadV,
  kLoadH,
  kWriteK,
  kWriteV,
  kWriteH,
  kFsyncK,
  kFsyncV,
  kFsyncH,
  kWriteWatermark,
  kFsyncWatermark,
};

struct StateIOCompletion {
  StateIOType type;
  CacheRange range;
  bool success;
};

using StateIOCallback = std::function<void(StateIOType, const CacheRange&, bool)>;

class GenerationState {
 public:
  using Qwen3Config = models::qwen3::Qwen3Config;
  using SyncFile = AioFile;

  explicit GenerationState(const std::filesystem::path& path);
  ~GenerationState();

  /* init */
  void create(const Qwen3Config& cfg);
  void load();
  void lazyLoad();
  void checkpoint();

  /* state machine transitions */
  void startPrefill(const Tensor& token_ids);
  void startDecode();
  [[nodiscard]] bool hasStarted() const { return phase_ != GenerationPhase::kInit; }
  [[nodiscard]] bool isPrefillDone() const { return phase_ == GenerationPhase::kDecode; }
  [[nodiscard]] GenerationPhase getPhase() const { return phase_; }

  /* getter for metadata */
  [[nodiscard]] int getMaxLength() const { return max_length_; }
  [[nodiscard]] int getLayerNums() const { return layer_nums_; }
  [[nodiscard]] int getQHeads() const { return q_heads_; }
  [[nodiscard]] int getKVHeads() const { return kv_heads_; }
  [[nodiscard]] int getKVDim() const { return kv_dim_; }
  [[nodiscard]] int getHiddenSize() const { return hidden_size_; }
  [[nodiscard]] const std::vector<int64_t>& getInputTokens() const { return input_tokens_; }

  /* getter for watermark */
  [[nodiscard]] int getMinWatermark(int offset, int count) const;
  [[nodiscard]] bool isPositionComplete(int pos) const;

  /* getter for loaded state */
  [[nodiscard]] bool isKVLoaded(const CacheRange& range) const { return isRangeLoaded(k_loaded_, range) && isRangeLoaded(v_loaded_, range); }
  [[nodiscard]] bool isHLoaded(const CacheRange& range) const { return isRangeLoaded(h_loaded_, range); }

  /* cache lazy loading */
  void loadLayerKCache(const CacheRange& range);
  void loadLayerVCache(const CacheRange& range);
  void loadLayerHCache(const CacheRange& range);

  /* getter/setter for key & value cache */
  [[nodiscard]] std::array<Tensor, 2> getKV(const CacheRange& range);
  [[nodiscard]] Tensor getH(const CacheRange& range);
  void updateKV(const CacheRange& range, const Tensor& k, const Tensor& v);
  void updateH(const CacheRange& range, const Tensor& h);

  bool submitLoadK(const CacheRange& range, StateIOCallback callback = nullptr);
  bool submitLoadV(const CacheRange& range, StateIOCallback callback = nullptr);
  bool submitLoadH(const CacheRange& range, StateIOCallback callback = nullptr);
  bool submitWriteK(const CacheRange& range, StateIOCallback callback = nullptr);
  bool submitWriteV(const CacheRange& range, StateIOCallback callback = nullptr);
  bool submitWriteH(const CacheRange& range, StateIOCallback callback = nullptr);
  bool submitFsync(StateIOType type, StateIOCallback callback = nullptr);
  bool submitWriteWatermark(StateIOCallback callback = nullptr);

  std::vector<StateIOCompletion> poll();
  [[nodiscard]] int32_t inflightCount() const;
  [[nodiscard]] bool hasInflight() const;
  [[nodiscard]] bool hasAsyncCapacity() const;

 private:
  const size_t ELEMENT_SIZE = bytesOfType(kFloat32) / lanesOfType(kFloat32);
  constexpr static size_t MAX_METADATA_FILE_SIZE = 1024 * 128;

  /* metadata management */
  void initMetadata(const Qwen3Config& cfg);
  void loadMetadata();
  void saveMetadata() const;

  /* watermark management */
  void initWatermark();
  void loadWatermark();
  void saveWatermark();

  /* loaded state management */
  void initLoadedState();
  void markLoaded(std::vector<uint8_t>& loaded, const CacheRange& range);
  void assertRangeLoaded(const std::vector<uint8_t>& loaded, const CacheRange& range) const;
  [[nodiscard]] bool isRangeLoaded(const std::vector<uint8_t>& loaded,
                                   const CacheRange& range) const;

  /* cache management */
  /* allocate k/v/h caches in memory */
  void initCaches();
  // Returns (start_pos, count) for positions that need to be written
  std::pair<int, int> findWriteRange(const std::function<bool(int)>& shouldWrite) const;
  int writeHCacheLayer(SyncFile& file, int layer);
  int writeKVCacheLayer(SyncFile& file, const Tensor& cache, int layer);
  void loadLayerKVCacheImpl(SyncFile& file, std::vector<Tensor>& cache,
                            std::vector<uint8_t>& loaded, const CacheRange& range);

  void createFiles();
  void openFiles();
  void openAsyncFiles();

  void fsyncWorkerLoop();

  struct PendingFsync {
    StateIOType type;
    CacheRange range;
    StateIOCallback callback;
  };

  std::filesystem::path path_;

  int max_length_;
  int layer_nums_;
  int q_heads_;
  int kv_heads_;
  int kv_dim_;
  int hidden_size_;
  std::vector<int64_t> input_tokens_;
  GenerationPhase phase_{GenerationPhase::kInit};

  std::vector<int8_t> layer_watermark_;
  std::vector<int8_t> last_saved_watermark_;

  std::vector<uint8_t> k_loaded_;
  std::vector<uint8_t> v_loaded_;
  std::vector<uint8_t> h_loaded_;

  std::vector<Tensor> k_cache_;
  std::vector<Tensor> v_cache_;
  std::vector<Tensor> h_cache_;

  SyncFile k_cache_file_;
  SyncFile v_cache_file_;
  SyncFile h_cache_file_;
  SyncFile watermark_file_;

  std::unique_ptr<AsyncFile> async_k_file_;
  std::unique_ptr<AsyncFile> async_v_file_;
  std::unique_ptr<AsyncFile> async_h_file_;

  std::mutex mutex_;
  std::vector<StateIOCompletion> completed_ios_;
  std::atomic<int32_t> inflight_count_{0};

  std::thread fsync_worker_;
  std::atomic<bool> stop_fsync_worker_{false};
  std::condition_variable fsync_cv_;
  std::vector<PendingFsync> pending_fsyncs_;
};

}  // namespace mllm::models::qwen3_i
