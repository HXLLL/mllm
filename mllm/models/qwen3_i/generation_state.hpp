#pragma once

#include "mllm/core/Tensor.hpp"
#include "mllm/mllm.hpp"
#include "mllm/models/qwen3/configuration_qwen3.hpp"
#include "mllm/models/qwen3_i/fs.hpp"

namespace mllm::models::qwen3_i {

struct CacheRange {
  int layer_idx;
  int offset;
  int count;

  [[nodiscard]] int end() const { return offset + count; }
  [[nodiscard]] bool isEmpty() const { return count <= 0; }
};

enum class GenerationPhase {
  kInit,
  kLoad,
  kLazyLoad,
  kCheckpoint,
  kStart,
  kStartDecode,
  kGenerate,
  kEnd,
};

class GenerationState {
 public:
  using Qwen3Config = models::qwen3::Qwen3Config;

  explicit GenerationState(const std::filesystem::path& path);

  /* init */
  void create(const Qwen3Config& cfg);
  void load();
  void lazyLoad();
  void checkpoint();

  /* state machine transitions */
  void start(const Tensor& token_ids);
  [[nodiscard]] int hasStarted() const { return started_; }
  void startDecode(const Tensor& token_id);

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
  [[nodiscard]] bool isParamLoaded(int layer) const { return param_loaded_[layer]; }

  /* cache lazy loading */
  void loadLayerKCache(const CacheRange& range);
  void loadLayerVCache(const CacheRange& range);
  void loadLayerHCache(const CacheRange& range);

  /* getter/setter for key & value cache */
  [[nodiscard]] std::array<Tensor, 2> getKV(const CacheRange& range);
  [[nodiscard]] Tensor getH(const CacheRange& range);
  void updateKV(const CacheRange& range, const Tensor& k, const Tensor& v);
  void updateH(const CacheRange& range, const Tensor& h);

 private:
  const size_t ELEMENT_SIZE = bytesOfType(kFloat32) / lanesOfType(kFloat32);
  constexpr static size_t MAX_METADATA_FILE_SIZE = 1024 * 4; // 4KB

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
  [[nodiscard]] bool isRangeLoaded(const std::vector<uint8_t>& loaded, const CacheRange& range) const;

  /* cache management */
  /* allocate k/v/h caches in memory */
  void initCaches();
  // Returns (start_pos, count) for positions that need to be written
  std::pair<int, int> findWriteRange(const std::function<bool(int)>& shouldWrite) const;
  int writeHCacheLayer(FileDescriptor& file, int layer);
  int writeKVCacheLayer(FileDescriptor& file, const Tensor& cache, int layer);
  /* init k/v/h loaded state in memory */
  void loadLayerKVCacheImpl(FileDescriptor& file, std::vector<Tensor>& cache, std::vector<uint8_t>& loaded,
                            const CacheRange& range);

  /* file management */
  void createFiles();
  void openFiles();

  std::filesystem::path path_;

  /* metadata */
  int max_length_;
  int layer_nums_;
  int q_heads_;
  int kv_heads_;
  int kv_dim_;
  int hidden_size_;
  int started_{};
  std::vector<int64_t> input_tokens_;

  /* watermark */
  std::vector<int8_t> layer_watermark_;       // -1 = not computed, N = h_cache_[0..N] valid
  std::vector<int8_t> last_saved_watermark_;  // tracks what's been persisted

  /* loaded states */
  std::vector<uint8_t> k_loaded_;  // [layer_nums_ * max_length_]
  std::vector<uint8_t> v_loaded_;  // [layer_nums_ * max_length_]
  std::vector<uint8_t> h_loaded_;  // [(layer_nums_ + 1) * max_length_]
  std::vector<uint8_t> param_loaded_;  // [layer_nums_]

  /* cache */
  std::vector<Tensor> k_cache_;  // Shape: [layer_nums, 1, q_heads, max_cache_length, kv_dims]
  std::vector<Tensor> v_cache_;  // Shape: [layer_nums, 1, q_heads, max_cache_length, kv_dims]
  std::vector<Tensor> h_cache_;  // Shape: [layer_nums + 1, 1, max_cache_length, hidden_size]

  /* file descriptors */
  FileDescriptor k_cache_file_;
  FileDescriptor v_cache_file_;
  FileDescriptor h_cache_file_;
  FileDescriptor watermark_file_;
};

}  // namespace mllm::models::qwen3_i
