#pragma once

#include <fstream>

#include "mllm/core/Tensor.hpp"
#include "mllm/mllm.hpp"
#include "mllm/models/qwen3/configuration_qwen3.hpp"

namespace mllm::models::qwen3_i {

struct CacheRange {
  int layer_idx;
  int offset;
  int count;

  [[nodiscard]] int end() const { return offset + count; }
  [[nodiscard]] bool isEmpty() const { return count <= 0; }
};
// TODO: add enum state

class GenerationState {
 public:
  using Qwen3Config = models::qwen3::Qwen3Config;

  explicit GenerationState(const std::filesystem::path& path);

  void create(const Qwen3Config& cfg);
  void load();
  void lazyLoad();
  void checkpoint();

  /* state machine transitions */
  void start(const Tensor& token_ids);
  [[nodiscard]] int hasStarted() const;
  void startDecode(const Tensor& token_id);

  [[nodiscard]] int getMinWatermark(int offset, int count) const;
  [[nodiscard]] bool isPositionComplete(int pos) const;

  /* cache lazy loading */
  void loadLayerKCache(const CacheRange& range);
  void loadLayerVCache(const CacheRange& range);
  void loadLayerHCache(const CacheRange& range);

  /* getter for metadata */
  [[nodiscard]] int getMaxLength() const { return max_length_; }
  [[nodiscard]] int getLayerNums() const { return layer_nums_; }
  [[nodiscard]] int getQHeads() const { return q_heads_; }
  [[nodiscard]] int getKVHeads() const { return kv_heads_; }
  [[nodiscard]] int getKVDim() const { return kv_dim_; }
  [[nodiscard]] int getHiddenSize() const { return hidden_size_; }
  [[nodiscard]] const std::vector<int64_t>& getInputTokens() const { return input_tokens_; }

  /* getter/setter for key & value cache */
  [[nodiscard]] std::array<Tensor, 2> getKV(int layer_idx);
  [[nodiscard]] std::array<Tensor, 2> getKV(const CacheRange& range);
  [[nodiscard]] Tensor getH(const CacheRange& range);
  void updateKV(const CacheRange& range, const Tensor& k, const Tensor& v);
  void updateH(const CacheRange& range, const Tensor& h);

  /* getter for loaded state */
  [[nodiscard]] bool isKVLoaded(const CacheRange& range) const;
  [[nodiscard]] bool isHLoaded(const CacheRange& range) const;
  [[nodiscard]] bool isParamLoaded(int layer) const { return param_loaded_[layer]; }

 private:
  const size_t ELEMENT_SIZE = bytesOfType(kFloat32) / lanesOfType(kFloat32);

  // Returns (start_pos, count) for positions that need to be written.
  // Returns count=0 if nothing needs to be written.
  std::pair<int, int> findWriteRange(const std::function<bool(int)>& shouldWrite) const;

  // Returns the total number of elements written
  int writeHCacheLayer(std::fstream& file, int layer);
  void writeKVCacheLayer(std::fstream& file, const Tensor& cache, int layer);

  /* metadata management */
  void initMetadata(const Qwen3Config& cfg);
  void loadMetadata();
  void saveMetadata() const;

  /* watermark management */
  void loadWatermark();
  void saveWatermark();

  /* init file streams */
  void initFiles();
  /* allocate k/v/h caches in memory*/
  void initCaches();
  /* init k/v/h loaded state in memory */
  void loadLayerKVCacheImpl(int fd, std::vector<Tensor>& cache, std::vector<uint8_t>& loaded,
                            const CacheRange& range);

          
  /* manage loaded state */
  void initLoadedState();
  void markLoaded(std::vector<uint8_t>& loaded, const CacheRange& range);
  void assertRangeLoaded(const std::vector<uint8_t>& loaded, const CacheRange& range) const;
  bool isRangeLoaded(const std::vector<uint8_t>& loaded, const CacheRange& range) const;

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

  /* cache */
  std::vector<Tensor> k_cache_;  // Shape: [layer_nums, 1, q_heads, max_cache_length, kv_dims]
  std::vector<Tensor> v_cache_;  // Shape: [layer_nums, 1, q_heads, max_cache_length, kv_dims]
  std::vector<Tensor> h_cache_;  // Shape: [layer_nums + 1, 1, max_cache_length, hidden_size]

  /* loaded states */
  std::vector<uint8_t> k_loaded_;  // [layer_nums_ * max_length_]
  std::vector<uint8_t> v_loaded_;  // [layer_nums_ * max_length_]
  std::vector<uint8_t> h_loaded_;  // [(layer_nums_ + 1) * max_length_]
  std::vector<uint8_t> param_loaded_;  // [layer_nums_]

  /* file descriptors */
  int k_cache_fd_;
  int v_cache_fd_;
  int h_cache_fd_;
  int watermark_fd_;
};

}  // namespace mllm::models::qwen3_i
