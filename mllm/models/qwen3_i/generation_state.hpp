#pragma once

#include <fstream>

#include "mllm/core/Tensor.hpp"
#include "mllm/mllm.hpp"
#include "mllm/models/qwen3/configuration_qwen3.hpp"

namespace mllm::models::qwen3_i {

class GenerationState {
 public:
  using Qwen3Config = models::qwen3::Qwen3Config;

  explicit GenerationState(const Qwen3Config& cfg, const std::filesystem::path& path);

  void load();
  void create();
  void checkpoint();

  void start(const Tensor& token_ids);
  [[nodiscard]] int hasStarted() const;

  void start_decode(const Tensor& token_id);

  [[nodiscard]] const std::vector<int64_t>& getInputTokens() const;

  [[nodiscard]] int getMinWatermark(int offset, int count) const;
  [[nodiscard]] bool isPositionComplete(int pos) const;

  void updateKV(int layer_idx, int offset, int count, const Tensor& k, const Tensor& v);
  [[nodiscard]] std::array<Tensor, 2> getKV(int layer_idx);
  [[nodiscard]] std::array<Tensor, 2> getKV(int layer_idx, int offset, int count);

  void updateH(int layer_idx, int offset, int count, const Tensor& h);
  Tensor getH(int layer_idx, int offset, int count);

  // Layerwise loading interface
  void prepareForLayerwiseLoad();
  void loadLayerKCache(int layer_idx, int offset, int count);
  void loadLayerVCache(int layer_idx, int offset, int count);
  void loadLayerHCache(int layer_idx, int offset, int count);

 private:
  const size_t ELEMENT_SIZE = bytesOfType(kFloat32) / lanesOfType(kFloat32);

  // Returns (start_pos, count) for positions that need to be written.
  // Returns count=0 if nothing needs to be written.
  std::pair<int, int> findWriteRange(const std::function<bool(int)>& shouldWrite) const;

  // Returns the total number of elements written
  int writeHCacheLayer(std::fstream& file, int layer);
  void writeKVCacheLayer(std::fstream& file, const Tensor& cache, int layer);

  void loadMetadata();
  void saveMetadata() const;

  void allocateCaches();
  void initLoadedState();
  void loadLayerKVCacheImpl(std::ifstream& file, std::vector<Tensor>& cache, std::vector<std::vector<bool>>& loaded,
                            int layer_idx, int offset, int count);
  void markLoaded(std::vector<std::vector<bool>>& loaded, int layer_idx, int offset, int count);

  std::filesystem::path path_;

  int max_length_;
  int layer_nums_;
  int q_heads_;
  int kv_heads_;
  int kv_dim_;
  int hidden_size_;
  int started_ = 0;

  std::vector<int8_t> layer_watermark_;       // -1 = not computed, N = h_cache_[0..N] valid
  std::vector<int8_t> last_saved_watermark_;  // tracks what's been persisted

  std::vector<int64_t> input_tokens_;
  std::vector<Tensor> k_cache_;  // Shape: [layer_nums, 1, q_heads, max_cache_length, kv_dims]
  std::vector<Tensor> v_cache_;  // Shape: [layer_nums, 1, q_heads, max_cache_length, kv_dims]
  std::vector<Tensor> h_cache_;  // Shape: [layer_nums + 1, 1, max_cache_length, hidden_size]

  std::ifstream k_cache_file_;
  std::ifstream v_cache_file_;
  std::ifstream h_cache_file_;

  std::vector<std::vector<bool>> k_loaded_;  // [layer_nums_][max_length_]
  std::vector<std::vector<bool>> v_loaded_;  // [layer_nums_][max_length_]
  std::vector<std::vector<bool>> h_loaded_;  // [layer_nums_ + 1][max_length_]
};

}  // namespace mllm::models::qwen3_i
