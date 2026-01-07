#include "mllm/utils/Tracing.hpp"
#include "mllm/mllm.hpp"

namespace mllm::models::qwen3_i {

template<typename Derived>
class LayerEvent : public Event {
 public:
  LayerEvent(int layer_idx, int seq_len, int token_idx) : layer_idx_(layer_idx), seq_len_(seq_len), token_idx_(token_idx) {}

  [[nodiscard]] std::map<std::string, std::string> toData() const override {
    return {{"layer_idx", std::to_string(layer_idx_)},
            {"seq_len", std::to_string(seq_len_)},
            {"token_idx", std::to_string(token_idx_)}};
  }

  [[nodiscard]] const char* typeName() const noexcept override { return Derived::kTypeName; }

 private:
  int layer_idx_;
  int seq_len_;
  int token_idx_;
};

struct LayerBeginEvent final : public LayerEvent<LayerBeginEvent> {
  using LayerEvent::LayerEvent;
  static constexpr const char* kTypeName = "LayerBegin";
};

struct LayerCompleteEvent final : public LayerEvent<LayerCompleteEvent> {
  using LayerEvent::LayerEvent;
  static constexpr const char* kTypeName = "LayerComplete";
};

struct KVCacheCompleteEvent final : public LayerEvent<KVCacheCompleteEvent> {
  using LayerEvent::LayerEvent;
  static constexpr const char* kTypeName = "KVCacheComplete";
};

struct SelfAttentionCompleteEvent final : public LayerEvent<SelfAttentionCompleteEvent> {
  using LayerEvent::LayerEvent;
  static constexpr const char* kTypeName = "SelfAttentionComplete";
};

struct MLPBeginEvent final : public LayerEvent<MLPBeginEvent> {
  using LayerEvent::LayerEvent;
  static constexpr const char* kTypeName = "MLPBegin";
};

struct MLPCompleteEvent final : public LayerEvent<MLPCompleteEvent> {
  using LayerEvent::LayerEvent;
  static constexpr const char* kTypeName = "MLPComplete";
};

struct SyncStartEvent final : public LayerEvent<SyncStartEvent> {
  using LayerEvent::LayerEvent;
  static constexpr const char* kTypeName = "SyncStart";
};

struct SyncCompleteEvent final : public LayerEvent<SyncCompleteEvent> {
  using LayerEvent::LayerEvent;
  static constexpr const char* kTypeName = "SyncComplete";
};

struct CheckpointBeginEvent final : public Event {
  [[nodiscard]] std::map<std::string, std::string> toData() const override { return {}; }
  [[nodiscard]] const char* typeName() const noexcept override { return "CheckpointBegin"; }
};

struct CheckpointCompleteEvent final : public Event {
  explicit CheckpointCompleteEvent(int count) : count_(count) {}
  [[nodiscard]] std::map<std::string, std::string> toData() const override { return {{"count", std::to_string(count_)}}; }
  [[nodiscard]] const char* typeName() const noexcept override { return "CheckpointComplete"; }
 private:
  int count_;
};

template<typename EventType>
static inline void recordEvent(int layer_idx, int seq_len, int token_idx) {
  Context::instance().tracer()->record<EventType>(layer_idx, seq_len, token_idx);
}


}  // namespace mllm::models::qwen3_i
