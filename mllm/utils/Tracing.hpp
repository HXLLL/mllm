// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include <string>
#include <vector>
#include <map>
#include <set>
#include <chrono>
#include <fstream>
#include <memory>

namespace mllm {

class Event {
public:
  Event() : timestamp_(std::chrono::high_resolution_clock::now()) {}
  virtual ~Event() = default;
  
  [[nodiscard]] virtual std::map<std::string, std::string> toData() const = 0;
  [[nodiscard]] virtual const char* typeName() const noexcept = 0;
  
  [[nodiscard]] std::chrono::high_resolution_clock::time_point timestamp() const {
    return timestamp_;
  }

protected:
  std::chrono::high_resolution_clock::time_point timestamp_;
};

template <typename Derived>
class MetaEvent : public Event {
public:
  [[nodiscard]] std::map<std::string, std::string> toData() const override { return {}; }
  [[nodiscard]] const char* typeName() const noexcept override {
    return Derived::kTypeName;
  }
};

struct StartTracingEvent final : public MetaEvent<StartTracingEvent> {
  static constexpr const char* kTypeName = "StartTracing";
};

struct StopTracingEvent final : public MetaEvent<StopTracingEvent> {
  static constexpr const char* kTypeName = "StopTracing";
};

class Tracer {
public:
  using ptr_t = std::shared_ptr<Tracer>;

  explicit Tracer(size_t initial_capacity = 1024) 
    : start_time_(std::chrono::high_resolution_clock::now()) {
    events_.reserve(initial_capacity);
  }
  
  void record(std::unique_ptr<Event> event) {
    if (enabled_) {
      events_.push_back(std::move(event));
    }
  }
  
  template<typename EventType, typename... Args>
  void record(Args&&... args) {
    if (enabled_) {
      events_.emplace_back(std::make_unique<EventType>(std::forward<Args>(args)...));
    }
  }
  
  void enable() {
    enabled_ = true;
    start_time_ = std::chrono::high_resolution_clock::now();
    record<StartTracingEvent>();
  }
  
  void disable() {
    if (enabled_) record<StopTracingEvent>();
    enabled_ = false;
  }
  
  [[nodiscard]] bool isEnabled() const { return enabled_; }
  
  [[nodiscard]] const std::vector<std::unique_ptr<Event>>& getEvents() const { 
    return events_; 
  }
  
  void clear() {
    events_.clear();
  }
  
  [[nodiscard]] size_t size() const { 
    return events_.size(); 
  }
  
  [[nodiscard]] bool exportToCSV(const std::string& filepath, bool append = false) const {
    std::ifstream check(filepath);
    bool file_exists = check.good();
    check.close();
    
    std::ofstream file(filepath, append && file_exists ? std::ios::app : std::ios::trunc);
    if (!file.is_open()) return false;
    file.imbue(std::locale::classic());
    
    std::set<std::string> keys;
    for (const auto& event : events_) {
      for (const auto& [key, value] : event->toData()) {
        keys.insert(key);
      }
    }
    
    if (!append || !file_exists) {
      file << "type,timestamp_us";
      for (const auto& key : keys) file << "," << key;
      file << "\n";
    }
    
    for (const auto& event : events_) {
      file << event->typeName() << ","
           << std::chrono::duration_cast<std::chrono::microseconds>(
                event->timestamp() - start_time_).count();
      const auto& data = event->toData();
      for (const auto& key : keys) {
        file << ",";
        auto it = data.find(key);
        if (it != data.end()) file << it->second;
      }
      file << "\n";
    }
    
    return true;
  }

private:
  std::vector<std::unique_ptr<Event>> events_;
  std::chrono::high_resolution_clock::time_point start_time_;
  bool enabled_ = false;
};

} // namespace mllm
