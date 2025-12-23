// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include <string>
#include <vector>
#include <map>
#include <chrono>
#include <fstream>
#include <algorithm>
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
  
  // 启用 tracing，重置开始时间
  void enable() {
    enabled_ = true;
    start_time_ = std::chrono::high_resolution_clock::now();
  }
  
  // 禁用 tracing
  void disable() {
    enabled_ = false;
  }
  
  // 检查是否启用
  [[nodiscard]] bool isEnabled() const {
    return enabled_;
  }
  
  [[nodiscard]] const std::vector<std::unique_ptr<Event>>& getEvents() const { 
    return events_; 
  }
  
  void clear() {
    events_.clear();
  }
  
  [[nodiscard]] size_t size() const { 
    return events_.size(); 
  }
  
  [[nodiscard]] bool exportToCSV(const std::string& filepath) const {
    std::ofstream file(filepath);
    if (!file.is_open()) {
      return false;
    }
    
    // Collect all unique keys
    std::vector<std::string> keys;
    for (const auto& event : events_) {
      const auto& data = event->toData();
      for (const auto& [key, value] : data) {
        if (std::find(keys.begin(), keys.end(), key) == keys.end()) {
          keys.push_back(key);
        }
      }
    }
    
    // Write header
    file << "type,timestamp_us";
    for (const auto& key : keys) {
      file << "," << key;
    }
    file << "\n";
    
    // Write events
    for (const auto& event : events_) {
      file << event->typeName() << ",";
      
      auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        event->timestamp() - start_time_);
      file << duration.count();
      
      const auto& data = event->toData();
      for (const auto& key : keys) {
        file << ",";
        auto it = data.find(key);
        if (it != data.end()) {
          file << it->second;
        }
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
