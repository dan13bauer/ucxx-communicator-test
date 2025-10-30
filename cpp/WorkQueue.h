#pragma once

#include <list>
#include <memory>
#include <mutex>

template <typename T>
class WorkQueue {
 public:
  WorkQueue() = default;
  ~WorkQueue() = default;

  // non-copyable
  WorkQueue(const WorkQueue&) = delete;
  WorkQueue& operator=(const WorkQueue&) = delete;

  // push always succeeds; never blocks
  void push(std::shared_ptr<T> item) {
    std::lock_guard<std::mutex> lock(mutex_);
    queue_.emplace_back(std::move(item));
  }

  // pop never blocks; returns nullptr if empty
  std::shared_ptr<T> pop() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (queue_.empty()) {
      return nullptr;
    }

    auto item = std::move(queue_.front());
    queue_.pop_front();
    return item;
  }

  // remove an element from the queue.
  bool erase(std::shared_ptr<T> item) {
    bool erased = false;
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = std::find(queue_.begin(), queue_.end(), item);
    if (it != queue_.end()) {
      queue_.erase(it);
      erased = true;
    }
    return erased;
  }

  // helpers
  bool empty() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.empty();
  }

  size_t size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.size();
  }

 private:
  mutable std::mutex mutex_;
  std::list<std::shared_ptr<T>> queue_;
};
