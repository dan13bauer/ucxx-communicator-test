#pragma once

#include <cudf/aggregation.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/contiguous_split.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/table/table.hpp>
#include <atomic>
#include <memory>
#include <thread>
#include "Receiver.h"

/// @brief Aggregator that computes the sum of values in the first column
/// of data received from a Receiver's queue using cuDF's sum aggregation.
class SumAggregator {
 public:
  /// @brief Constructor
  /// @param receiver The receiver whose data queue will be processed
  explicit SumAggregator(std::shared_ptr<Receiver> receiver);

  /// @brief Destructor - stops the aggregator thread
  ~SumAggregator();

  /// @brief Start the aggregator worker thread
  void start();

  /// @brief Stop the aggregator worker thread
  void stop();

  /// @brief Get the current cumulative sum
  /// @return The cumulative sum of all processed values
  int64_t getCumulativeSum() const;

  /// @brief Get the number of chunks processed
  /// @return The number of data chunks that have been aggregated
  uint64_t getChunksProcessed() const;

 private:
  /// @brief Worker thread function that processes data from the queue
  void workerThread();

  /// @brief Process a single ReceivedData item
  /// @param data The received data containing packed columns
  void processData(std::shared_ptr<Receiver::ReceivedData> data);

  std::shared_ptr<Receiver> receiver_;
  std::atomic<bool> running_{false};
  std::unique_ptr<std::thread> workerThread_;

  // Aggregated results
  std::atomic<int64_t> cumulativeSum_{0};
  std::atomic<uint64_t> chunksProcessed_{0};
};
