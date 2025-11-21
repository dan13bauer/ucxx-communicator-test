#include "SumAggregator.h"
#include <cudf/stream_compaction.hpp>
#include <cudf/types.hpp>
#include <iostream>
#include <chrono>
#include <thread>

SumAggregator::SumAggregator(std::shared_ptr<Receiver> receiver)
    : receiver_(std::move(receiver)) {}

SumAggregator::~SumAggregator() {
  stop();
}

void SumAggregator::start() {
  if (running_.exchange(true)) {
    std::cerr << "SumAggregator already running" << std::endl;
    return;
  }

  std::cout << "Starting SumAggregator for " << receiver_->toString() << std::endl;
  workerThread_ = std::make_unique<std::thread>(&SumAggregator::workerThread, this);
}

void SumAggregator::stop() {
  if (!running_.exchange(false)) {
    return; // Already stopped
  }

  std::cout << "Stopping SumAggregator..." << std::endl;
  if (workerThread_ && workerThread_->joinable()) {
    workerThread_->join();
  }

  std::cout << "SumAggregator stopped. Final statistics:" << std::endl;
  std::cout << "  Chunks processed: " << chunksProcessed_.load() << std::endl;
  std::cout << "  Cumulative sum: " << cumulativeSum_.load() << std::endl;
}

int64_t SumAggregator::getCumulativeSum() const {
  return cumulativeSum_.load();
}

uint64_t SumAggregator::getChunksProcessed() const {
  return chunksProcessed_.load();
}

void SumAggregator::workerThread() {
  std::cout << "SumAggregator worker thread started" << std::endl;

  while (running_.load()) {
    // Try to pop data from the receiver's queue
    auto data = receiver_->getDataQueue().pop();

    if (!data) {
      // Queue is empty or we received nullptr stop signal
      // With busy-waiting, continue to check again
      // The nullptr sentinel will eventually stop us when receiver is done
      continue;
    }

    // Process the data
    processData(data);
  }

  // Process any remaining items in the queue before exiting
  std::cout << "SumAggregator worker thread draining queue..." << std::endl;
  while (true) {
    auto data = receiver_->getDataQueue().pop();
    if (!data) {
      // Received nullptr - this is the stop signal, or queue is empty
      std::cout << "SumAggregator finished draining queue (received nullptr or empty)" << std::endl;
      break;
    }
    processData(data);
  }

  std::cout << "SumAggregator worker thread exited" << std::endl;
}

void SumAggregator::processData(std::shared_ptr<Receiver::ReceivedData> data) {
  try {
    auto startTime = std::chrono::high_resolution_clock::now();

    std::cout << "SumAggregator processing seq=" << data->sequenceNumber
              << ", columns ptr=" << (data->columns ? "valid" : "null") << std::endl;

    if (!data->columns) {
      std::cerr << "ERROR: columns unique_ptr is null for seq " << data->sequenceNumber << std::endl;
      return;
    }

    // Unpack the packed_columns back into a table
    cudf::table unpackedTable = cudf::unpack(*(data->columns));
    cudf::table_view tableView = unpackedTable.view();

    std::cout << "  Unpacked: num_columns=" << tableView.num_columns()
              << ", num_rows=" << tableView.num_rows() << std::endl;

    if (tableView.num_columns() == 0) {
      std::cerr << "Warning: Table has no columns, skipping aggregation" << std::endl;
      return;
    }

    // Get the first column
    cudf::column_view firstColumn = tableView.column(0);

    if (firstColumn.size() == 0) {
      std::cout << "Chunk " << data->sequenceNumber
                << " is empty, skipping" << std::endl;
      return;
    }

    // Create sum aggregation
    auto agg = cudf::make_sum_aggregation<cudf::reduce_aggregation>();

    // Perform reduction to compute the sum
    std::unique_ptr<cudf::scalar> result = cudf::reduce(
        firstColumn,
        *agg,
        firstColumn.type());

    // Cast the result to numeric scalar and get the value
    if (result->is_valid()) {
      auto numericScalar = static_cast<cudf::numeric_scalar<int64_t>*>(result.get());
      int64_t columnSum = numericScalar->value();

      // Add to cumulative sum
      int64_t oldSum = cumulativeSum_.fetch_add(columnSum);
      chunksProcessed_.fetch_add(1);

      auto endTime = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
          endTime - startTime).count();

      std::cout << "Chunk " << data->sequenceNumber
                << ": sum=" << columnSum
                << ", rows=" << firstColumn.size()
                << ", cumulative_sum=" << (oldSum + columnSum)
                << ", time=" << duration << "us"
                << std::endl;
    } else {
      std::cerr << "Warning: Sum result is invalid for chunk "
                << data->sequenceNumber << std::endl;
    }

  } catch (const std::exception& e) {
    std::cerr << "Error processing chunk " << data->sequenceNumber
              << ": " << e.what() << std::endl;
  }
}
