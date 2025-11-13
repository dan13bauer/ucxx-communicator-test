#include <cuda_runtime.h>
#include <gflags/gflags.h>
#include <rmm/cuda_stream_pool.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>
#include <chrono>
#include <csignal>
#include <iostream>
#include <thread>
#include <random>
#include "util.h"
#include "Communicator.h"
#include "Receiver.h"

DEFINE_uint32(listener_port, 4567, "Port number of the listener");
DEFINE_string(ports, "", "Comma-separated list of port numbers to which to connect to");
DEFINE_string(hostnames, "127.0.0.1", "Comma-separated list of host names to which to connect to");
DEFINE_uint32(num_chunks, 10, "Number of chunks");
DEFINE_uint64(rows, 1, "Number of rows");
DEFINE_string(
    mem_mgr,
    "pool",
    "Cuda device resource manager: none | cuda | pool | async | arena | managed | managed_pool");
DEFINE_bool(ucxx_error_handling, true, "Whether to use UCXX error handling");
DEFINE_bool(ucxx_blocking_polling, true, "Use blocking polling (true) or spinning polling (false)");

// Assisted by watsonx Code Assistant

std::vector<uint32_t> stringToVector(const std::string& str) {
  std::vector<uint32_t> vec;
  std::stringstream ss(str);
  std::string token;

  while (std::getline(ss, token, ',')) {
    // Trim whitespace from token
    token.erase(0, token.find_first_not_of(" \t\n\r\f\v"));
    token.erase(token.find_last_not_of(" \t\n\r\f\v") + 1);
    if (!token.empty()) {
      try {
        uint32_t num = std::stoul(token);
        vec.push_back(num);
      } catch (const std::invalid_argument& e) {
        // Handle invalid conversions (optional)
        std::cerr << "Invalid number: " << token << std::endl;
      }
    }
  }

  return vec;
}

std::vector<std::string> stringToHostnames(const std::string& str) {
  std::vector<std::string> vec;
  std::stringstream ss(str);
  std::string token;

  while (std::getline(ss, token, ',')) {
    // Trim whitespace from token
    token.erase(0, token.find_first_not_of(" \t\n\r\f\v"));
    token.erase(token.find_last_not_of(" \t\n\r\f\v") + 1);
    if (!token.empty()) {
      vec.push_back(token);
    }
  }

  return vec;
}

static std::shared_ptr<Communicator> communicatorPtr;

void signalHandler(int signal) {
  if (signal == SIGTERM) {
    std::cout << "Caught signal - stopping." << std::endl;
    communicatorPtr->stop();
  }
}

using namespace std::chrono_literals;


uint32_t getRandomTaskId() {
    // Create a random device and seed the generator
    std::random_device rd;
    std::mt19937 gen(rd()); // Mersenne Twister engine
    
    // Define the range for positive integers (example: 1 to 100)
    std::uniform_int_distribution<> dist(1, 1000000);

    // Generate a random number
    return dist(gen);

}

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  // Tell UCX to re-use the port.
  setenv("UCX_TCP_CM_REUSEADDR", "y", 1);

  // Force CUDA context creation
  cudaFree(0);

  // configure a cuda memory manager.
  auto mr = createMemoryResource(FLAGS_mem_mgr);
  rmm::cuda_stream_view stream = cudf::get_default_stream();
  if (mr) {
    rmm::mr::set_current_device_resource(mr.get());
  }

  // Setup the communicator with the listener on the given port.
  auto communicator = Communicator::initAndGet(FLAGS_listener_port);
  communicatorPtr = communicator;  // Fixed: Use the shared_ptr directly, don't create a new one from raw pointer
  std::signal(SIGTERM, signalHandler);

  std::thread commThread =
      std::thread([communicator]() { communicator->run(); });

  std::cout << "Started communicator. Waiting before starting clients. "
            << std::endl;
  // Sleep before starting the clients
  std::this_thread::sleep_for(5s);

  // create as many receivers as ports are given.
  std::vector<std::shared_ptr<Receiver>> receivers;

  std::vector<uint32_t> portsVec = stringToVector(FLAGS_ports);
  std::vector<std::string> hostnamesVec = stringToHostnames(FLAGS_hostnames);

  // Validate that hostnames and ports lists have equal length
  if (!portsVec.empty() && hostnamesVec.size() != portsVec.size()) {
    std::cerr << "Error: Number of hostnames (" << hostnamesVec.size()
              << ") does not match number of ports (" << portsVec.size() << ")" << std::endl;
    std::cerr << "Please provide equal number of comma-separated hostnames and ports." << std::endl;
    return 1;
  }

  uint32_t taskId = getRandomTaskId();

  for (size_t i = 0; i < portsVec.size(); ++i) {
    // create and start a receiver using corresponding hostname and port
    std::cout << "Creating receiver from taskId " << taskId
              << " for " << hostnamesVec[i] << ":" << portsVec[i] << std::endl;
    std::shared_ptr<Receiver> recv =
        Receiver::create(communicator, hostnamesVec[i], portsVec[i], taskId);
    receivers.push_back(recv);
    communicator->registerCommElement(receivers.back());
    taskId++;
  }

  // Communicator will stop when the last communication element is finished
  // and has de-registered itself. Join in the thread.
  commThread.join();

  if (receivers.size() == 0) {
    return 0;
  }

  // Aggregate metrics from all receivers using TransitionRecord vectors
  std::cout << "\n=== Aggregating metrics from " << receivers.size()
            << " receivers ===\n" << std::endl;

  // Collect all TransitionRecords from all receivers, grouped by receiver and transition type
  std::map<std::pair<std::string, std::string>,
           std::vector<std::vector<StateTransitionMetrics::TransitionRecord>>> recordsByTransition;

  // Iterate through all receivers and collect their transition records
  for (size_t i = 0; i < receivers.size(); ++i) {
    const auto& receiverMetrics = receivers[i]->getStateTransitionMetrics();
    const auto& history = receiverMetrics.getTransitionHistory();

    std::cout << "Receiver " << i << " has " << history.size()
              << " transition records." << std::endl;

    // First pass: group records by transition type for this receiver
    std::map<std::pair<std::string, std::string>,
             std::vector<StateTransitionMetrics::TransitionRecord>> receiverRecords;
    for (const auto& record : history) {
      auto key = std::make_pair(record.fromState, record.toState);
      receiverRecords[key].push_back(record);
    }

    // Second pass: add this receiver's records to the global map
    for (const auto& [key, records] : receiverRecords) {
      if (recordsByTransition.find(key) == recordsByTransition.end()) {
        recordsByTransition[key] = std::vector<std::vector<StateTransitionMetrics::TransitionRecord>>();
      }
      recordsByTransition[key].push_back(records);
    }
  }

  // Structure to hold aggregated statistics
  std::map<std::pair<std::string, std::string>, StateTransitionMetrics::TransitionStats>
      aggregatedStats;

  // Process each transition type: align records by position, average durations, sum bytes
  for (auto& [transitionKey, receiverRecordsVec] : recordsByTransition) {
    if (receiverRecordsVec.empty()) {
      continue;
    }

    // Find the minimum number of records across all receivers for this transition type
    size_t minRecords = receiverRecordsVec[0].size();
    for (const auto& receiverRecords : receiverRecordsVec) {
      minRecords = std::min(minRecords, receiverRecords.size());
    }

    std::cout << "  " << transitionKey.first << " -> " << transitionKey.second
              << ": " << receiverRecordsVec.size() << " receivers, "
              << "min records per receiver=" << minRecords << std::endl;

    if (minRecords == 0) {
      continue;
    }

    // Calculate 5% boundaries based on the minimum record count
    size_t trimCount = static_cast<size_t>(minRecords * 0.05);
    size_t startIdx = trimCount;
    size_t endIdx = minRecords - trimCount;

    // If we have too few records, use all of them
    if (endIdx <= startIdx) {
      startIdx = 0;
      endIdx = minRecords;
    }

    std::cout << "    Using records [" << startIdx << ".." << endIdx << ")"
              << " (trimmed " << (startIdx + (minRecords - endIdx)) << ")" << std::endl;

    // Compute statistics from the trimmed range
    StateTransitionMetrics::TransitionStats stats;
    stats.count = 0;
    stats.totalDurationMicros = 0;
    stats.totalBytes = 0;
    stats.minDurationMicros = INT64_MAX;
    stats.maxDurationMicros = INT64_MIN;
    stats.minBytes = UINT64_MAX;
    stats.maxBytes = 0;

    // For each position i in the trimmed range
    for (size_t i = startIdx; i < endIdx; ++i) {
      // Average duration across all receivers at position i
      int64_t totalDuration = 0;
      uint64_t totalBytes = 0;
      int64_t minDur = INT64_MAX;
      int64_t maxDur = INT64_MIN;
      uint64_t minByt = UINT64_MAX;
      uint64_t maxByt = 0;

      for (const auto& receiverRecords : receiverRecordsVec) {
        const auto& record = receiverRecords[i];
        totalDuration += record.durationMicros;
        totalBytes += record.bytes;  // Sum bytes across receivers
        minDur = std::min(minDur, record.durationMicros);
        maxDur = std::max(maxDur, record.durationMicros);
        minByt = std::min(minByt, record.bytes);
        maxByt = std::max(maxByt, record.bytes);
      }

      // Average duration across receivers
      int64_t avgDuration = totalDuration / receiverRecordsVec.size();

      stats.count++;
      stats.totalDurationMicros += avgDuration;
      stats.totalBytes += totalBytes;
      stats.minDurationMicros = std::min(stats.minDurationMicros, minDur);
      stats.maxDurationMicros = std::max(stats.maxDurationMicros, maxDur);
      stats.minBytes = std::min(stats.minBytes, minByt);
      stats.maxBytes = std::max(stats.maxBytes, maxByt);
    }

    // Handle edge case where no records were processed
    if (stats.count == 0) {
      stats.minDurationMicros = 0;
      stats.minBytes = 0;
    }

    aggregatedStats[transitionKey] = stats;
  }

  // Display aggregated statistics
  std::cout << "\n=== AGGREGATED METRICS FROM ALL RECEIVERS ===\n";
  std::cout << "  (First and last 5% of records trimmed for each transition type)\n\n";

  if (aggregatedStats.empty()) {
    std::cout << "  No transitions recorded across all receivers.\n";
  } else {
    std::cout << "  Total unique transition types: " << aggregatedStats.size() << "\n\n";

    for (const auto& [key, stats] : aggregatedStats) {
      std::cout << "  " << key.first << " -> " << key.second << ":\n";
      std::cout << "    Count: " << stats.count << "\n";
      std::cout << "    Total duration: " << stats.totalDurationMicros << " µs\n";
      std::cout << "    Average duration: " << stats.getAverageMicros() << " µs\n";
      std::cout << "    Min duration: " << stats.minDurationMicros << " µs\n";
      std::cout << "    Max duration: " << stats.maxDurationMicros << " µs\n";
      std::cout << "    Total bytes: " << stats.totalBytes << "\n";
      std::cout << "    Average bytes: " << stats.getAverageBytes() << "\n";
      std::cout << "    Min bytes: " << stats.minBytes << "\n";
      std::cout << "    Max bytes: " << stats.maxBytes << "\n";

      // Only output throughput if bytes were transferred
      if (stats.totalBytes > 0) {
        std::cout << "    Overall throughput: " << stats.getThroughputMBps()
                  << " MB/s\n";
        std::cout << "    Average throughput: " << stats.getAverageThroughputMBps()
                  << " MB/s\n";
      }
      std::cout << "\n";
    }
  }

  std::cout << "=== END OF AGGREGATED METRICS ===\n" << std::endl;
}
