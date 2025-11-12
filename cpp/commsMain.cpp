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
DEFINE_string(ports, "", "List of port numbers to which to connect to");
DEFINE_string(hostname, "127.0.0.1", "Host name to which to connect to");
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
    try {
      uint32_t num = std::stoul(token);
      vec.push_back(num);
    } catch (const std::invalid_argument& e) {
      // Handle invalid conversions (optional)
      std::cerr << "Invalid number: " << token << std::endl;
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

  uint32_t taskId = getRandomTaskId();

  for (auto port : portsVec) {
    // create and start a receiver
    std::cout << "Creating receiver from taskId " << taskId << std::endl;
    std::shared_ptr<Receiver> recv =
        Receiver::create(communicator, FLAGS_hostname, port, taskId);
    receivers.push_back(recv);
    communicator->registerCommElement(receivers.back());
    taskId++;
  }

  // Communicator will stop when the last communication element is finished
  // and has de-registered itself. Join in the thread.
  commThread.join();
}
