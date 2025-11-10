#pragma once

#include <cudf/contiguous_split.hpp>
#include <inttypes.h>
#include <ucxx/api.h>
#include <memory>
#include "protocol.h"
#include "CommElement.h"
#include "EndpointRef.h"
#include "StateTransitionMetrics.h"

class Sender : public CommElement, public std::enable_shared_from_this<Sender> {
 public:
  static std::shared_ptr<Sender> create(
      const std::shared_ptr<Communicator> communicator,
      std::shared_ptr<EndpointRef> endpointRef,
      const std::string& key,
      uint64_t initialValue);

  void process() override;

  void close() override;

  std::string toString() const override;

  /// @return Reference to the state transition metrics.
  const StateTransitionMetrics& getStateTransitionMetrics() const {
    return stateMetrics_;
  }

 private:
  enum class ServerState : uint32_t {
    Created,
    ReadyToTransfer,
    WaitingForDataFromQueue,
    DataReady,
    WaitingForSendComplete,
    Done
  };

  Sender(
      const std::shared_ptr<Communicator> communicator,
      std::shared_ptr<EndpointRef> endpointRef,
      const std::string& key,
      uint64_t initialValue);

  /// @return A shared pointer to itself.
  std::shared_ptr<Sender> getSelfPtr();

  /// @brief Sends metadata and data to the connected receiver.
  void sendData();

  /// @brief Completion handler after data has been sent.
  void sendComplete(ucs_status_t status, std::shared_ptr<void> arg);

  /// @brief Sets the new state of this exchange server using
  /// sequential consistency and records the transition metrics.
  /// @param newState the new state of the CudfExchangeServer.
  /// @param bytes Optional: number of bytes transferred during this transition.
  void setState(ServerState newState, uint64_t bytes = 0) {
    ServerState oldState = state_.load(std::memory_order_seq_cst);
    if (oldState != newState) {
      auto endTime = std::chrono::high_resolution_clock::now();
      stateMetrics_.recordTransition(
          getStateNameForEnum(oldState),
          getStateNameForEnum(newState),
          lastStateChangeTime_,
          endTime,
          bytes);
      lastStateChangeTime_ = endTime;
    }
    state_.store(newState, std::memory_order_seq_cst);
  }

  /// @brief Returns the state.
  ServerState getState() const {
    return state_.load(std::memory_order_seq_cst);
  }

  /// @brief Converts ServerState enum to string representation.
  static std::string getStateNameForEnum(ServerState state) {
    const std::string stateMap[] = {
        "Created",
        "ReadyToTransfer",
        "WaitingForDataFromQueue",
        "DataReady",
        "WaitingForSendComplete",
        "Done"};
    return stateMap[static_cast<uint32_t>(state)];
  }

  std::string key_; // The unique identifier of the connected receiver.
  uint32_t keyHash_; // A hash of above, used to create unique tags.

  std::atomic<ServerState> state_;
  std::unique_ptr<cudf::packed_columns> dataPtr_;
  std::atomic<bool> closed_{false};

  uint32_t sequenceNumber_{0};
  HandshakeMsg handshake_;

  // The outstanding requests - there can only be one outstanding request
  // of each type at any point in time.
  // NOTE: The request owns/holds references to the upcall function
  // and must therefore exist until the upcall is done.
  std::shared_ptr<ucxx::Request> metaRequest_{nullptr};
  std::shared_ptr<ucxx::Request> dataRequest_{nullptr};

  std::chrono::time_point<std::chrono::high_resolution_clock> sendStart_;
  std::size_t bytes_;

  StateTransitionMetrics stateMetrics_;
  std::chrono::time_point<std::chrono::high_resolution_clock>
      lastStateChangeTime_{std::chrono::high_resolution_clock::now()};

  // For testing only:
  std::unique_ptr<cudf::packed_columns> makePackedColumns(
      std::size_t numRows,
      uint64_t initialValue,
      rmm::cuda_stream_view stream = cudf::get_default_stream(),
      rmm::device_async_resource_ref mr =
          cudf::get_current_device_resource_ref());

  uint32_t numExchanges_;
  uint64_t initialValue_;
};
