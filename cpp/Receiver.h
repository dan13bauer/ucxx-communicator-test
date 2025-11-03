#pragma once

#include <cudf/contiguous_split.hpp>
#include <inttypes.h>
#include <ucxx/api.h>
#include <memory>
#include "protocol.h"
#include "CommElement.h"
#include "EndpointRef.h"

class Receiver : public CommElement,
                 public std::enable_shared_from_this<Receiver> {
 public:
  static std::shared_ptr<Receiver> create(
      const std::shared_ptr<Communicator> communicator,
      const std::string& host,
      uint16_t port,
      const uint32_t receiverId);

  void process() override;

  void close();

  std::string toString();

 private:
  enum class ReceiverState : uint32_t {
    Created,
    WaitingForHandshakeComplete,
    ReadyToReceive,
    WaitingForMetadata,
    WaitingForData,
    Done
  };

  std::map<ReceiverState, std::string> receiverStateNames_ = {
    {ReceiverState::Created, "Created"},
    {ReceiverState::WaitingForHandshakeComplete, "WaitingForHandshakeComplete"},
    {ReceiverState::ReadyToReceive, "ReadyToReceive"},
    {ReceiverState::WaitingForMetadata, "WaitingForMetadata"},
    {ReceiverState::WaitingForData, "WaitingForData"},
    {ReceiverState::Done, "Done" }

  };


  struct DataAndMetadata {
    MetadataMsg metadata;
    std::unique_ptr<rmm::device_buffer> dataBuf;
  };

  /// @brief The constructor is private in order to ensure that Receivers are
  /// always generated through a shared pointer. This ensures that
  /// shared_from_this works properly.
  explicit Receiver(
      const std::shared_ptr<Communicator> communicator,
      const std::string& host,
      uint16_t port,
      const uint32_t receiverId);

  /// @return A shared pointer to itself.
  std::shared_ptr<Receiver> getSelfPtr();

  /// @brief Sets the endpoint for this receiver.
  void setEndpoint(std::shared_ptr<EndpointRef> endpointRef);

  /// @brief Sends a handshake request to the server. The endpoint must exist.
  void sendHandshake();

  /// @brief Called by the transport layer when handshake is completed
  /// @param status indication by transport layer of transfer status
  /// @param arg NOT USED at the moment
  void onHandshake(ucs_status_t status, std::shared_ptr<void> arg);

  /// @brief Waits for metadata and installs the onMetadata callback.
  void getMetadata();

  /// @brief Called by the transport layer when data is available
  /// @param status indication by transport layer of transfer status
  /// @param arg the serialized form of the metadata
  void onMetadata(ucs_status_t status, std::shared_ptr<void> arg);

  /// @brief Called by the transport layer when data is available
  /// @param status indication by transport layer of transfer status
  /// @param arg
  void onData(ucs_status_t status, std::shared_ptr<void> arg);

  /// @brief Sets the new state of this exchange source using
  /// sequential consistency.
  /// @param newState the new state of the CudfExchangeSource.
  void setState(ReceiverState newState) {
    state_.store(newState, std::memory_order_seq_cst);
  }

  /// @brief Returns the state.
  ReceiverState getState() {
    return state_.load(std::memory_order_seq_cst);
  }

  /// @brief Remove the state associated with the source called by the
  /// state-machine
  void cleanUp();

  /// @brief Sets the state to "desired" if and only if the current
  /// state is "expected".
  /// @param expected The expected state
  /// @param desired The desired state
  /// @return Returns true if state was changed, false otherwise.
  bool setStateIf(ReceiverState expected, ReceiverState desired);


  void dumpValues(
      std::unique_ptr<cudf::packed_columns> columns,
      MetadataMsg& metadata);

  // The connection parameters
  const std::string host_;
  uint16_t port_;

  // URL of the remote task producing data.
  const std::string key_;
  // the initial value requested from the server.
  const int initialValue_;
  const uint32_t taskIdHash_;
  std::atomic<ReceiverState> state_;

  uint32_t sequenceNumber_{0};
  std::atomic<bool> closed_{false};
  bool atEnd_{false}; // set when "atEnd" is being received.

  // The outstanding request - there can only be one outstanding request
  // at any point in time. Used for handshake, metadata and data.
  // NOTE: The request owns/holds a reference to the upcall function
  // and must therefore exist until the upcall is done.
  std::shared_ptr<ucxx::Request> request_{nullptr};
};
