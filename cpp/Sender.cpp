#include "Sender.h"
#include <cudf/column/column_factories.hpp>
#include <gflags/gflags.h> // FIXME: for testing.
#include <numeric>
#include <sstream>
#include <stdexcept>
#include "protocol.h"
#include "Communicator.h"

// FIXME: for testing
DECLARE_uint32(num_chunks);
DECLARE_uint64(rows);

// This constructor is private.
Sender::Sender(
    std::shared_ptr<Communicator> communicator,
    std::shared_ptr<EndpointRef> endpointRef,
    const std::string& key,
    uint64_t initialValue)
    : CommElement(communicator, endpointRef),
      key_(key),
      keyHash_(fnv1a_32(key)),
      numExchanges_(FLAGS_num_chunks),
      initialValue_(initialValue) {
          setState(ServerState::Created);
      }

// static
std::shared_ptr<Sender> Sender::create(
    const std::shared_ptr<Communicator> communicator,
    std::shared_ptr<EndpointRef> endpointRef,
    const std::string& key,
    uint64_t initialValue) {
  auto ptr = std::shared_ptr<Sender>(
      new Sender(communicator, endpointRef, key, initialValue));
  return ptr;
}

void Sender::process() {
  switch (state_) {
    case ServerState::Created:
      // Allocate the packed columns structure once
      dataPtr_ = makePackedColumns(FLAGS_rows, initialValue_);
      setState(ServerState::ReadyToTransfer);
      communicator_->addToWorkQueue(getSelfPtr());
      break;
    case ServerState::ReadyToTransfer:
      setState(ServerState::WaitingForDataFromQueue);

      // Reuse the same dataPtr_ until we reach the end
      if (sequenceNumber_ >= numExchanges_) {
        dataPtr_ = nullptr; // signal that we are at the end.
      }
      this->setState(ServerState::DataReady);
      communicator_->addToWorkQueue(getSelfPtr());
      break;
    case ServerState::WaitingForDataFromQueue:
      // Waiting for data is handled by an upcall from the data queue. Nothing
      // to do
      break;
    case ServerState::DataReady:
      sendData();
      break;
    case ServerState::WaitingForSendComplete:
      // Waiting for send complete is handled by an upcall from UCXX. Nothing to
      // do
      break;
    case ServerState::Done:
      close();
      if (endpointRef_) {
        endpointRef_->removeCommElem(getSelfPtr());
        endpointRef_ = nullptr;
      }
      break;
  };
}

void Sender::close() {
  bool expected = false;
  bool desired = true;
  if (!closed_.compare_exchange_strong(expected, desired)) {
    return; // already closed.
  }
  std::cout << "Close Sender to remote " << key_;
  communicator_->unregister(getSelfPtr());
}

std::string Sender::toString() const {
  std::stringstream out;
  out << "[Sender " << key_ << " - " << initialValue_ << ":" << sequenceNumber_
      << "]";
  return out.str();
}

// ------ private methods ---------

std::shared_ptr<Sender> Sender::getSelfPtr() {
  return shared_from_this();
}

void Sender::sendData() {
  // Create the MetaDataRecord.
  std::shared_ptr<MetadataMsg> metadataMsg = std::make_shared<MetadataMsg>();

  if (dataPtr_) {
    metadataMsg->cudfMetadata = std::move(dataPtr_->metadata);
    metadataMsg->dataSizeBytes = dataPtr_->gpu_data->size();
    metadataMsg->remainingBytes = {};
    metadataMsg->atEnd = false;
  } else {
    std::cout << "Final exchange for " << key_ << std::endl;
    metadataMsg->cudfMetadata = nullptr;
    metadataMsg->dataSizeBytes = 0;
    metadataMsg->remainingBytes = {};
    metadataMsg->atEnd = true;
  }

  auto [serializedMetadata, serMetaSize] = metadataMsg->serialize();

  // send metadata, no callback needed.
  uint64_t metadataTag =
      getMetadataTag(this->keyHash_, this->sequenceNumber_);
  metaRequest_ = endpointRef_->endpoint_->tagSend(
      serializedMetadata.get(),
      serMetaSize,
      ucxx::Tag{metadataTag},
      false,
      [tid = key_, metadataTag, this](
          ucs_status_t status, std::shared_ptr<void> arg) {
        if (status != UCS_OK) {
          std::cerr << "Error in sendData, send metadata "
                  << ucs_status_string(status) << " failed for task: " << tid << std::endl;
          this->setState(ServerState::Done);
          this->communicator_->addToWorkQueue(getSelfPtr());
        }
      },
      serializedMetadata);

  // send the data chunk (if any)
  if (dataPtr_) {
      sendStart_ = std::chrono::high_resolution_clock::now();
      bytes_ = dataPtr_->gpu_data->size();

    setState(ServerState::WaitingForSendComplete, bytes_);
    uint64_t dataTag = getDataTag(this->keyHash_, this->sequenceNumber_);
    dataRequest_ = endpointRef_->endpoint_->tagSend(
        dataPtr_->gpu_data->data(),
        dataPtr_->gpu_data->size(),
        ucxx::Tag{dataTag},
        false,
        std::bind(
            &Sender::sendComplete,
            this,
            std::placeholders::_1,
            std::placeholders::_2));
  } else {
    // Data pointer is null, so no more data will be coming.
    std::cout << "Finished transferring partition for task " << key_
              << std::endl;
    std::cout << std::endl << stateMetrics_.toString() << std::endl;
    setState(ServerState::Done);
    communicator_->addToWorkQueue(getSelfPtr());
  }
}

void Sender::sendComplete(
  ucs_status_t status,
  std::shared_ptr<void> arg) {
  if (status == UCS_OK) {
    CHECK(dataPtr_ != nullptr, "dataPtr_ is null");

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = end - sendStart_;
    auto micros =
        std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
    auto throughput = bytes_ / micros;

    std::cout << "duration: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(duration)
                   .count()
            << " ms " << std::endl;
    std::cout << "throughput: " << throughput << " MByte/s" << std::endl;

    this->sequenceNumber_++;
    // Don't reset dataPtr_ here - we reuse it for all sends
    setState(ServerState::ReadyToTransfer, bytes_);
  } else {
    std::cerr << "Error in sendComplete, send complete "
              << ucs_status_string(status) << std::endl;
    setState(ServerState::Done);
  }
  communicator_->addToWorkQueue(getSelfPtr());
}

// ---- for testing only ---

std::unique_ptr<cudf::packed_columns> Sender::makePackedColumns(
    std::size_t numRows,
    uint64_t initialValue,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  // Create a numeric column using cudf::make_numeric_column
  auto counterCol = cudf::make_numeric_column(
      cudf::data_type{cudf::type_id::INT64},
      numRows,
      cudf::mask_state::UNALLOCATED, // no nulls.
      stream,
      mr);

  // fill with some recognizable data.
  std::size_t len = numRows;
  auto mutable_view = counterCol->mutable_view();

  // Cast the underlying data pointer to uint64_t*
  uint64_t* data1 = mutable_view.template data<uint64_t>();

  std::vector<uint64_t> vec1(len);
  std::iota(vec1.begin(), vec1.end(), initialValue);
  cudaMemcpy(
      data1,
      vec1.data(),
      vec1.size() * sizeof(uint64_t),
      cudaMemcpyHostToDevice);

  // Build cudf::table
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(std::move(counterCol));
  auto table = std::make_unique<cudf::table>(std::move(columns));

  cudf::packed_columns packed = cudf::pack(table->view());

  // sync the stream before giving the packed columns to UCX since UCX
  // is not stream aware.
  stream.synchronize();

  return std::unique_ptr<cudf::packed_columns>(new cudf::packed_columns(
      std::move(packed.metadata), std::move(packed.gpu_data)));
}
