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
    uint64_t initialValue,
    rmm::cuda_stream_view stream)
    : CommElement(communicator, endpointRef),
      key_(key),
      keyHash_(fnv1a_32(key)),
      numExchanges_(FLAGS_num_chunks),
      initialValue_(initialValue),
      stream_(stream) {
          setState(ServerState::Created);
      }

// static
std::shared_ptr<Sender> Sender::create(
    const std::shared_ptr<Communicator> communicator,
    std::shared_ptr<EndpointRef> endpointRef,
    const std::string& key,
    uint64_t initialValue,
    rmm::cuda_stream_view stream) {
  auto ptr = std::shared_ptr<Sender>(
      new Sender(communicator, endpointRef, key, initialValue, stream));
  return ptr;
}

void Sender::process() {
  switch (state_) {
    case ServerState::Created:
      // Create the cudf::table once with initialized data
      createTable(FLAGS_rows, initialValue_);
      setState(ServerState::ReadyToTransfer);
      communicator_->addToWorkQueue(getSelfPtr());
      break;
    case ServerState::ReadyToTransfer:
      setState(ServerState::WaitingForDataFromQueue);

      // Pack the table into packed_columns for each send, or signal end
      if (sequenceNumber_ >= numExchanges_) {
        dataPtr_ = nullptr; // signal that we are at the end.
      } else {
        // Pack the existing table (metadata is regenerated each time)
        dataPtr_ = packTable();
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
  std::cout << "Close Sender to remote " << key_ << std::endl;
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
    // Copy the metadata vector into a new unique_ptr (can't assign unique_ptr)
    metadataMsg->cudfMetadata = std::make_unique<std::vector<uint8_t>>(*dataPtr_->metadata);
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

  std::cout << "Sending metadata of size: " << metadataMsg->dataSizeBytes << std::endl;

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

void Sender::createTable(
    std::size_t numRows,
    uint64_t initialValue,
    rmm::device_async_resource_ref mr) {
  // Create a numeric column using cudf::make_numeric_column
  auto counterCol = cudf::make_numeric_column(
      cudf::data_type{cudf::type_id::INT64},
      numRows,
      cudf::mask_state::UNALLOCATED, // no nulls.
      stream_,
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

  // Build cudf::table and store it for reuse
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(std::move(counterCol));
  table_ = std::make_unique<cudf::table>(std::move(columns));

  // sync the stream after table creation
  stream_.synchronize();
}

std::unique_ptr<cudf::packed_columns> Sender::packTable() {
  // Pack the existing table (can be called multiple times)
  cudf::packed_columns packed = cudf::pack(table_->view());

  // sync the stream before giving the packed columns to UCX since UCX
  // is not stream aware.
  stream_.synchronize();

  return std::unique_ptr<cudf::packed_columns>(new cudf::packed_columns(
      std::move(packed.metadata), std::move(packed.gpu_data)));
}

std::unique_ptr<cudf::packed_columns> Sender::makePackedColumns(
    std::size_t numRows,
    uint64_t initialValue,
    rmm::device_async_resource_ref mr) {
  // Legacy method - now implemented using createTable + packTable
  createTable(numRows, initialValue, mr);
  return packTable();
}
