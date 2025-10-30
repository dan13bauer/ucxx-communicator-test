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
      state_(SenderState::Created),
      numExchanges_(FLAGS_num_chunks),
      initialValue_(initialValue) {}

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
    case SenderState::Created:
      state_ = SenderState::ReadyToTransfer;
      communicator_->addToWorkQueue(getSelfPtr());
      break;
    case SenderState::ReadyToTransfer:
      // Get the data and store it in the dataPtr_;
      // FIXME: Fetch data from CudfQueueManager instead of generating it here.
      // continue, until we reach the end.
      if (sequenceNumber_ < numExchanges_) {
        dataPtr_ = makePackedColumns(FLAGS_rows, sequenceNumber_);
      } else {
        dataPtr_ = nullptr; // signal that we are at the end.
      }
      state_ = SenderState::DataReady;
      communicator_->addToWorkQueue(getSelfPtr());
      break;
    case SenderState::WaitingForDataFromQueue:
      // Waiting for data is handled by an upcall from the data queue. Nothing
      // to do
      break;
    case SenderState::DataReady:
      sendData();
      break;
    case SenderState::WaitingForSendComplete:
      // Waiting for send complete is handled by an upcall from UCXX. Nothing to
      // do
      break;
    case SenderState::Done:
      // unregister.
      communicator_->unregister(getSelfPtr());
      break;
  };
}

void Sender::close(bool endpointClosing) {
   std::cout << "+ Sender::close" << std::endl;
  if (state_ != SenderState::Done) {
    std::cerr << "Close sender to remote " << key_ << " in error !" << std::endl;
  } else {
    std::cerr << "Close sender to remote " << key_ << "." << std::endl;
  }
  if (endpointRef_ && ! endpointClosing) {
    endpointRef_->removeCommunicator(getSelfPtr());
  }
  std::cout << "Before unregister Sender" << std::endl;
  communicator_->unregister(getSelfPtr());
  state_ = SenderState::Done;
  std::cout << "- Sender::close" << std::endl;
}

std::string Sender::toString() {
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
  uint64_t metadataTag = getMetadataTag(this->keyHash_, this->sequenceNumber_);
  metaRequest_ = endpointRef_->endpoint_->tagSend(
      serializedMetadata.get(),
      serMetaSize,
      ucxx::Tag{metadataTag},
      false,
      [tid = key_, metadataTag](
          ucs_status_t status, std::shared_ptr<void> arg) {
        std::cout << "metadata successfully sent to " << tid
                  << " with tag: " << std::hex << metadataTag << std::endl;
      },
      serializedMetadata);
 
  metaRequest_->checkError();
  auto s = metaRequest_->getStatus();
  if (s != UCS_INPROGRESS && s != UCS_OK) {
    std::cout << "Error in sendData, send metadata " << ucs_status_string(s)
              << " failed for task: " << key_ << std::endl;
  }

  // send the data chunk (if any)
  if (dataPtr_) {
    std::cout << "Sending rmm::buffer: " << std::hex << dataPtr_->gpu_data.get()
              << " pointing to device memory: " << std::hex
              << dataPtr_->gpu_data->data() << std::dec << " to task " << key_
              << ":" << this->sequenceNumber_ << std::endl;

    state_ = SenderState::WaitingForSendComplete;
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


    if (dataRequest_ == nullptr) {
      std::cout << "dataRequest is NULL" << std::endl;

    }
    dataRequest_->checkError();
    s = dataRequest_->getStatus();
    if (s != UCS_INPROGRESS && s != UCS_OK) {
      std::cout << "Error in sendData, send rmm::buffer "
                << ucs_status_string(s) << " failed for task: " << key_
                << std::endl;
      close();
    }
  } else {
    // Data pointer is null, so no more data will be coming.
    std::cout << "Finished transferring partition for task " << key_
              << std::endl;
    state_ = SenderState::Done;
    close();
  }
}

void Sender::sendComplete(ucs_status_t status, std::shared_ptr<void> arg) {

  std::cout << "+ Sender::sendComplete" << std::endl;
  if (status == UCS_OK) {
    CHECK(dataPtr_ != nullptr, "dataPtr_ is null");
    this->sequenceNumber_++;
    dataPtr_.reset(); // release memory.
    state_ = SenderState::ReadyToTransfer;
    communicator_->addToWorkQueue(getSelfPtr());
  } else {
    std::cout << "Error in sendComplete, send complete "
              << ucs_status_string(status) << std::endl;

    if (status == UCS_ERR_CONNECTION_RESET) {
      // Don't close as this will handle elsewhere
    //close(true);
    }
    else {
      close();
    }
  }
  std::cout << "- Sender::sendComplete" << std::endl;
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

  return std::unique_ptr<cudf::packed_columns>(new cudf::packed_columns(
      std::move(packed.metadata), std::move(packed.gpu_data)));
}
