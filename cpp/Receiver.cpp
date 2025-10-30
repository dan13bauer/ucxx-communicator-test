#include "Receiver.h"
#include <fmt/format.h>
#include <gflags/gflags.h> // FIXME: for testing.
#include <sstream>
#include "protocol.h"
#include "Communicator.h"


// This constructor is private.
Receiver::Receiver(
    const std::shared_ptr<Communicator> communicator,
    const std::string& host,
    uint16_t port,
    const uint32_t receiverId)
    : CommElement(communicator),
      host_(host),
      port_(port),
      state_(ReceiverState::Created),
      key_("receiver_" + std::to_string(receiverId)),
      initialValue_(receiverId),
      taskIdHash_(fnv1a_32(key_)) {}

// static
std::shared_ptr<Receiver> Receiver::create(
    const std::shared_ptr<Communicator> communicator,
    const std::string& host,
    uint16_t port,
    const uint32_t receiverId) {
  auto ptr = std::shared_ptr<Receiver>(
      new Receiver(communicator, host, port, receiverId));
  return ptr;
}

void Receiver::process() {
  std::cout << "+ Receiver::process state " << receiverStateNames_[state_] << std::endl;
  switch (state_) {
    case ReceiverState::Created: {
      // Get the endpoint.
      HostPort hp{host_, port_};
      std::shared_ptr<Receiver> selfPtr = getSelfPtr();
      auto epRef = communicator_->assocEndpointRef(selfPtr, hp);
      if (epRef) {
        setEndpoint(epRef);
        sendHandshake();
      } else {
        // connection failed.
        std::cerr << "Failed to connect to " << host_ << ":"
                  << std::to_string(port_) << std::endl;
        close();
      }
      communicator_->addToWorkQueue(getSelfPtr());
    } break;
    case ReceiverState::WaitingForHandshakeComplete:
      // Waiting for metadata is handled by an upcall from UCXX. Nothing to do
      break;
    case ReceiverState::ReadyToReceive:
      // change state before calling getMetadata since immediate upcalls in
      // getMetadata will also change state.
      state_ = ReceiverState::WaitingForMetadata;
      getMetadata();
      break;
    case ReceiverState::WaitingForMetadata:
      // Waiting for metadata is handled by an upcall from UCXX. Nothing to do
      break;
    case ReceiverState::WaitingForData:
      // Waiting for data is handled by an upcall from UCXX. Nothing to do.
      break;
    case ReceiverState::Done:
      // unregister.
      close();
      //communicator_->unregister(getSelfPtr());
      exit(0); // Just For Testing
      break;
  }
   //std::cout << "- Receiver::process state " << state_ << std::endl;
}

void Receiver::close(bool endpointClosing) {
  std::cout << "+ Receiver::close" << std::endl;
  if (state_ != ReceiverState::Done) {
    std::cerr << "Close receiver to remote " << key_ << " in error!"
              << std::endl;
  } else {
    std::cerr << "Close receiver to remote " << key_ << "." << std::endl;
  }

  if (endpointRef_ && !endpointClosing) {
    endpointRef_->removeCommunicator(getSelfPtr());
  }
  communicator_->unregister(getSelfPtr());
  state_ = ReceiverState::Done;
  std::cout << "- Receiver::close" << std::endl;
}

std::string Receiver::toString() {
  std::stringstream out;
  out << "[Receiver " << key_ << "]";
  return out.str();
}

void Receiver::setEndpoint(std::shared_ptr<EndpointRef> endpointRef) {
  endpointRef_ = std::move(endpointRef);
}

std::shared_ptr<Receiver> Receiver::getSelfPtr() {
  return shared_from_this();
}

void Receiver::sendHandshake() {
  std::shared_ptr<HandshakeMsg> handshakeReq = std::make_shared<HandshakeMsg>();
  handshakeReq->initialValue = initialValue_;
  strcpy(handshakeReq->key, key_.c_str());

  std::cout << toString() << " Sending handshake with initial value: "
            << handshakeReq->initialValue << " to server" << std::endl;

  // Create the handshake which will register client's existence with the server
  state_ = ReceiverState::WaitingForHandshakeComplete;
  ucxx::AmReceiverCallbackInfo info(
      communicator_->kAmCallbackOwner, communicator_->kAmCallbackId);
  request_ = endpointRef_->endpoint_->amSend(
      handshakeReq.get(),
      sizeof(HandshakeMsg),
      UCS_MEMORY_TYPE_HOST,
      info,
      false,
      std::bind(
          &Receiver::onHandshake,
          this,
          std::placeholders::_1,
          std::placeholders::_2),
      handshakeReq);

  request_->checkError();
  auto s = request_->getStatus();
  if (s != UCS_INPROGRESS && s != UCS_OK) {
    std::cerr << "Error in sendHandshake " << ucs_status_string(s)
              << " failed for task: " << key_ << std::endl;
    close();
  }
}

void Receiver::onHandshake(ucs_status_t status, std::shared_ptr<void> arg) {
  if (status != UCS_OK) {
    std::string errorMsg = fmt::format(
        "Failed to send handshake ot host {}:{}, task {}: {}",
        host_,
        port_,
        key_,
        ucs_status_string(status));
    std::cerr << errorMsg << std::endl;
    close();
  } else {
    std::cout << toString() << "+ onHandshake " << ucs_status_string(status)
              << std::endl;
    state_ = ReceiverState::ReadyToReceive;
    // more work to do
    communicator_->addToWorkQueue(getSelfPtr());
  }
}

void Receiver::getMetadata() {
  uint32_t sizeMetadata = 4096; // shouldn't be a fixed size.
  auto metadataReq = std::make_shared<std::vector<uint8_t>>(sizeMetadata);
  uint64_t metadataTag = getMetadataTag(taskIdHash_, sequenceNumber_);

  std::cout << toString()
            << " waiting for metadata for chunk: " << sequenceNumber_
            << " using tag: " << std::hex << metadataTag << std::dec
            << std::endl;

  request_ = endpointRef_->endpoint_->tagRecv(
      reinterpret_cast<void*>(metadataReq->data()),
      sizeMetadata,
      ucxx::Tag{metadataTag},
      ucxx::TagMaskFull,
      false,
      std::bind(
          &Receiver::onMetadata,
          this,
          std::placeholders::_1,
          std::placeholders::_2),
      metadataReq);

  request_->checkError();
  auto s = request_->getStatus();
  if (s != UCS_INPROGRESS && s != UCS_OK) {
    std::cout << "Error in getMetadata, receive metadata "
              << ucs_status_string(s) << " failed for task: " << key_
              << std::endl;
  }
}

void Receiver::onMetadata(ucs_status_t status, std::shared_ptr<void> arg) {
  if (status != UCS_OK) {
    std::string errorMsg = fmt::format(
        "Failed to receive metadata from host {}:{}, task {}: {}",
        host_,
        port_,
        key_,
        ucs_status_string(status));
    std::cerr << errorMsg << std::endl;
    close();
  } else {
    CHECK(arg != nullptr, "Didn't get metadata");

    // arg contains the actual serialized metadata, deserialize the metadata
    std::shared_ptr<std::vector<uint8_t>> metadataMsg =
        std::static_pointer_cast<std::vector<uint8_t>>(arg);

    auto ptr = std::make_shared<DataAndMetadata>();

    ptr->metadata =
        std::move(MetadataMsg::deserializeMetadataMsg(metadataMsg->data()));

    if (ptr->metadata.atEnd) {
      // It seems that all data has been transferred
      std::cout << toString() << "There is no more data to transfer!"
                << std::endl;
      // Let the statement decide the final action
      state_ = ReceiverState::Done;
      communicator_->addToWorkQueue(getSelfPtr());
      //close();
      return;
    }

    // Now allocate memory for the CudaVector
    // Get a stream from the global stream pool

    auto stream = cudf::get_default_stream();

    try {
      ptr->dataBuf = std::make_unique<rmm::device_buffer>(
          ptr->metadata.dataSizeBytes, stream);
    } catch (const rmm::bad_alloc& e) {
      std::cerr << "!!! RMM bad_alloc: " << e.what() << "\n";
      state_ = ReceiverState::Done;
      communicator_->addToWorkQueue(getSelfPtr());
      return;
    } catch (const rmm::cuda_error& e) {
      std::cerr << "!!! General allocation exception: " << e.what() << "\n";
    }

    // Initiate the transfer of the actual data from GPU-2-GPU
    uint64_t dataTag = getDataTag(taskIdHash_, sequenceNumber_);
    std::cout << toString()
              << " waiting for data for chunk: " << sequenceNumber_
              << " using tag: " << std::hex << dataTag << std::dec << std::endl;

    state_ = ReceiverState::WaitingForData;
    request_ = endpointRef_->endpoint_->tagRecv(
        ptr->dataBuf->data(),
        ptr->metadata.dataSizeBytes,
        ucxx::Tag{dataTag},
        ucxx::TagMaskFull,
        false,
        std::bind(
            &Receiver::onData,
            this,
            std::placeholders::_1,
            std::placeholders::_2),
        ptr // DataAndMetadata
    );

    request_->checkError();
    auto s = request_->getStatus();
    if (s != UCS_INPROGRESS && s != UCS_OK) {
      std::cout << "Error in onMetadata, receive data " << ucs_status_string(s)
                << " failed for task: " << key_ << std::endl;
      close();
    }
  }
}

void Receiver::onData(ucs_status_t status, std::shared_ptr<void> arg) {
  if (status != UCS_OK) {
    std::string errorMsg = fmt::format(
        "Failed to receive data from host {}:{}, task {}: {}",
        host_,
        port_,
        key_,
        ucs_status_string(status));
    std::cout << toString() << errorMsg << std::endl;
    close();
  } else {
    std::cout << toString() << "+ onData " << ucs_status_string(status)
              << "got chunk: " << sequenceNumber_ << std::endl;

    this->sequenceNumber_++;
    std::shared_ptr<DataAndMetadata> ptr =
        std::static_pointer_cast<DataAndMetadata>(arg);

    std::unique_ptr<cudf::packed_columns> columns =
        std::make_unique<cudf::packed_columns>(
            std::move(ptr->metadata.cudfMetadata), std::move(ptr->dataBuf));
    dumpValues(std::move(columns), ptr->metadata);
    state_ = ReceiverState::ReadyToReceive;
    communicator_->addToWorkQueue(getSelfPtr());
  }
}

void Receiver::dumpValues(
    std::unique_ptr<cudf::packed_columns> columns,
    MetadataMsg& metadata) {
  if (columns == nullptr) {
    std::cout << toString() << " reached the end." << std::endl;
    return;
  }

  // Convert the cudf::packed_columns into a cudf::tableView
  cudf::table_view tblView = cudf::unpack(*columns);
  uint32_t numRows = tblView.num_rows() > 10 ? 10 : tblView.num_rows();
  std::vector<uint64_t> hostVec(numRows);
  cudaMemcpy(
      &hostVec[0],
      tblView.column(0).head<uint64_t>(),
      numRows * sizeof(uint64_t),
      cudaMemcpyDeviceToHost);
  std::cout << toString() << "data: ";
  for (auto& val : hostVec) {
    std::cout << val << " ";
 }
  std::cout << std::endl;
}
