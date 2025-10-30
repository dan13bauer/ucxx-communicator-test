#include "Acceptor.h"
#include "protocol.h"
#include "Communicator.h"
#include "EndpointRef.h"

/*static*/
void Acceptor::cStyleAMCallback(
    std::shared_ptr<ucxx::Request> request,
    ucp_ep_h ep) {
  CHECK(request, "AMCallback called with nullptr request!");
  CHECK(request->isCompleted(), "AMCallback called with incomplete request!");
  auto buffer =
      std::dynamic_pointer_cast<ucxx::Buffer>(request->getRecvBuffer());
  CHECK(
      buffer->getSize() == sizeof(HandshakeMsg),
      "AMCallback: unexpected size of handshake.");
  HandshakeMsg* handshakePtr = reinterpret_cast<HandshakeMsg*>(buffer->data());

  // Create a sender based on the information received in the initial handshake.
  std::shared_ptr<Communicator> communicator = Communicator::getInstance();

  auto it = communicator->acceptor_.handleToEndpointRef_.find(ep);
  CHECK(
      it != communicator->acceptor_.handleToEndpointRef_.end(),
      "Could not find endpoint reference");
  std::shared_ptr<EndpointRef> epRef = it->second;

  auto sender = Sender::create(
      communicator,
      epRef,
      std::string(handshakePtr->key),
      handshakePtr->initialValue);

  // Add this sender to the endpoint reference.
  epRef->addCommunicator(sender);

  // Register sender with communicator.
  communicator->registerComms(sender);

  std::cout << "- Acceptor::cStyleAMCallback " << buffer->getSize()
            << std::endl;
}

// Add endpoint reference to ucp_cp -> epRef map.
void Acceptor::registerEndpointRef(std::shared_ptr<EndpointRef> endpointRef) {
  auto epHandle = endpointRef->endpoint_->getHandle();
  auto res = handleToEndpointRef_.insert(std::pair{epHandle, endpointRef});
  CHECK(res.second, "Endpoint handle already exists!");
}
