#include "EndpointRef.h"
#include <iostream>
#include "Communicator.h"

/* static */
void EndpointRef::onClose(ucs_status_t status, std::shared_ptr<void> arg) {
  std::shared_ptr<EndpointRef> ep = std::static_pointer_cast<EndpointRef>(arg);

  std::cout << "+ EndpointRef::onClose number of CommElements "
            << ep->communicators_.size() << std::endl;

  for (auto it = ep->communicators_.begin(); it != ep->communicators_.end();) {
    auto& ptr = *it;
    if (std::shared_ptr<CommElement> spt = ptr.lock()) {
      // communicator reference is valid so we need to close it
      std::cout << "In EndpointRef::onClose " << std::endl;
      spt->close(true);
    }
    it = ep->communicators_.erase(it);
  }

  auto c = Communicator::getInstance();
  c->removeEndpointRef(ep);

  std::cout << "- EndpointRef::onClose" << std::endl;
}

bool EndpointRef::addCommunicator(std::shared_ptr<CommElement> communicator) {
  if (!communicator) {
    return false; // nothing to do, no communicator.
  }
  auto ret = communicators_.insert(communicator);
  return ret.second;
}

void EndpointRef::removeCommunicator(
    std::shared_ptr<CommElement> communicator) {
  if (!communicator) {
    return;
  }
  communicators_.erase(communicator);
  // FIXME: Should the endpoint be closed when the count reaches 0?

  if (communicators_.empty()) {
    auto c = Communicator::getInstance();
    // auto sp = shared_from_this(); //
    // c->removeEndpointRef(sp);  FIXME
  }
}

bool EndpointRef::operator<(EndpointRef const& other) {
  if (endpoint_ == other.endpoint_) {
    return false; // covers the case where both are nullptr
  }
  if (endpoint_ == nullptr) {
    return true; // nullptr comes before anything else.
  }
  if (other.endpoint_ == nullptr) {
    return false;
  }
  return endpoint_->getHandle() < other.endpoint_->getHandle();
}
