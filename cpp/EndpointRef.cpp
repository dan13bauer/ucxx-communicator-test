#include "EndpointRef.h"
#include <iostream>
#include "Communicator.h"

/* static */
void EndpointRef::onClose(ucs_status_t status, std::shared_ptr<void> arg) {
  std::shared_ptr<EndpointRef> ep = std::static_pointer_cast<EndpointRef>(arg);
  ep->cleanup();
  while (!ep->communicators_.empty()) {
    auto& ptr = *ep->communicators_.begin();
    if (std::shared_ptr<CommElement> spt = ptr.lock()) {
      // communicator reference is valid so we need to close it.
      spt->close();
    }
    ep->communicators_.erase(ptr);
  }

  auto c = Communicator::getInstance();
  c->removeEndpointRef(ep);

  std::cout << "- EndpointRef::onClose" << std::endl;
}

bool EndpointRef::addCommElem(std::shared_ptr<CommElement> commElem) {
  if (!commElem) {
    return false; // nothing to do, no commElem.
  }
  cleanup();
  auto ret = communicators_.insert(commElem);
  return ret.second;
}

void EndpointRef::removeCommElem(std::shared_ptr<CommElement> commElem) {
  if (!commElem) {
    return;
  }  
  communicators_.erase(commElem);
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

void EndpointRef::cleanup() {
  for(auto it = communicators_.begin(); it != communicators_.end();) {
    if (it->expired()) {
      it = communicators_.erase(it);
    } else {
      ++it;
    }
  }
}
