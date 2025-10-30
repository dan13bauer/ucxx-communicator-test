#pragma once

#include <ucxx/api.h>
#include <map>
#include "Sender.h"

/// @brief The acceptor creates a new sender each time a handshake message is
/// received. Handshakes are sent as active messages, using a worker-wide
/// handler. The acceptor is a passive component that is used by the
/// Communicator.
struct Acceptor {
  // The static callback function for incoming handshake requests.
  static void cStyleAMCallback(
      std::shared_ptr<ucxx::Request> request,
      ucp_ep_h ep);

  /// @brief Adds the endpoint reference to the handleToEndpointRef_ map such
  /// that endpoint handls can be resoved
  void registerEndpointRef(std::shared_ptr<EndpointRef> endpointRef);

  // Maps the lower-layer UCP endpoint handle to an endpoint reference.
  std::map<ucp_ep_h, std::shared_ptr<EndpointRef>> handleToEndpointRef_;
};
