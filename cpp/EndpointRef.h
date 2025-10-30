#pragma once

#include <ucxx/api.h>
#include <memory>
#include <set>

#include "CommElement.h"

/// @brief The endpoint reference keeps track of the communicators that use
/// a given UCXX endpoint. When this endpoint is closed, the communicators are
/// notified. When a communicator is done, it notifies the endpoint.
/// This class is not thread safe.
class EndpointRef : std::enable_shared_from_this<EndpointRef> {
 public:
  EndpointRef(const std::shared_ptr<ucxx::Endpoint> endpoint)
      : endpoint_{endpoint}, communicators_{} {}

  /// @brief Static method that is called when the underlying UCXX system closes
  /// the endpoint. In this case, all communicators are informed that the
  /// endpoint has been closed.
  /// @param status The status (reason) why the endpoint has been closed.
  /// @param arg A reference to the EndpointRef (since this is a static method)
  static void onClose(ucs_status_t status, std::shared_ptr<void> arg);

  /// @brief Adds a new communicator that is using this endpoint.
  /// @param communicator A shared pointer to the communicator.
  /// @return True, if communicator could be added.
  bool addCommunicator(std::shared_ptr<CommElement> communicator);

  /// @brief Removes a communicator from this endpiont again.
  /// @param communicator The shared pointer to the communicator.
  void removeCommunicator(std::shared_ptr<CommElement> communicator);

  /// implement < operator such that this endpoint can be used in a
  /// std::map
  bool operator<(EndpointRef const& other);

  const std::shared_ptr<ucxx::Endpoint> endpoint_;

 private:
  // For the references to the communicators, a weak pointer is used.
  std::set<
      std::weak_ptr<CommElement>,
      std::owner_less<std::weak_ptr<CommElement>>>
      communicators_;
};
