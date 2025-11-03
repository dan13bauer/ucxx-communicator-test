#pragma once

#include <ucxx/api.h>
#include <memory>
#include <set>

#include "CommElement.h"

/// @brief The endpoint reference keeps track of the communication elements that
/// use a given UCXX endpoint. When this endpoint is closed, the elements
/// (CudfExchangeSources and CudfExchangeServers) are notified. When an element
/// is done, it notifies the endpoint. This class is not thread safe.
class EndpointRef : std::enable_shared_from_this<EndpointRef> {
 public:
  EndpointRef(const std::shared_ptr<ucxx::Endpoint> endpoint)
      : endpoint_{endpoint}, communicators_{} {}

  /// @brief Static method that is called when the underlying UCXX system closes
  /// the endpoint. In this case, all communication elements are informed that
  /// the endpoint has been closed.
  /// @param status The status (reason) why the endpoint has been closed.
  /// @param arg A reference to the EndpointRef (since this is a static method)
  static void onClose(ucs_status_t status, std::shared_ptr<void> arg);

  /// @brief Adds a new CommElement that is using this endpoint.
  /// @param commElem A shared pointer to the CudfExchangeSource or
  /// CudfExchangeServer.
  /// @return True, if commElem could be added.
  bool addCommElem(std::shared_ptr<CommElement> commElem);

  /// @brief Removes a CommElement from this endpoint again.
  /// @param commElem A shared pointer to the CudfExchangeSource or
  /// CudfExchangeServer.
  void removeCommElem(std::shared_ptr<CommElement> commElem);

  /// implement < operator such that this endpoint can be used in a
  /// std::map
  bool operator<(EndpointRef const& other);

  const std::shared_ptr<ucxx::Endpoint> endpoint_;

 private:
  void cleanup(); // cleans up expired communication elements.

  std::set<
      std::weak_ptr<CommElement>,
      std::owner_less<std::weak_ptr<CommElement>>>
      communicators_;
};
