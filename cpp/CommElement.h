#pragma once

#include <stdexcept>

// The CommElement is the abstract base class of both the
// per-client context on the exchange server side as well as the
// exchange source side.

class Communicator;
class EndpointRef;

class CommElement {
 public:
  CommElement(
      const std::shared_ptr<Communicator> communicator,
      std::shared_ptr<EndpointRef> endpointRef)
      : communicator_{communicator}, endpointRef_{endpointRef} {}

  CommElement(const std::shared_ptr<Communicator> communicator)
      : communicator_{communicator}, endpointRef_{nullptr} {}

  virtual ~CommElement() = default;

  /// @brief Advance the communication by executing the communication elements
  /// specific communication pattern.
  virtual void process() = 0;

  // Called when the underlying endpoint was closed
  // or the communicator is finished.
  virtual void close() = 0;

  /// @brief Returns a unique signature of the communication element.
  virtual std::string toString() const = 0;


 protected:
  
  const std::shared_ptr<Communicator> communicator_;
  std::shared_ptr<EndpointRef> endpointRef_;
  
};
