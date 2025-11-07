#include "Communicator.h"
#include <cuda_runtime.h>
#include <ucxx/api.h>
#include <ucxx/utils/ucx.h>
#include <iostream>
#include "protocol.h"
#include "CommElement.h"
#include "EndpointRef.h"
#include <gflags/gflags.h>

DECLARE_bool(ucxx_error_handling);
DECLARE_bool(ucxx_blocking_polling);


// static
std::once_flag Communicator::onceFlag;
std::shared_ptr<Communicator> Communicator::instancePtr_ = nullptr;

/* static */
std::shared_ptr<Communicator> Communicator::initAndGet(uint16_t port) {
  std::call_once(onceFlag, [&] {
    instancePtr_ = std::shared_ptr<Communicator>(new Communicator());
    instancePtr_->port_ = port;
  });
  CHECK(
      instancePtr_->port_ == port,
      "Cannot initialize communicator again with different port");
  return instancePtr_;
}

/* static */
std::shared_ptr<Communicator> Communicator::getInstance() {
  if (!instancePtr_) {
    throw std::logic_error(
        "Communicator not initialized. Call init(port) first.");
  }
  return instancePtr_;
}

/* static */ void Communicator::cStyleListenerCallback(
    ucp_conn_request_h conn_request,
    void* arg) {
  // cast the argument back to our instance variable:
  Communicator* instance = static_cast<Communicator*>(arg);
  instance->listenerCallback(conn_request);
}

Communicator::~Communicator() {
  listener_.reset();
  if (FLAGS_ucxx_blocking_polling) {
    auto req = worker_->flush();
    worker_->progressWorkerEvent(100);
  } else {
    worker_->progress();
  }
  worker_.reset();
  context_.reset();
  std::cout << "Communicator destructed" << std::endl;
}

/// @brief Run doesn't return until stop() is called.
/// All operations of the communicator will be carried out in the thread
/// that calls run.
void Communicator::run() {
  std::cout << "Using error handling mode: " << FLAGS_ucxx_error_handling << std::endl;
  std::cout << "Using blocking progress mode: " << FLAGS_ucxx_blocking_polling << std::endl;

  running_.store(true);
  // Force CUDA context creation
  cudaFree(0);

  // create the UCXX context, worker, listener-context etc.
  if (FLAGS_ucxx_blocking_polling) {
    context_ = ucxx::createContext({}, ucxx::Context::defaultFeatureFlags);
  } else {
    context_ = ucxx::createContext({}, UCP_FEATURE_TAG | UCP_FEATURE_AM);
  }

  worker_ = context_->createWorker();

  if (FLAGS_ucxx_blocking_polling) {
    // Communicator is using blocking progress mode.
    worker_->initBlockingProgressMode();
  }

  listener_ = worker_->createListener(
      port_, Communicator::cStyleListenerCallback, this);

  // Setup the active message callback that handles the
  // initial handshake and creates the senders.
  ucxx::AmReceiverCallbackInfo info(kAmCallbackOwner, kAmCallbackId);
  worker_->registerAmReceiverCallback(info, &Acceptor::cStyleAMCallback);

  std::cout << "Communicator running." << std::endl;
  while (running_) {
    try {
      // wait for progress.
      if (FLAGS_ucxx_blocking_polling) {
        worker_->progressWorkerEvent(0);
      } else {
        worker_->progress();
      }

      // process the work queue. Make sure that communication is progressed
      // after each call to a comms element, otherwise we will deadlock.
      while (!workQueue_.empty()) {
        auto comms = workQueue_.pop();
        comms->process();
        if (FLAGS_ucxx_blocking_polling) {
          worker_->progressWorkerEvent(0);
        } else {
          worker_->progress();
        }
      }
    } catch (ucxx::IOError& e) {
      std::cerr << "In Communicator main loop UCXX Exception: " << e.what()
                << std::endl;
      throw e;
    }
  }
  std::cout << "Communicator stopping." << std::endl;
}

/// @brief Stops the communicator, called from an outside thread.
void Communicator::stop() {
  running_.store(false);
}

void Communicator::registerCommElement(std::shared_ptr<CommElement> comms) {
  std::lock_guard<std::mutex> lock(elemMutex_);
  auto ret = elements_.insert(comms);
  CHECK(ret.second, "CommElement already registered!");
  // Also put the comms element into the work queue.
  workQueue_.push(comms);
}

void Communicator::addToWorkQueue(std::shared_ptr<CommElement> comms) {
  if (!comms) {
    return;
  }
  workQueue_.push(comms);
}

void Communicator::unregister(std::shared_ptr<CommElement> comms) {
  std::lock_guard<std::mutex> lock(elemMutex_);
  if (!comms) {
    return;
  }
  workQueue_.erase(comms);
  elements_.erase(comms);
}

std::shared_ptr<EndpointRef> Communicator::assocEndpointRef(
    std::shared_ptr<CommElement> comms,
    HostPort hostPort) {
  auto it = endpoints_.find(hostPort);
  if (it != endpoints_.end()) {
    std::shared_ptr<EndpointRef> ep = it->second;
    ep->addCommElem(comms);
    return ep;
  }
  // endpoint doesn't exist. Need to connect. Enable error handling.
  auto ep = worker_->createEndpointFromHostname(
      hostPort.hostname, hostPort.port, FLAGS_ucxx_error_handling);
  std::shared_ptr<EndpointRef> epRef = nullptr;
  if (ep != nullptr) {
    epRef = std::make_shared<EndpointRef>(ep);
    epRef->addCommElem(comms);
    if (FLAGS_ucxx_error_handling) {
      // register on close callback.
      ep->setCloseCallback(EndpointRef::onClose, epRef);
    }
    endpoints_.insert(std::pair{hostPort, epRef});
  }
  return epRef;
}

void Communicator::removeEndpointRef(std::shared_ptr<EndpointRef> ep) {
  std::cout << "In Communicator::removeEndpointRef for Communicator with port = "
          << Communicator::getInstance()->port_ << std::endl;
  std::string worker_info = ep->endpoint_->getWorker()->getInfo();
  if (ep->endpoint_ && ep->endpoint_->isAlive()) {
    std::cout << "In Communicator::removeEndpointRef call closeBlocking" << std::endl;
    ep->endpoint_->closeBlocking();
  }
  for (auto it = endpoints_.begin(); it != endpoints_.end();) {
    if (it->second == ep) {
      it = endpoints_.erase(it); // erase returns the next iterator
    } else {
      ++it;
    }
  }
  std::cout << "- Communicator::removeEndpointRef" << std::endl;;
}

/// @brief The callback method that is invoked when a client connects.
void Communicator::listenerCallback(ucp_conn_request_h conn_request) {
  char ip_str[INET6_ADDRSTRLEN];
  char port_str[INET6_ADDRSTRLEN];
  ucp_conn_request_attr_t attr{};

  attr.field_mask = UCP_CONN_REQUEST_ATTR_FIELD_CLIENT_ADDR;
  ucxx::utils::ucsErrorThrow(ucp_conn_request_query(conn_request, &attr));
  ucxx::utils::sockaddr_get_ip_port_str(
      &attr.client_address, ip_str, port_str, INET6_ADDRSTRLEN);
  std::cout
      << "Communicator received a connection request from client at address "
      << ip_str << ":" << port_str << std::endl;

  // incoming endpoints are not shared. Outgoing endpoints to the same node are
  // shared. This guarantees that between any two nodes, there will be at most 2
  // endpoints, one per direction. For compatibility reasons, both incoming and
  // outgoing endpoints are represented using the EndpointRef.
  auto endpoint = listener_->createEndpointFromConnRequest(conn_request, FLAGS_ucxx_error_handling);
  auto epRef = std::make_shared<EndpointRef>(endpoint);
  if (FLAGS_ucxx_error_handling) {
    endpoint->setCloseCallback(EndpointRef::onClose, epRef);
  }
  // Add this endpoint reference to the list of endpoints.
  unsigned long val = std::strtoul(port_str, nullptr, 10);
  CHECK(
      val <= static_cast<unsigned long>(std::numeric_limits<uint16_t>::max()),
      "Port out of range for uint16_t!");

  uint16_t port = static_cast<uint16_t>(val);
  HostPort hp(ip_str, port);
  auto res = endpoints_.insert(std::pair{hp, epRef});
  std::cout << "In Communicator::listenerCallback adding endpoint to a set of size " << endpoints_.size() << std::endl;
  CHECK(res.second, "Endpoint already exists! " + hp.toString() );
  acceptor_.registerEndpointRef(epRef);
}
