#!/bin/bash

# Default parameter values
# No listener by default, set port to 0.
LISTENER_PORT=0
PORTS="4567"
HOSTNAMES="127.0.0.1"
UCXX_ERROR_HANDLING=false
UCXX_BLOCKING_POLLING=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --listener_port)
      LISTENER_PORT="$2"
      shift 2
      ;;
    --ports)
      PORTS="$2"
      shift 2
      ;;
    --hostnames)
      HOSTNAMES="$2"
      shift 2
      ;;
    --ucxx_error_handling)
      UCXX_ERROR_HANDLING="$2"
      shift 2
      ;;
    --ucxx_blocking_polling)
      UCXX_BLOCKING_POLLING="$2"
      shift 2
      ;;
    -h|--help)
      echo "Usage: $0 [OPTIONS]"
      echo ""
      echo "Options:"
      echo "  --listener_port <port>         Listener port (default: ${LISTENER_PORT})"
      echo "  --ports <port_list>            Comma-separated ports to connect to (default: ${PORTS})"
      echo "  --hostnames <hostnames>        Hostnames to connect to (default: ${HOSTNAMES})"
      echo "  --ucxx_error_handling <bool>   UCXX error handling (default: ${UCXX_ERROR_HANDLING})"
      echo "  --ucxx_blocking_polling <bool> UCXX blocking polling (default: ${UCXX_BLOCKING_POLLING})"
      echo "  -h, --help                     Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use -h or --help for usage information"
      exit 1
      ;;
  esac
done

IMG=communicator-run-dnb-img:latest

echo "Starting communicator client with:"
echo "  Listener Port: $LISTENER_PORT"
echo "  Connect to Ports: $PORTS"
echo "  Hostnames: $HOSTNAMES"
echo "  UCXX Error Handling: $UCXX_ERROR_HANDLING"
echo "  UCXX Blocking Polling: $UCXX_BLOCKING_POLLING"
echo ""

docker run --rm -it \
    --gpus all \
    --network=host \
    --device /dev/infiniband/rdma_cm \
    --device=/dev/infiniband/uverbs0 \
    --cap-add=IPC_LOCK \
    --shm-size=1g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    ${IMG} \
    --listener_port=${LISTENER_PORT} \
    --ports=${PORTS} \
    --hostnames=${HOSTNAMES} \
    --ucxx_error_handling=${UCXX_ERROR_HANDLING} \
    --ucxx_blocking_polling=${UCXX_BLOCKING_POLLING}

# Meaning of some of the parameters:
    # Allow memory locking for RDMA
    # --cap-add=IPC_LOCK \
    # Shared memory size
    # --shm-size=1g \
    # Unlimited locked memory for RDMA
    # --ulimit memlock=-1 \
    # Stack size limit
    # --ulimit stack=67108864 \
