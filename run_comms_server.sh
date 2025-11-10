#!/bin/bash

# Default parameter values
LISTENER_PORT=4567
NUM_CHUNKS=100
ROWS=16777216
UCXX_ERROR_HANDLING=true
UCXX_BLOCKING_POLLING=true
GPU=7

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --listener_port)
      LISTENER_PORT="$2"
      shift 2
      ;;
    --num_chunks)
      NUM_CHUNKS="$2"
      shift 2
      ;;
    --rows)
      ROWS="$2"
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
    --gpu)
      GPU="$2"
      shift 2
      ;;
    -h|--help)
      echo "Usage: $0 [OPTIONS]"
      echo ""
      echo "Options:"
      echo "  --listener_port <port>         Listener port (default: 4567)"
      echo "  --num_chunks <number>          Number of chunks (default: 10)"
      echo "  --rows <number>                Number of rows (default: 10000000)"
      echo "  --ucxx_error_handling <bool>   UCXX error handling (default: true)"
      echo "  --ucxx_blocking_polling <bool> UCXX blocking polling (default: true)"
      echo "  --gpu <number>                 The GPU (0-7) on which to run (default: 7)"
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

IMG=communicator-dev-dnb-img:latest

echo "Starting communicator server with:"
echo "  Listener Port: $LISTENER_PORT"
echo "  Num Chunks: $NUM_CHUNKS"
echo "  Rows: $ROWS"
echo "  UCXX Error Handling: $UCXX_ERROR_HANDLING"
echo "  UCXX Blocking Polling: $UCXX_BLOCKING_POLLING"
echo ""

docker run --rm -it \
    --gpus all \
    --network=host \
    --device /dev/infiniband/rdma_cm \
    --device=/dev/infiniband/uverbs0 \
    --device=/dev/infiniband/uverbs1 \
    --device=/dev/infiniband/uverbs2 \
    --device=/dev/infiniband/uverbs3 \
    --device=/dev/infiniband/uverbs4 \
    --device=/dev/infiniband/uverbs5 \
    --device=/dev/infiniband/uverbs6 \
    --device=/dev/infiniband/uverbs7 \
    --device=/dev/infiniband/uverbs8 \
    --device=/dev/infiniband/uverbs9 \
    --cap-add=IPC_LOCK \
    --shm-size=1g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -e "CUDA_VISIBLE_DEVICES=${GPU}" \
    ${IMG} \
    /workspace/ucxx-communicator-test/build/cpp/communicator \
    --listener_port=${LISTENER_PORT} \
    --num_chunks=${NUM_CHUNKS} \
    --rows=${ROWS} \
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
