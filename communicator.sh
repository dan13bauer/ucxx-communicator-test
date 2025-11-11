#!/bin/bash

IMG=communicator-dev-dnb-img:latest
NAME=communicator-dev-dnb

docker run -it --rm --gpus all  --network=host --device /dev/infiniband/rdma_cm  \
       --cap-add=IPC_LOCK \
       --shm-size=1g \
       --ulimit memlock=-1 \
       --ulimit stack=67108864 \
       --pid host \
       --name ${NAME} \
       -e UCX_RNDV_PIPELINE_ERROR_HANDLING=y \
       -e UCX_PROTO_INFO=y \
       -e UCX_LOG_LEVEL=debug \
       ${IMG} \
       $@
