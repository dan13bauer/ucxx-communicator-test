#!/bin/bash

IMG=communicator-dev-dnb-img:latest

docker run -d --rm -it --gpus all  --network=host \
       --device /dev/infiniband/rdma_cm  \
       --device=/dev/infiniband/uverbs0 --device=/dev/infiniband/uverbs1 \
       --device=/dev/infiniband/uverbs2 --device=/dev/infiniband/uverbs3 \
       --device=/dev/infiniband/uverbs4 --device=/dev/infiniband/uverbs5 \
       --device=/dev/infiniband/uverbs6 --device=/dev/infiniband/uverbs7 \
       --device=/dev/infiniband/uverbs8 --device=/dev/infiniband/uverbs9 \
       --cap-add=IPC_LOCK \
       --shm-size=1g \
       --ulimit memlock=-1 \
       --ulimit stack=67108864 \
       --pid host \
       --entrypoint='' \
       ${IMG} \
       tail -f /dev/null
