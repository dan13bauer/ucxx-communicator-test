#!/bin/bash

docker build -f Dockerfile.optimized -t communicator-run-dnb-img:latest .
# docker build -f Dockerfile -t communicator-dev-dnb-img:latest .