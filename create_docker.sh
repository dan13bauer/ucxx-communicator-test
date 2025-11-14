#!/bin/bash

set -e  # Exit on error

# Function to display usage information
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Build Docker images for UCX/UCXX projects"
    echo ""
    echo "Options:"
    echo "  --all              Build all three images (default)"
    echo "  --run              Build communicator-run-dnb-img:latest from Dockerfile.optimized"
    echo "  --dev              Build communicator-dev-dnb-img:latest from Dockerfile"
    echo "  --perftest         Build ucx-ucxx-perftest-img:latest from Dockerfile.perftest"
    echo "  -h, --help         Display this help message"
    echo ""
    echo "Examples:"
    echo "  $0                 # Build all images"
    echo "  $0 --run           # Build only the optimized runtime image"
    echo "  $0 --dev --perftest # Build dev and perftest images"
    exit 0
}

# Build flags
BUILD_RUN=false
BUILD_DEV=false
BUILD_PERFTEST=false
BUILD_ALL=false

# Parse command line arguments
if [ $# -eq 0 ]; then
    BUILD_ALL=true
else
    while [ $# -gt 0 ]; do
        case "$1" in
            --all)
                BUILD_ALL=true
                shift
                ;;
            --run)
                BUILD_RUN=true
                shift
                ;;
            --dev)
                BUILD_DEV=true
                shift
                ;;
            --perftest)
                BUILD_PERFTEST=true
                shift
                ;;
            -h|--help)
                usage
                ;;
            *)
                echo "Unknown option: $1"
                usage
                ;;
        esac
    done
fi

# If --all is set, enable all builds
if [ "$BUILD_ALL" = true ]; then
    BUILD_RUN=true
    BUILD_DEV=true
    BUILD_PERFTEST=true
fi

# Build the images
echo "=========================================="
echo "Building Docker images..."
echo "=========================================="
echo ""

if [ "$BUILD_RUN" = true ]; then
    echo "Building communicator-run-dnb-img:latest from Dockerfile.optimized..."
    docker build -f Dockerfile.optimized -t communicator-run-dnb-img:latest .
    echo "✓ communicator-run-dnb-img:latest built successfully"
    echo ""
fi

if [ "$BUILD_DEV" = true ]; then
    echo "Building communicator-dev-dnb-img:latest from Dockerfile..."
    docker build -f Dockerfile -t communicator-dev-dnb-img:latest .
    echo "✓ communicator-dev-dnb-img:latest built successfully"
    echo ""
fi

if [ "$BUILD_PERFTEST" = true ]; then
    echo "Building ucx-ucxx-perftest-img:latest from Dockerfile.perftest..."
    docker build -f Dockerfile.perftest -t ucx-ucxx-perftest-img:latest .
    echo "✓ ucx-ucxx-perftest-img:latest built successfully"
    echo ""
fi

echo "=========================================="
echo "Build process completed successfully!"
echo "=========================================="
echo ""
echo "Available images:"
if [ "$BUILD_RUN" = true ]; then
    echo "  - communicator-run-dnb-img:latest"
fi
if [ "$BUILD_DEV" = true ]; then
    echo "  - communicator-dev-dnb-img:latest"
fi
if [ "$BUILD_PERFTEST" = true ]; then
    echo "  - ucx-ucxx-perftest-img:latest"
fi