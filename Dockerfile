# Start from Ubuntu 24.04 base image
FROM nvidia/cuda:12.8.0-devel-ubuntu24.04

# Prevent interactive prompts during package installs
ENV DEBIAN_FRONTEND=noninteractive

# Update and install prerequisites
RUN apt-get update && dpkg --configure -a && apt-get autoremove && apt-get autoclean && \
    apt-get install -y \
    wget \
    gnupg2 \
    software-properties-common

RUN apt-get install -y \
    build-essential \
    autoconf \
    automake \
    git git-lfs \    
    libz-dev \
    libssl-dev \
    zlib1g-dev \
    emacs-nox \
    zsh \
    libgflags-dev \
    libfmt-dev \
    libtool \
    ninja-build \
    libnuma-dev \
    libibverbs-dev \
    librdmacm-dev

# Install cmake.
WORKDIR /usr/local/share
RUN wget https://github.com/Kitware/CMake/releases/download/v3.31.9/cmake-3.31.9-linux-x86_64.tar.gz && \
    tar fvx cmake-3.31.9-linux-x86_64.tar.gz

RUN ln -fs /usr/local/share/cmake-3.31.9-linux-x86_64/bin/cmake /usr/local/bin && \
    ln -fs /usr/local/share/cmake-3.31.9-linux-x86_64/bin/ctest /usr/local/bin && \
    ln -fs /usr/local/share/cmake-3.31.9-linux-x86_64/bin/cmake-gui /usr/local/bin && \
    ln -fs /usr/local/share/cmake-3.31.9-linux-x86_64/bin/cpack /usr/local/bin && \
    ln -fs /usr/local/share/cmake-3.31.9-linux-x86_64/bin/ccmake /usr/local/bin

# Build UCX from master branch
WORKDIR /workspace
RUN git clone --depth 1 https://github.com/openucx/ucx.git

WORKDIR /workspace/ucx
RUN ./autogen.sh && \
    ./configure --prefix=/usr/local \
                --enable-mt \
                --enable-optimizations \
                --enable-srd \
		--with-cuda=/usr/local/cuda \
		--without-go \
        	--without-java \		
                --disable-logging \
                --disable-debug \
                --disable-assertions \
                --disable-params-check

RUN make -j$(nproc) && \
    make install && \
    ldconfig && \
    cd .. && \
    rm -rf ucx

# Clone UCXX version 0.46.0
WORKDIR /workspace
RUN git clone https://github.com/rapidsai/ucxx.git && \
    cd ucxx && \
    git checkout branch-0.46

# Configure and build ucxx 
WORKDIR /workspace/ucxx/cpp
RUN mkdir -p build && cd build && \
    cmake .. -DCMAKE_INSTALL_PREFIX="/usr/local" \
      -DBUILD_TESTS=OFF \
      -DBUILD_BENCHMARKS=ON \
      -DCMAKE_BUILD_TYPE=Release \
      -DUCXX_ENABLE_PYTHON=OFF \
      -DUCXX_ENABLE_RMM=ON \
      -DUCXX_BENCHMARKS_ENABLE_CUDA=ON

WORKDIR /workspace/ucxx/cpp/build

RUN make -j
RUN make -j install

ARG ARROW_VERSION=21.0.0
WORKDIR /opt
RUN git clone https://github.com/apache/arrow.git && \
    cd arrow && \
    git checkout apache-arrow-${ARROW_VERSION} && \
    mkdir -p cpp/build && cd cpp/build && \
    cmake .. -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/opt/arrow \
      -DARROW_BUILD_SHARED=ON \
      -DARROW_BUILD_STATIC=OFF \
      -DARROW_PARQUET=ON \
      -DARROW_DATASET=ON \
      -DARROW_FILESYSTEM=ON \
      -DARROW_CSV=ON \
      -DARROW_JSON=ON \
      -DARROW_IPC=ON \
      -DARROW_COMPUTE=ON \
      -DARROW_WITH_UTF8PROC=OFF \
      -DARROW_BUILD_TESTS=OFF && \
    ninja && ninja install

# Clone cudf and checkout branch 25.10
WORKDIR /workspace
RUN git clone https://github.com/rapidsai/cudf.git && \
    cd cudf && \
    git checkout branch-25.10

# Configure and build libcudf (C++ core). Point to Arrow install via CMAKE_PREFIX_PATH.
WORKDIR /workspace/cudf/cpp
RUN mkdir -p build && cd build && \
    cmake .. \
      -DCMAKE_INSTALL_PREFIX="/usr/local" \
      -DCMAKE_CUDA_ARCHITECTURES="80" \
      -DUSE_NVTX=OFF \
      -DBUILD_TESTS=OFF \
      -DBUILD_BENCHMARKS=OFF \
      -DDISABLE_DEPRECATION_WARNINGS=ON \
      -DCUDF_USE_PER_THREAD_DEFAULT_STREAM=OFF \
      -DCUDF_LARGE_STRINGS_DISABLED=OFF \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_PREFIX_PATH="/opt/arrow"

RUN cmake --build build -j -v
RUN cmake --build build -j --target install -v

# Set up working directory for the project
WORKDIR /workspace

# Copy project files
COPY . .

# Clean any existing build artifacts
RUN rm -rf _build

# Configure and build the project
RUN cmake -B _build -S . \
    -DCMAKE_BUILD_TYPE=Release

RUN cmake --build _build -j$(nproc)

# Set the entrypoint to ucxx_perftest
ENTRYPOINT ["/workspace/_build/cpp/communicator"]
