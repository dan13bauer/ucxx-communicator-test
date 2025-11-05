cmake_minimum_required(VERSION 3.28)

project(ucxx-communicator-test
    VERSION 0.1.0
    LANGUAGES CXX CUDA
)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build type default
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(CUDAToolkit REQUIRED)
set(CUDA_TOOLKIT_ROOT_DIR $ENV{CUDA_ROOT})
set(CUDA_INCLUDE_DIR ${CUDA_TOOLKIT_ROOT_DIR}/include)
set(CMAKE_CUDA_ARCHITECTURES $ENV{CUDA_ARCHITECTURES})

# Pick up dependencies installed in the system
find_package(cudf REQUIRED)
find_package(rmm REQUIRED)
find_package(ucxx REQUIRED)
find_package(fmt REQUIRED)
find_package(ucx REQUIRED)

# Compile flags, taken from distributed-join CMakeLists.txt
add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-Wall>")
add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-Werror>")
add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:--compiler-options=-Wall>")
add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:--compiler-options=-Werror>")

add_subdirectory(cpp)
