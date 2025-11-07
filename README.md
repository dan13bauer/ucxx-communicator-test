# UCXX Communicator Test

## Compile

Create the cmake configuration:
```
cmake -DCMAKE_CUDA_ARCHITECTURES=80 -S . -B _build
```

Compile:
```
cmake --build _build -j
```

## Running

Each communicator is both client/server at the same time. A listener (==server) is always started.
The number of clients is given by the number of ports to connect to. By default, there is none.

Starting a server without client:
```
CUDA_VISIBLE_DEVICES=7  _build/cpp/communicator
```

Starting a client for connecting to above communicator (needs to run on a different listener_port):
```
CUDA_VISIBLE_DEVICES=6 _build/cpp/communicator -ports 4567 -listener_port 4568
```

## Running on AWS with SDR

Following is true for version 0.21.0 (latest from main as of Nov 5) of UCX and version 0.46.0 of UCXX.

SDR does not support the full features of UCX. If a feature such as blocking poll mode is requested,
UCXX falls back to TCP. If error handling is requested, performance drops by almost one order of magnitude.

Running the communicator on AWS with SRD is possible by:
- disabling blocking polling and use "spinning" polling
- disabling error control
- requesting only TAG and AM features from UCX

The command lines:

server
```
./_build/cpp/communicator -listener_port 4568 -ports 4567 -ucxx_blocking_polling=false -ucxx_error_handling=false -hostname=ip-172-31-0-45
```

client
```
./_build/cpp/communicator -ucxx_blocking_polling=false -ucxx_error_handling=false -rows 10000000
```
