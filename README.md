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