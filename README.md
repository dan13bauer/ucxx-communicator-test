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