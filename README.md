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

## Running with Docker Scripts

The repository includes convenience scripts to run the communicator inside Docker containers with proper GPU and InfiniBand device access. There are two sets of scripts for different environments.

### Small Scripts (AWS Single-GPU Systems)

The `small_server.sh` and `small_client.sh` scripts are designed for single-GPU AWS instances using SDR (Scalable Reliable Datagram). They disable blocking polling and error handling for SDR compatibility.

**small_server.sh** - Starts a server that listens for connections and sends data.

| Option | Default | Description |
|--------|---------|-------------|
| `--listener_port` | 4567 | Port to listen on |
| `--num_chunks` | 10 | Number of data chunks to send |
| `--rows` | 134217728 | Number of rows per chunk |
| `--ucxx_error_handling` | false | UCXX error handling (disabled for SDR) |
| `--ucxx_blocking_polling` | false | Blocking polling (disabled for SDR) |

**small_client.sh** - Starts a client that connects to a server and receives data.

| Option | Default | Description |
|--------|---------|-------------|
| `--listener_port` | 0 | Port to listen on (0 = no listener) |
| `--ports` | 4567 | Comma-separated server ports to connect to |
| `--hostnames` | 127.0.0.1 | Server hostname(s) |
| `--ucxx_error_handling` | false | UCXX error handling (disabled for SDR) |
| `--ucxx_blocking_polling` | false | Blocking polling (disabled for SDR) |

**Example on AWS:**
```bash
# Terminal 1 (Server)
./small_server.sh --listener_port 4567 --num_chunks 10

# Terminal 2 (Client)
./small_client.sh --ports 4567 --hostnames 127.0.0.1
```

### Big Scripts (8-way A100 Systems like "sally")

The `big_server.sh` and `big_client.sh` scripts are designed for multi-GPU systems with full InfiniBand support (e.g., 8-way A100 systems). They enable full UCX features including blocking polling and error handling, and allow GPU selection.

**big_server.sh** - Starts a server on a specific GPU.

| Option | Default | Description |
|--------|---------|-------------|
| `--listener_port` | 4567 | Port to listen on |
| `--num_chunks` | 100 | Number of data chunks to send |
| `--rows` | 16777216 | Number of rows per chunk |
| `--gpu` | 7 | GPU index (0-7) to run on |
| `--ucxx_error_handling` | true | UCXX error handling |
| `--ucxx_blocking_polling` | true | Blocking polling mode |

**big_client.sh** - Starts a client on a specific GPU.

| Option | Default | Description |
|--------|---------|-------------|
| `--listener_port` | 0 | Port to listen on (0 = no listener) |
| `--ports` | 4567 | Comma-separated server ports to connect to |
| `--hostnames` | 127.0.0.1 | Server hostname(s) |
| `--gpu` | 3 | GPU index (0-7) to run on |
| `--ucxx_error_handling` | true | UCXX error handling |
| `--ucxx_blocking_polling` | true | Blocking polling mode |

**Example on 8-way A100:**
```bash
# Terminal 1 (Server on GPU 7)
./big_server.sh --listener_port 4567 --gpu 7

# Terminal 2 (Client on GPU 3)
./big_client.sh --ports 4567 --gpu 3
```

### Key Differences

| Feature | Small Scripts (AWS/SDR) | Big Scripts (8-way A100) |
|---------|------------------------|--------------------------|
| Target system | Single-GPU AWS instances | Multi-GPU systems (sally) |
| InfiniBand devices | uverbs0 only | uverbs0-9 |
| Error handling | Disabled | Enabled |
| Blocking polling | Disabled | Enabled |
| GPU selection | Not available | `--gpu` option (0-7) |

All scripts support `--help` for full option documentation.
