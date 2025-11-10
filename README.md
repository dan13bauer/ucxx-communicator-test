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

The repository includes convenience scripts to run the communicator inside Docker containers with proper GPU and InfiniBand device access.

### Server Script

The `run_comms_server.sh` script starts a communicator server that listens for incoming connections and generates data to send.

**Default configuration:**
- Listener port: 4567
- Number of chunks: 10
- Rows per chunk: 10000000
- UCXX error handling: enabled
- UCXX blocking polling: enabled

**Usage:**
```bash
# Run with defaults
./run_comms_server.sh

# Run with custom parameters
./run_comms_server.sh --listener_port 5000 --num_chunks 20 --rows 5000000

# Show help
./run_comms_server.sh --help
```

**Available options:**
- `--listener_port <port>` - Port to listen on (default: 4567)
- `--num_chunks <number>` - Number of data chunks to send (default: 10)
- `--rows <number>` - Number of rows per chunk (default: 10000000)
- `--ucxx_error_handling <bool>` - Enable UCXX error handling (default: true)
- `--ucxx_blocking_polling <bool>` - Use blocking polling mode (default: true)

### Client Script

The `run_comms_client.sh` script starts a communicator client that connects to a server and receives data.

**Default configuration:**
- Listener port: 4568
- Connect to ports: 4567
- Hostname: 127.0.0.1
- UCXX error handling: enabled
- UCXX blocking polling: enabled

**Usage:**
```bash
# Run with defaults (connects to server on port 4567)
./run_comms_client.sh

# Run with custom parameters
./run_comms_client.sh --ports "4567,4568,4569" --hostname "192.168.1.100"

# Show help
./run_comms_client.sh --help
```

**Available options:**
- `--listener_port <port>` - Port to listen on (default: 4568)
- `--ports <port_list>` - Comma-separated list of server ports to connect to (default: "4567")
- `--hostname <hostname>` - Hostname/IP of the server to connect to (default: "127.0.0.1")
- `--ucxx_error_handling <bool>` - Enable UCXX error handling (default: true)
- `--ucxx_blocking_polling <bool>` - Use blocking polling mode (default: true)

### Example: Running Server and Client

**Terminal 1 (Server):**
```bash
./run_comms_server.sh --listener_port 4567 --num_chunks 10
```

**Terminal 2 (Client):**
```bash
./run_comms_client.sh --ports 4567 --listener_port 4568
```

The client will connect to the server on port 4567 and receive the data chunks. Both containers run with full GPU and InfiniBand device access for high-performance RDMA communication.
