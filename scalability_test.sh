#!/bin/bash

# Scalability test script - measures performance with increasing number of servers
# Tests client connecting to 1, 2, 3, ... 8 servers

set -e

# Server list
SERVERS=(
    "ip-172-31-37-183"
    "ip-172-31-39-69"
    "ip-172-31-35-232"
    "ip-172-31-46-205"
    "ip-172-31-47-54"
    "ip-172-31-46-211"
    "ip-172-31-35-0"
    "ip-172-31-35-195"
)

MAX_SERVERS=${#SERVERS[@]}
PORT=4567
SERVER_SCRIPT="bash ./small_server.sh --rows 67108864"
CLIENT_SCRIPT="./big_client.sh"
SERVER_STARTUP_DELAY=5  # seconds to wait for servers to start

echo "==================================================================="
echo "UCXX Communicator Scalability Test"
echo "==================================================================="
echo "Testing with 1 to ${MAX_SERVERS} servers"
echo "Servers: ${SERVERS[*]}"
echo "==================================================================="
echo ""

# Function to start server on remote host
start_server() {
    local hostname=$1
    echo "  Starting server on ${hostname}..."
    ssh "${hostname}" "nohup ${SERVER_SCRIPT} --num_chunks 100 > server.log 2>&1 &" 2>/dev/null
}

# Main test loop
for ((num_servers=1; num_servers<=MAX_SERVERS; num_servers++)); do
    echo "==================================================================="
    echo "Test ${num_servers}/${MAX_SERVERS}: Testing with ${num_servers} server(s)"
    echo "==================================================================="

    # Build comma-separated lists for this iteration
    hostnames=""
    ports=""

    for ((i=0; i<num_servers; i++)); do
        if [ $i -gt 0 ]; then
            hostnames="${hostnames},"
            ports="${ports},"
        fi
        hostnames="${hostnames}${SERVERS[$i]}"
        ports="${ports}${PORT}"
    done

    echo "Hostnames: ${hostnames}"
    echo "Ports: ${ports}"
    echo ""

    # Start servers on remote hosts
    echo "Starting ${num_servers} server(s)..."
    for ((i=0; i<num_servers; i++)); do
        start_server "${SERVERS[$i]}"
    done

    # Wait for servers to be ready
    echo "Waiting ${SERVER_STARTUP_DELAY} seconds for servers to start..."
    sleep ${SERVER_STARTUP_DELAY}

    # Run client test
    log_file="../${num_servers}_server.log"
    echo "Running client test (output: ${log_file})..."

    ${CLIENT_SCRIPT} \
        --ucxx_error_handling false \
        --ucxx_blocking_polling false \
        --ports "${ports}" \
        --hostnames "${hostnames}" \
        >& "${log_file}"

    echo "Client test completed (servers will stop automatically)"
    echo ""
done

echo "==================================================================="
echo "All tests completed!"
echo "==================================================================="
echo "Results:"
for ((i=1; i<=MAX_SERVERS; i++)); do
    log_file="../${i}_server.log"
    if [ -f "${log_file}" ]; then
        echo "  ${i} server(s): ${log_file}"
    fi
done
echo "==================================================================="
