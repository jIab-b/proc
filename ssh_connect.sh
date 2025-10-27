#!/bin/bash

# SSH connection script - replicates the default behavior of ssh_connect.py
# Connects to remote server using runpodprivate SSH key

# Default values (matching ssh_connect.py)
DEFAULT_HOST="209.38.93.185"
DEFAULT_KEY="/home/beed/.ssh/runpodprivate"

# Parse command line arguments
HOST="$DEFAULT_HOST"
KEY_PATH="$DEFAULT_KEY"

while [[ $# -gt 0 ]]; do
    case $1 in
        --host)
            HOST="$2"
            shift 2
            ;;
        --key)
            KEY_PATH="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--host HOST] [--key KEY_PATH]"
            exit 1
            ;;
    esac
done

# Check if key file exists
if [[ ! -f "$KEY_PATH" ]]; then
    echo "Error: SSH key not found at $KEY_PATH"
    exit 1
fi

# Set proper permissions on the key file (SSH requires 600 or 400)
chmod 600 "$KEY_PATH"

echo "Connecting to $HOST..."

# Connect using standard SSH with the private key
ssh -i "$KEY_PATH" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null root@"$HOST"

echo "Connection closed."
