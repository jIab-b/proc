#!/bin/bash
set -e  # Exit on error

# Script to set up and launch uvicorn backend service on droplet
# Run as root in /root/proc after syncing code
# Usage: ./uvi_launch.sh

echo "🚀 Setting up uvicorn backend service..."

# Check root
if [[ $EUID -ne 0 ]]; then
    echo "❌ Run as root"
    exit 1
fi

cd /root/proc

# Service file path
SERVICE_FILE="/etc/systemd/system/webgpu-backend.service"

# Idempotent: Skip if exists and running
if [[ -f "$SERVICE_FILE" ]] && systemctl is-active --quiet webgpu-backend; then
    echo "✓ Service already running"
    systemctl status webgpu-backend
    exit 0
fi

# Create service file
cat > "$SERVICE_FILE" << 'EOF'
[Unit]
Description=WebGPU Minecraft Editor Backend (FastAPI + Uvicorn)
After=network.target

[Service]
User=root
Group=root
WorkingDirectory=/root/proc
Environment=PATH=/root/proc/venv/bin
ExecStart=/root/proc/venv/bin/python main.py --prod
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

# Activate
systemctl daemon-reload
systemctl enable webgpu-backend
systemctl start webgpu-backend
systemctl status webgpu-backend

echo "🎉 Backend launched!"