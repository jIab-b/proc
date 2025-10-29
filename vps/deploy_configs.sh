#!/bin/bash
set -e  # Exit on error

# Deploy configs to production on droplet
# Copies vps/nginx_config.cfg and runs vps/uvi_launch.sh
# Run as root after syncing code and building frontend
# Usage: ./vps/deploy_configs.sh

echo "🚀 Deploying configs and starting services..."

# Check root
if [[ $EUID -ne 0 ]]; then
    echo "❌ Run as root"
    exit 1
fi

cd /root/proc

# Backup existing if present
NGINX_DEFAULT="/etc/nginx/sites-available/default"
SERVICE_FILE="/etc/systemd/system/webgpu-backend.service"

[[ -f "$NGINX_DEFAULT" ]] && cp "$NGINX_DEFAULT" "${NGINX_DEFAULT}.bak.$(date +%Y%m%d)"
[[ -f "$SERVICE_FILE" ]] && cp "$SERVICE_FILE" "${SERVICE_FILE}.bak.$(date +%Y%m%d)"

# Prepare static files for serving (one-click: copy built dist to public location)
echo "📁 Preparing static frontend files..."
mkdir -p /var/www/webgpu
rm -rf /var/www/webgpu/*  # Clear old files
cp -r /root/proc/webgpu/dist/* /var/www/webgpu/  # Copy built assets
chown -R www-data:www-data /var/www/webgpu
chmod -R 755 /var/www/webgpu
echo "✓ Static files copied to /var/www/webgpu with correct permissions"

# Copy nginx config from vps/
cp vps/nginx_config.cfg "$NGINX_DEFAULT"

# Set up backend service
./vps/uvi_launch.sh

# Apply nginx
nginx -t || { echo "❌ Nginx config error"; exit 1; }
systemctl reload nginx

# Restart backend if needed
systemctl restart webgpu-backend

# Health check
sleep 5
if curl -f -s https://localhost/api/health > /dev/null 2>&1; then
    echo "✓ Full stack healthy!"
else
    echo "⚠️ Health check failed - check logs"
    exit 1
fi

echo "🎉 Deployment complete!"
echo "Site: https://clockworktower.com"
echo "Logs: journalctl -u webgpu-backend -f"
echo "Nginx: tail -f /var/log/nginx/access.log"

chmod +x /root/proc/vps/deploy_configs.sh
