#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_DIR="/opt/memecoin-bot"

echo "=== Memecoin Quant Desk Deployment ==="
echo "Source: $SCRIPT_DIR"
echo "Target: $DEPLOY_DIR"

if [[ $EUID -ne 0 ]]; then
   echo "This script must be run as root (use sudo)"
   exit 1
fi

echo "Creating deployment directory..."
mkdir -p "$DEPLOY_DIR"/{data,logs,models,config}

echo "Copying application files..."
rsync -av --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' \
    "$SCRIPT_DIR/" "$DEPLOY_DIR/"

echo "Setting up environment file..."
if [[ ! -f "$DEPLOY_DIR/.env" ]]; then
    cp "$DEPLOY_DIR/.env.example" "$DEPLOY_DIR/.env"
    echo "Created .env from example - EDIT IT WITH YOUR KEYS!"
fi

echo "Setting permissions..."
chown -R root:root "$DEPLOY_DIR"
chmod 600 "$DEPLOY_DIR/.env" 2>/dev/null || true

echo "Installing Docker if needed..."
if ! command -v docker &> /dev/null; then
    curl -fsSL https://get.docker.com | sh
fi

if ! docker compose version &> /dev/null; then
    apt-get update && apt-get install -y docker-compose-plugin
fi

echo "Building Docker image..."
cd "$DEPLOY_DIR"
docker compose build --pull

echo "Installing systemd service..."
cp "$DEPLOY_DIR/memecoin-bot.service" /etc/systemd/system/
systemctl daemon-reload
systemctl enable memecoin-bot

echo ""
echo "=== Deployment Complete ==="
echo ""
echo "Next steps:"
echo "1. Edit $DEPLOY_DIR/.env with your API keys"
echo "2. Start the service: systemctl start memecoin-bot"
echo "3. Check logs: journalctl -u memecoin-bot -f"
echo "4. Check health: curl http://localhost:8080/health"
echo ""
echo "Useful commands:"
echo "  systemctl status memecoin-bot"
echo "  systemctl restart memecoin-bot"
echo "  docker compose -f $DEPLOY_DIR/docker-compose.yml logs -f"
echo "  docker compose -f $DEPLOY_DIR/docker-compose.yml pull && docker compose -f $DEPLOY_DIR/docker-compose.yml up -d --build"