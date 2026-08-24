#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_DIR="/opt/memecoin-shadow"
SERVICE_NAME="memecoin-shadow"

echo "=== Memecoin Shadow Desk (dry-run only) ==="
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

echo "Setting up dry-run environment file..."
if [[ ! -f "$DEPLOY_DIR/.env" ]]; then
    cp "$DEPLOY_DIR/.env.example" "$DEPLOY_DIR/.env"
    echo "Created .env from example. API data keys are optional. Do not add a wallet key."
fi

if grep -Eq '^ALLOW_LIVE_TRADING=.+$|^SOLANA_PRIVATE_KEY=.+$' "$DEPLOY_DIR/.env"; then
    echo "Refusing deployment: shadow .env contains a live acknowledgement or wallet private key"
    exit 1
fi

echo "Setting permissions..."
chown -R root:root "$DEPLOY_DIR"
chown -R 10001:10001 "$DEPLOY_DIR/data" "$DEPLOY_DIR/logs" "$DEPLOY_DIR/models"
chmod 600 "$DEPLOY_DIR/.env" 2>/dev/null || true

if ! command -v docker &> /dev/null || ! docker compose version &> /dev/null; then
    echo "Docker with the Compose plugin must be installed by the VPS administrator"
    exit 1
fi

echo "Building Docker image..."
cd "$DEPLOY_DIR"
docker compose -p memecoin-shadow build --pull

echo "Installing systemd service..."
cp "$DEPLOY_DIR/memecoin-shadow.service" /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now "$SERVICE_NAME"

echo ""
echo "=== Deployment Complete ==="
echo ""
echo "Next steps:"
echo "The isolated dry-run service is started."
echo "Health: curl http://127.0.0.1:18080/health"
echo ""
echo "Useful commands:"
echo "  systemctl status $SERVICE_NAME"
echo "  journalctl -u $SERVICE_NAME -f"
echo "  docker compose -p memecoin-shadow -f $DEPLOY_DIR/docker-compose.yml logs -f"
