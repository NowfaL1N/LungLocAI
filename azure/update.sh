#!/usr/bin/env bash
# Deploy the newest image (built by GitHub Actions for the latest commit on main) to the Azure Container App.
# Run in Azure Cloud Shell:  curl -sL https://raw.githubusercontent.com/NowfaL1N/LungLocAI/main/azure/update.sh | bash
set -euo pipefail
RG="${RG:-lunglocai-rg}"; APP="${APP:-lunglocai}"; REPO="${REPO:-NowfaL1N/LungLocAI}"
SHA=$(curl -s "https://api.github.com/repos/$REPO/commits/main" | python3 -c "import sys,json;print(json.load(sys.stdin)['sha'])")
IMAGE="${IMAGE:-ghcr.io/$(echo "$REPO" | tr '[:upper:]' '[:lower:]' | cut -d/ -f1)/lunglocai:$SHA}"
echo "Deploying $IMAGE"
az extension add --name containerapp --upgrade -y --only-show-errors >/dev/null
# a new image tag always creates a new revision (works on every environment type)
az containerapp update -n "$APP" -g "$RG" --image "$IMAGE" -o none --only-show-errors
echo "Updated: https://$(az containerapp show -n "$APP" -g "$RG" --query properties.configuration.ingress.fqdn -o tsv)"
