#!/usr/bin/env bash
# Pull the newest image (built by GitHub Actions) into the Azure Container App.
# Run in Azure Cloud Shell:  curl -sL https://raw.githubusercontent.com/NowfaL1N/LungLocAI/main/azure/update.sh | bash
set -euo pipefail
RG="${RG:-lunglocai-rg}"; APP="${APP:-lunglocai}"; IMAGE="${IMAGE:-ghcr.io/nowfal1n/lunglocai:latest}"
az extension add --name containerapp --upgrade -y --only-show-errors >/dev/null
az containerapp update -n "$APP" -g "$RG" --image "$IMAGE" --revision-suffix "r$(date +%m%d%H%M)" -o none --only-show-errors
echo "Updated: https://$(az containerapp show -n "$APP" -g "$RG" --query properties.configuration.ingress.fqdn -o tsv)"
