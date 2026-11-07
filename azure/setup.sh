#!/usr/bin/env bash
# One-time setup of LungLocAI on Azure Container Apps (works with Azure for Students).
# Run it in Azure Cloud Shell (https://shell.azure.com, choose "Bash"):
#
#   curl -sL https://raw.githubusercontent.com/NowfaL1N/LungLocAI/main/azure/setup.sh | bash
#
# Creates: resource group lunglocai-rg, a Container Apps environment and the app "lunglocai"
# (2 vCPU / 4 GiB, scales to zero when idle -> stays inside the monthly free grant).
set -uo pipefail

IMAGE="${IMAGE:-ghcr.io/nowfal1n/lunglocai:latest}"
RG="${RG:-lunglocai-rg}"
ENV_NAME="${ENV_NAME:-lunglocai-env}"
APP="${APP:-lunglocai}"
# Azure for Students only allows some regions; try these in order.
REGIONS="${REGIONS:-centralindia southindia southeastasia eastasia uaenorth westeurope northeurope eastus westus2}"

say() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }

say "Subscription: $(az account show --query name -o tsv 2>/dev/null || echo 'NOT LOGGED IN')"
az account show >/dev/null 2>&1 || { echo "Open this in Azure Cloud Shell (https://shell.azure.com)"; exit 1; }

say "Installing Container Apps CLI extension"
az extension add --name containerapp --upgrade -y --only-show-errors >/dev/null

say "Registering resource providers (first time takes ~1-2 minutes)"
for ns in Microsoft.App Microsoft.OperationalInsights; do
  az provider register --namespace "$ns" --wait --only-show-errors
done

if az group show -n "$RG" >/dev/null 2>&1; then
  say "Resource group $RG already exists"
else
  for r in $REGIONS; do
    say "Creating resource group in $r"
    az group create -n "$RG" -l "$r" -o none --only-show-errors 2>/tmp/err && break
    echo "  not allowed here: $(head -c 200 /tmp/err)"
  done
  az group show -n "$RG" >/dev/null 2>&1 || { echo "Could not create a resource group. See allowed regions: az policy assignment list -o table"; exit 1; }
fi

LOC=$(az containerapp env show -n "$ENV_NAME" -g "$RG" --query location -o tsv 2>/dev/null || true)
if [ -n "$LOC" ]; then
  say "Container Apps environment already exists ($LOC)"
else
  for r in $REGIONS; do
    say "Creating Container Apps environment in $r (takes 2-4 minutes)"
    if az containerapp env create -n "$ENV_NAME" -g "$RG" -l "$r" --logs-destination none -o none --only-show-errors 2>/tmp/err; then
      LOC="$r"; break
    fi
    echo "  not available here: $(head -c 300 /tmp/err)"
  done
  [ -n "$LOC" ] || { echo "Could not create the environment in any region. Allowed regions: az policy assignment list -o table ; then rerun with REGIONS=\"<region>\""; exit 1; }
fi

if az containerapp show -n "$APP" -g "$RG" >/dev/null 2>&1; then
  say "App exists - updating it to $IMAGE"
  az containerapp update -n "$APP" -g "$RG" --image "$IMAGE" -o none --only-show-errors || exit 1
else
  say "Creating the app from $IMAGE"
  az containerapp create -n "$APP" -g "$RG" --environment "$ENV_NAME" \
    --image "$IMAGE" --target-port 7860 --ingress external \
    --cpu 2 --memory 4Gi --min-replicas 0 --max-replicas 1 \
    --env-vars JOB_TTL_HOURS=2 MAX_UPLOAD_MB=1500 \
    -o none --only-show-errors || exit 1
fi

URL="https://$(az containerapp show -n "$APP" -g "$RG" --query properties.configuration.ingress.fqdn -o tsv)"

say "Optional: credentials so GitHub can update the site automatically on every push"
SP_JSON=$(az ad sp create-for-rbac --name "lunglocai-github-deploy" --role contributor \
  --scopes "$(az group show -n "$RG" --query id -o tsv)" --json-auth --only-show-errors 2>/tmp/sperr)
if [ -n "$SP_JSON" ]; then
  echo "Copy EVERYTHING between the lines into a GitHub secret named AZURE_CREDENTIALS"
  echo "(https://github.com/NowfaL1N/LungLocAI/settings/secrets/actions/new):"
  echo "------------------------------------------------------------------"
  echo "$SP_JSON"
  echo "------------------------------------------------------------------"
else
  echo "Your account can't create deploy credentials ($(head -c 200 /tmp/sperr))."
  echo "No problem: after pushing changes to GitHub, update the site with"
  echo "  curl -sL https://raw.githubusercontent.com/NowfaL1N/LungLocAI/main/azure/update.sh | bash"
fi

say "DONE. LungLocAI is live at:  $URL"
echo "(first visit after idle takes ~30-60 s while the container starts)"
