#!/usr/bin/env bash
# Turn on private research storage + the owner-only admin page for the LungLocAI Azure app.
# Run in Azure Cloud Shell:
#   curl -sL https://raw.githubusercontent.com/NowfaL1N/LungLocAI/main/azure/enable-storage.sh | bash
#
# You will be asked (input hidden, never printed or logged) for:
#   1. a GitHub fine-grained token that can read/write ONLY your private LungLocAI-data repo
#   2. an admin password for https://<your-site>/admin
# Both are stored as Azure Container Apps *secrets*.
set -uo pipefail
RG="${RG:-lunglocai-rg}"; APP="${APP:-lunglocai}"; DATA_REPO="${DATA_REPO:-NowfaL1N/LungLocAI-data}"
say() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }

az containerapp show -n "$APP" -g "$RG" >/dev/null 2>&1 || { echo "App $APP not found in $RG - run azure/setup.sh first."; exit 1; }

say "Storage repository: $DATA_REPO (must be PRIVATE)"
printf "Paste the GitHub token (starts with github_pat_), then press Enter: "
read -rs TOKEN < /dev/tty; echo
[ -n "$TOKEN" ] || { echo "No token entered."; exit 1; }

TOKEN=$(printf '%s' "$TOKEN" | tr -d '[:space:]')
echo "Received a token of ${#TOKEN} characters starting with '${TOKEN:0:11}'."
UCODE=$(curl -s -o /tmp/user.json -w "%{http_code}" -H "Authorization: Bearer $TOKEN" "https://api.github.com/user")
if [ "$UCODE" != "200" ]; then
  echo "GitHub rejected this token (HTTP $UCODE): it was not pasted completely, or it expired/was deleted."
  echo "Tip: in Cloud Shell paste with right-click -> Paste or Ctrl+Shift+V (nothing is shown while typing)."; exit 1
fi
echo "Token belongs to GitHub user: $(python3 -c "import json;print(json.load(open('/tmp/user.json'))['login'])")"
CODE=$(curl -s -o /tmp/repo.json -w "%{http_code}" -H "Authorization: Bearer $TOKEN" -H "Accept: application/vnd.github+json" "https://api.github.com/repos/$DATA_REPO")
if [ "$CODE" != "200" ]; then
  echo "The token is valid but has NO access to $DATA_REPO (HTTP $CODE)."
  echo "Edit it at https://github.com/settings/personal-access-tokens : Repository access -> Only select repositories -> $DATA_REPO,"
  echo "and Repository permissions -> Contents -> Read and write. Then run this script again."; exit 1
fi
PERM=$(python3 -c "import json;print(json.load(open('/tmp/repo.json')).get('permissions',{}).get('push'))")
[ "$PERM" = "True" ] || echo "Warning: the token may be read-only - set Contents -> Read and write."
if ! grep -q '"private": *true' /tmp/repo.json; then echo "$DATA_REPO is PUBLIC. Make it private first (Settings -> Danger Zone)."; exit 1; fi
echo "Token OK - repository is private."

while true; do
  printf "Choose an admin password (min 10 characters): "; read -rs PW < /dev/tty; echo
  printf "Repeat it: "; read -rs PW2 < /dev/tty; echo
  [ "${#PW}" -ge 10 ] && [ "$PW" = "$PW2" ] && break
  echo "Passwords differ or are shorter than 10 characters - try again."
done

say "Saving secrets in Azure"
az containerapp secret set -n "$APP" -g "$RG" --secrets "data-token=$TOKEN" "admin-pw=$PW" -o none --only-show-errors || exit 1
unset TOKEN PW PW2
say "Restarting the app with storage enabled"
az containerapp update -n "$APP" -g "$RG" \
  --set-env-vars "DATA_REPO=$DATA_REPO" "DATA_REPO_TOKEN=secretref:data-token" "ADMIN_PASSWORD=secretref:admin-pw" \
  -o none --only-show-errors || exit 1
URL="https://$(az containerapp show -n "$APP" -g "$RG" --query properties.configuration.ingress.fqdn -o tsv)"
say "Done. Admin page: $URL/admin  (log in, then press 'Test storage')"
