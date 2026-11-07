#!/usr/bin/env bash
# Remove everything LungLocAI created in Azure (stops all charges).
az group delete -n "${RG:-lunglocai-rg}" -y && echo "Deleted resource group ${RG:-lunglocai-rg}"
