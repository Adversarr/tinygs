#!/usr/bin/env bash

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'
BLUE='\e[34m'


# Ensure we are in the repo root (contains .git folder)
if [ ! -d .git ]; then
  echo "Error: .git folder not found. Please run this script from the repository root."
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  echo -e "${RED}Error: 'uv' not found. Install via 'pip install uv' or 'curl -Ls https://astral.sh/uv/install.sh | sh'.${NC}"
  exit 1
fi

uv sync


if [ $? -ne 0 ]; then
  echo -e "${RED}Failed to ensure the python environment.${NC}"
  exit 1
fi

MODEL_FILE="model_tracker_fixed_e30.pt"
MODEL_URL="https://huggingface.co/facebook/VGGT_tracker_fixed/resolve/main/model_tracker_fixed_e30.pt"

if [ ! -f "${MODEL_FILE}" ]; then
  echo -e "${BLUE}Downloading VGGT checkpoint...${NC}"
  if ! wget -q "${MODEL_URL}" -O "${MODEL_FILE}"; then
    echo -e "${RED}Error: failed to download ${MODEL_FILE}${NC}"
    exit 1
  fi
  echo -e "${GREEN}Downloaded ${MODEL_FILE}.${NC}"
else
  echo -e "${GREEN}Checkpoint already present: ${MODEL_FILE}.${NC}"
fi

echo -e "${GREEN}VGGT setup completed.${NC}"
