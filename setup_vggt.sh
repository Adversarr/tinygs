#!/usr/bin/env bash

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

uv sync


if [ $? -ne 0 ]; then
  echo -e "${RED}Failed to ensure the python environment.${NC}"
  exit 1
fi


if [ ! -f model_tracker_fixed_e30.pt ]; then
  wget https://huggingface.co/facebook/VGGT_tracker_fixed/resolve/main/model_tracker_fixed_e30.pt -O model_tracker_fixed_e30.pt
fi

if [ $? -ne 0 ]; then
  echo "Error: wget failed to download model_tracker_fixed_e30.pt"
  exit 1
fi
