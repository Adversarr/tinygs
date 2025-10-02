#!/usr/bin/env bash

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'
BLUE='\e[34m'

show_help() {
  cat <<'EOF'
Usage: ./launch.sh <root> <scene_id> [output_dir]

Arguments:
  root        Path to dataset root containing the scene folder
  scene_id    Scene folder name under root
  output_dir  Output directory (default: output)

Environment variables:
  VIDEO_TO_PNG  Path to video_to_png executable (default: ./video_to_png)
  CONFIG_TRAIN  Path to config_train executable (default: ./config_train)
  PYTHON        Python interpreter (default: python)

Examples:
  ./launch.sh /data my_scene
  OUTPUT_DIR=out ./launch.sh /data my_scene
  PYTHON=python3 VIDEO_TO_PNG=./bin/video_to_png ./launch.sh /data my_scene
EOF
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  show_help
  exit 0
fi

# Positional arguments
ROOT="${1:-}"
SCENE_ID="${2:-}"
OUTPUT_DIR="${3:-output}"

if [ -z "${ROOT}" ] || [ -z "${SCENE_ID}" ]; then
  echo -e "${RED}Error: missing required arguments 'root' and 'scene_id'.${NC}"
  show_help
  exit 1
fi

# Executable overrides via env vars
VIDEO_TO_PNG="${VIDEO_TO_PNG:-./video_to_png}"
CONFIG_TRAIN="${CONFIG_TRAIN:-./config_train}"
PYTHON_BIN="${PYTHON:-python}"

# Validate executables
for cmd in "${VIDEO_TO_PNG}" "${CONFIG_TRAIN}"; do
  if [ ! -x "$cmd" ]; then
    if [ -f "$cmd" ]; then
      echo -e "${RED}Error: $cmd exists but is not executable.${NC}"
    else
      echo -e "${RED}Error: $cmd not found${NC}, build first or set env var to its path."
    fi
    exit 1
  fi
done

# Validate Python
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo -e "${RED}Error: Python interpreter not found: ${PYTHON_BIN}.${NC} Set PYTHON=python3 if needed."
  exit 1
fi

# Check required Python scripts
for script in "scripts/aligner.py" "scripts/gen_config.py"; do
  if [ ! -f "$script" ]; then
    echo -e "${RED}Error: $script not found.${NC} Are you in the project root?"
    exit 1
  fi
done

# Validate input scene directory
SCENE_PATH="${ROOT}/${SCENE_ID}"
if [ ! -d "${SCENE_PATH}" ]; then
  echo -e "${RED}Error: scene directory not found: ${SCENE_PATH}.${NC}"
  exit 1
fi

# Prepare output directories
mkdir -p "${OUTPUT_DIR}/${SCENE_ID}/images"
mkdir -p "${OUTPUT_DIR}/${SCENE_ID}/aligned_points"

echo -e "${GREEN}Output directory: ${OUTPUT_DIR}/${SCENE_ID}${NC}"

echo -e "${GREEN}Launching scene ${SCENE_ID} from ${ROOT}${NC}"
echo -e "${GREEN}1. Preparing data: video_to_png${NC}"
"${VIDEO_TO_PNG}"        \
  -f "${SCENE_PATH}"     \
  -i "${SCENE_ID}"       \
  -o "${OUTPUT_DIR}/${SCENE_ID}/images/"

if [ $? -eq 0 ]; then
  echo -e "${GREEN}video_to_png completed successfully${NC}"
else
  echo -e "${RED}video_to_png failed${NC}"
  exit 1
fi

echo -e "${GREEN}2. Initial Point Estimate${NC}"
echo -e "${BLUE}If you encountered network issues, set a HF mirror to download the model: export HF_ENDPOINT=https://hf-mirror.com${NC}"

"${PYTHON_BIN}" scripts/aligner.py \
  --root "${ROOT}" \
  --id "${SCENE_ID}" \
  --working_dir "${OUTPUT_DIR}/${SCENE_ID}/" \
  --out "${OUTPUT_DIR}/${SCENE_ID}/aligned_points/"

if [ $? -eq 0 ]; then
  echo -e "${GREEN}Initial Point Estimate completed successfully${NC}"
else
  echo -e "${RED}Initial Point Estimate failed${NC}"
  exit 1
fi

echo -e "${GREEN}3. Generate Config${NC}"
"${PYTHON_BIN}" scripts/gen_config.py \
  --root "${ROOT}" \
  --id "${SCENE_ID}" \
  --working_dir "${OUTPUT_DIR}/${SCENE_ID}/" \
  --out "${OUTPUT_DIR}/${SCENE_ID}/config.json"

if [ $? -eq 0 ]; then
  echo -e "${GREEN}gen_config completed successfully${NC}"
else
  echo -e "${RED}gen_config failed${NC}"
  exit 1
fi

echo -e "${GREEN} ============================================================================ ${NC}"
echo -e "${GREEN}4. Train${NC}"
echo -e "${GREEN}The timer should start now.${NC}"
time "${CONFIG_TRAIN}" --config "${OUTPUT_DIR}/${SCENE_ID}/config.json"
echo -e "${GREEN}The timer should end now.${NC}"
echo -e "${GREEN} ============================================================================ ${NC}"
