# Get the first three command-line arguments
ROOT="$1"
SCENE_ID="$2"
OUTPUT_DIR="$3"
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'
BLUE='\e[34m'

if [ -z "$OUTPUT_DIR" ]; then
  OUTPUT_DIR="output"
fi

# Check if required executables exist
for cmd in "./video_to_png" "./config_train"; do
  if [ ! -f "$cmd" ]; then
    echo -e "${RED}Error: $cmd not found${NC}, build first."
    exit 1
  fi
done

# Check if required Python scripts exist
for script in "scripts/aligner.py" "scripts/gen_config.py"; do
  if [ ! -f "$script" ]; then
    echo -e "${RED}Error: $script not found${NC}. Are you in the root directory of the project?${NC}"
    exit 1
  fi
done

mkdir -p "${OUTPUT_DIR}/${SCENE_ID}/images"
mkdir -p "${OUTPUT_DIR}/${SCENE_ID}/aligned_points"

echo -e "${GREEN}Output directory: ${OUTPUT_DIR}/${SCENE_ID}${NC}"

echo -e "${GREEN}Launching scene ${SCENE_ID} from ${ROOT}${NC}"
echo -e "${GREEN}1. Preparing data: video_to_png${NC}"
./video_to_png             \
  -f "${ROOT}/${SCENE_ID}" \
  -i "${SCENE_ID}"         \
  -o "${OUTPUT_DIR}/${SCENE_ID}/images/"

if [ $? -eq 0 ]; then
  echo -e "${GREEN}video_to_png completed successfully${NC}"
else
  echo -e "${RED}video_to_png failed${NC}"
  exit 1
fi

echo -e "${GREEN}2. Initial Point Estimate${NC}"
echo -e "${BLUE}If you encountnered network issue, please set the HF-mirror to download the model. \"export HF_ENDPOINT=https://hf-mirror.com\"${NC}"

python scripts/aligner.py \
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
python scripts/gen_config.py \
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
time
./config_train --config "${OUTPUT_DIR}/${SCENE_ID}/config.json"
echo -e "${GREEN}The timer should end now.${NC}"
echo -e "${GREEN} ============================================================================ ${NC}"
