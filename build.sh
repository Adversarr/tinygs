RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'
BLUE='\e[34m'

# Detect cmake
if ! command -v cmake >/dev/null 2>&1; then
    echo -e "${RED}Error: cmake is not installed or not in PATH.${NC}" >&2
    exit 1
fi

# Detect nvcc
if ! command -v nvcc >/dev/null 2>&1; then
    echo -e "${RED}Error: nvcc is not installed or not in PATH.${NC}" >&2
    exit 1
fi

NVCC=$(command -v nvcc)
NPROC=$(nproc)
echo -e "${GREEN}nvcc found at: ${NVCC}, ${NPROC} cores available for build.${NC}"

# Ensure we are in a Git repository (directory containing .git)
if [ ! -d .git ]; then
    echo "Error: This script must be run from the root of our repository (no .git directory found)." >&2
    exit 1
fi

cmake -S . -B build/Release     \
  -DCMAKE_BUILD_TYPE=Release    \
  -DCMAKE_CUDA_COMPILER=${NVCC}

if [ $? -ne 0 ]; then
    echo -e "${RED}CMake configure failed. Please ensure your terminal has access to GitHub/GitLab to download 3rd-party dependencies.${NC}" >&2
    exit 1
fi

cmake --build build/Release --config Release --target video_to_png config_train -j $(nproc)
if [ $? -ne 0 ]; then
    echo -e "${RED}Error: CMake build failed.${NC}" >&2
    exit 1
fi

# Detect whether the built executables exist
if [ ! -f build/Release/examples/video_to_png ]; then
    echo -e "${RED}Error: build/Release/video_to_png not found.${NC}" >&2
    exit 1
fi

if [ ! -f build/Release/examples/config_train ]; then
    echo -e "${RED}Error: build/Release/config_train not found.${NC}" >&2
    exit 1
fi

cp build/Release/examples/video_to_png .
cp build/Release/examples/config_train .

echo -e "${BLUE}Essential binaries built!${NC}"
