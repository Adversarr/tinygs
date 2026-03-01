#!/usr/bin/env bash

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'
BLUE='\e[34m'

# Usage helper
if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  cat <<'EOF'
Usage: ./build.sh

Environment variables:
  NVCC        Path to nvcc binary if not in PATH
  BUILD_TYPE  CMake build type (Release|Debug). Default: Release
  JOBS        Parallel build jobs. Default: number of cores
  TARGETS     CMake targets to build. Default: "video_to_png config_train"
EOF
  exit 0
fi

# Detect cmake
if ! command -v cmake >/dev/null 2>&1; then
    echo -e "${RED}Error: cmake is not installed or not in PATH.${NC}" >&2
    exit 1
fi

# Resolve NVCC (env var takes precedence, then PATH)
if [ -n "${NVCC:-}" ]; then
    NVCC_BIN="${NVCC}"
elif command -v nvcc >/dev/null 2>&1; then
    NVCC_BIN="$(command -v nvcc)"
else
    echo -e "${RED}Error: nvcc not found. Set NVCC=/path/to/nvcc or add nvcc to PATH.${NC}" >&2
    exit 1
fi

if [ ! -x "${NVCC_BIN}" ]; then
    echo -e "${RED}Error: NVCC points to a non-executable: ${NVCC_BIN}.${NC}" >&2
    exit 1
fi

NPROC=$(nproc)
JOBS="${JOBS:-${NPROC}}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
TARGETS="${TARGETS:-config_train}"
BUILD_DIR="build/${BUILD_TYPE}"

echo -e "${GREEN}nvcc found at: ${NVCC_BIN}, ${JOBS} cores available for build.${NC}"

# Ensure we are at repository root (CMakeLists.txt should be present)
if [ ! -f CMakeLists.txt ]; then
    echo "Error: Run this script from the repository root (CMakeLists.txt not found)." >&2
    exit 1
fi

cmake -S . -B "${BUILD_DIR}"     \
  -DCMAKE_BUILD_TYPE="${BUILD_TYPE}"    \
  -DCMAKE_CUDA_COMPILER="${NVCC_BIN}"

if [ $? -ne 0 ]; then
    echo -e "${RED}CMake configure failed. Please ensure your terminal has access to GitHub/GitLab to download 3rd-party dependencies.${NC}" >&2
    exit 1
fi

cmake --build "${BUILD_DIR}" --config "${BUILD_TYPE}" --target ${TARGETS} -j "${JOBS}"
if [ $? -ne 0 ]; then
    echo -e "${RED}Error: CMake build failed.${NC}" >&2
    exit 1
fi

# Detect whether the built executables exist
for t in ${TARGETS}; do
    exe="${BUILD_DIR}/examples/${t}"
    if [ ! -f "${exe}" ]; then
        echo -e "${RED}Error: ${exe} not found.${NC}" >&2
        exit 1
    fi
done

# Copy executables to repo root
for t in ${TARGETS}; do
    cp "${BUILD_DIR}/examples/${t}" .
done

echo -e "${BLUE}Essential binaries built: ${TARGETS}.${NC}"
