#!/usr/bin/env bash
set -euo pipefail

# Directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

BUILD_DIR="${REPO_ROOT}/build"
OUTPUT_NAME="profile_pipepye_probe"
OUTPUT_PATH="${REPO_ROOT}/${OUTPUT_NAME}"

echo "================================================================="
echo " Profiling PipePye CUDA Executable with NVIDIA Nsight Systems"
echo "================================================================="

if [ ! -f "${BUILD_DIR}/bin/pipepye_device_probe" ]; then
    echo "Error: Binary ${BUILD_DIR}/bin/pipepye_device_probe not found."
    echo "Please build the project first:"
    echo "  cmake -B build -G Ninja -DPIPEPYE_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86"
    echo "  ninja -C build"
    exit 1
fi

echo "[1/3] Executing Nsight Systems profile on ./build/bin/pipepye_device_probe..."
# Setting DEBUGINFOD_URLS="" prevents remote symbol download delays on Fedora Linux
env DEBUGINFOD_URLS="" nsys profile \
    --trace=cuda,nvtx,osrt \
    --stats=true \
    --output="${OUTPUT_PATH}" \
    --force-overwrite=true \
    "${BUILD_DIR}/bin/pipepye_device_probe"

echo ""
echo "[2/3] Profiling Complete!"
echo "  Artifact Report: ${OUTPUT_PATH}.nsys-rep"
echo "  SQLite Database: ${OUTPUT_PATH}.sqlite"
echo ""
echo "[3/3] Viewing Timeline in Nsight Systems GUI:"
echo "  Open NVIDIA Nsight Systems GUI, click 'File' -> 'Open', and select:"
echo "  ${OUTPUT_PATH}.nsys-rep"
echo "  You will see the timeline tracks: OS Runtime, NVTX Ranges, CUDA API, and GPU Kernels/Memory Copies."
echo "================================================================="
