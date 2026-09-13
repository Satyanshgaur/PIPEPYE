#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# PipePye CUDA SpMV & Reductions Profiler (NVIDIA Nsight Systems)
# Measures:
#  - Memory throughput & H2D / D2H transfer overheads
#  - Kernel launch latencies & stream synchronization
#  - Kernel execution duration across SpMV variants & reduction routines
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

BUILD_DIR="${REPO_ROOT}/build"
PROFILE_BIN="${BUILD_DIR}/bin/pipepye_bench_spmv_cuda"
OUTPUT_DIR="${REPO_ROOT}/benchmarks/profiles"
OUTPUT_NAME="spmv_nsys_profile"
OUTPUT_PATH="${OUTPUT_DIR}/${OUTPUT_NAME}"

mkdir -p "${OUTPUT_DIR}"

echo "================================================================================"
echo " PipePye SpMV & Reductions Profiling with NVIDIA Nsight Systems"
echo " Target Binary: ${PROFILE_BIN}"
echo " Output Report: ${OUTPUT_PATH}.nsys-rep"
echo "================================================================================"

if [ ! -f "${PROFILE_BIN}" ]; then
    echo "Error: Binary ${PROFILE_BIN} not found."
    echo "Building benchmarks first..."
    cmake --build "${BUILD_DIR}" --target pipepye_bench_spmv_cuda -j
fi

echo ""
echo "[Step 1/2] Executing High-Resolution Nsight Systems Trace..."
env DEBUGINFOD_URLS="" nsys profile \
    --trace=cuda,nvtx,osrt \
    --stats=true \
    --force-overwrite=true \
    --output="${OUTPUT_PATH}" \
    "${PROFILE_BIN}"

echo ""
echo "================================================================================"
echo "[Step 2/2] Profiling Report Successfully Generated!"
echo "  Artifact Report: ${OUTPUT_PATH}.nsys-rep"
echo "  SQLite Database: ${OUTPUT_PATH}.sqlite"
echo ""
echo "To visualize the timeline interactively in NVIDIA Nsight Systems GUI:"
echo "  nsys-ui ${OUTPUT_PATH}.nsys-rep"
echo "================================================================================"
