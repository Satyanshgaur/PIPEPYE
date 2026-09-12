# Build & Execution Guide

This guide provides reproducible, exact instructions for building and verifying **PipePye** across CPU and CUDA targets.

---

## 1. Prerequisites

Ensure the following tools are installed and accessible in your `PATH`:
- **C++ Compiler**: GCC 14+ or Clang 18+ (C++20 compliant)
- **CUDA Toolkit**: CUDA 12.0+ (Tested on CUDA 13.3)
- **CMake**: 3.24 or newer (Tested on CMake 4.3.0)
- **Build System**: Ninja (`ninja-build`) or GNU Make
- **GPU**: NVIDIA RTX 3050 Laptop GPU (Compute Capability 8.6 / Ampere) or compatible NVIDIA GPU

---

## 2. Quick Start Build (CUDA sm_86 + Release)

To configure and compile the entire project (libraries, executables, benchmarks, and tests) targeting the NVIDIA RTX 3050:

```bash
# 1. Create build directory and configure with CMake & Ninja
cmake -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DPIPEPYE_ENABLE_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES=86 \
    -DPIPEPYE_BUILD_TESTS=ON \
    -DPIPEPYE_BUILD_BENCHMARKS=ON

# 2. Compile all targets
ninja -C build

# 3. Run all unit tests via CTest
ctest --test-dir build --output-on-failure
```

All compiled binaries will reside in `build/bin/`:
- `build/bin/pipepye_device_probe`: Hardware probe and CUDA kernel verification executable.
- `build/bin/test_core`: Core C++20 unit tests (types, version, status, timer).
- `build/bin/test_cuda`: CUDA unit tests (error handling, memory, kernel correctness).
- `build/bin/pipepye_microbench_cuda`: CPU vs CUDA DAXPY micro-benchmark harness.

---

## 3. Running the CUDA Device Probe

Run the Phase 0 hardware probe and verification program:

```bash
./build/bin/pipepye_device_probe
```

### Expected Output
1. Detects `NVIDIA GeForce RTX 3050 6GB Laptop GPU`.
2. Confirms Compute Capability `8.6 (Ampere sm_86 - native)`.
3. Displays VRAM (5.67 GiB), SM count (20 SMs), clock rate, and bus width.
4. Executes a 4-million element FP64 DAXPY kernel.
5. Verifies 100% agreement against CPU double-precision reference (`diff <= 1e-12`).
6. Reports kernel memory bandwidth (>100 GB/s) and execution breakdown.

---

## 4. Running the Micro-Benchmark

To observe the CPU vs GPU performance crossover across different problem sizes:

```bash
./build/bin/pipepye_microbench_cuda
```

This will run sweeps across $10^4, 10^5, 10^6, 5 \times 10^6, 10^7$ double-precision vector operations and display CPU latency, GPU kernel latency, H2D/D2H transfer latency, achieved bandwidth, and speedup factors.

---

## 5. Alternative Build Configurations

### CPU-Only Build (for environments without GPU or headless CI)
```bash
cmake -B build_cpu -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DPIPEPYE_ENABLE_CUDA=OFF \
    -DPIPEPYE_BUILD_TESTS=ON \
    -DPIPEPYE_BUILD_BENCHMARKS=OFF

ninja -C build_cpu
ctest --test-dir build_cpu --output-on-failure
```

### Debug Build with AddressSanitizer (Host Code)
```bash
cmake -B build_debug -G Ninja \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -g" \
    -DPIPEPYE_ENABLE_CUDA=ON

ninja -C build_debug
```
