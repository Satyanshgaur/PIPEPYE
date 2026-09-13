# Reproducible Development Environment

This document records the exact development environment, compiler toolchains, CUDA runtime, and hardware configurations for the **PipePye** sovereign optimization solver.

---

## 1. Hardware Specification

| Component | Detail | Notes |
|---|---|---|
| **Host System** | x86_64 Architecture, Linux | Fedora 44 (Kernel 7.1.13-200.fc44.x86_64) |
| **GPU Model** | NVIDIA GeForce RTX 3050 6GB Laptop GPU | GA107 Ampere Architecture |
| **CUDA Compute Capability** | `sm_86` | Ampere generation (8.6) |
| **Streaming Multiprocessors (SMs)** | 16 SMs | ~2048 FP32 CUDA cores, 64 Tensor cores |
| **Global Memory (VRAM)** | 6144 MiB (6.0 GiB GDDR6) | Dedicated device memory |
| **Memory Bus Width** | 96-bit | High-efficiency mobile memory bus |
| **L2 Cache Size** | 2048 KiB (2 MiB) | Ultra-low latency on-chip cache |
| **Warp Size** | 32 threads | Standard hardware SIMT scheduling unit |
| **Max Threads per Block** | 1024 threads | Optimal block sizing: 128 / 256 / 512 |
| **NVIDIA Driver Version** | `610.57.04` | Kernel Module Driver (KMD) & User Module Driver (UMD) |

---

## 2. Software & Toolchain Versions

| Tool | Version | Executable / Verification Command |
|---|---|---|
| **C++ Standard** | C++20 (`-std=c++20`) | Enforced in root `CMakeLists.txt` |
| **Host C++ Compiler** | GCC 16.2.1 20260819 (Red Hat 16.2.1-2) | `g++ --version` |
| **Alternative Compiler** | Clang 22.1.8 (Fedora 22.1.8-4.fc44) | `clang++ --version` |
| **CUDA Compiler (NVCC)** | CUDA 13.3 (V13.3.73, Build cuda_13.3.r13.3/compiler.38244171_0) | `nvcc --version` |
| **Build System** | CMake 4.3.0 | `cmake --version` |
| **Build Generator** | Ninja 1.12+ / GNU Make 4.4+ | `ninja --version` |
| **Testing Framework** | GoogleTest v1.15.2 | Integrated via CMake `FetchContent` |
| **Profiler** | NVIDIA Nsight Systems 2026.1.3 | `/usr/local/cuda/bin/nsys --version` |
| **Instrumentation** | NVIDIA NVTX v3 | Header-only (`<nvtx3/nvtx3.hpp>`) |

---

## 3. Dependency Architecture & Technological Sovereignty
 
PipePye is engineered **from scratch** to uphold absolute technological sovereignty, algorithmic transparency, and auditability:
- **Zero External Solver Dependencies**: PipePye contains no linkage to third-party solvers (GLPK, COIN-OR, HiGHS, SuiteSparse) or proprietary engines (CPLEX, Gurobi, Xpress). All core sparse linear algebra, matrix representations, vector primitives, and solver algorithms are implemented natively from first principles.
- **System-Provided Profiling**: NVIDIA NVTX v3 provided directly by the CUDA Toolkit (`<nvtx3/nvtx3.hpp>`).
- **Hermetic Unit & Benchmark Testing**: GoogleTest fetched and compiled from pinned Git tags (`v1.15.2`).
