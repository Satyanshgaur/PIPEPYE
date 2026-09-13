India’s industrial optimization workloads depend on a small number of proprietary foreign solvers like CPLEX, Gurobi and Xpress, which introduce creating cost, licensing and dependency concerns. 
Existing open source solvers provide foundation, but fall short on large, sparse and difficult industrial optimization problems in complex MILPs. 
General-purpose solvers are also not specifically engineered or validated around relevant Indian industry.
There is also a lack of optimization engines that can intelligently determine how a workload should execute on CPU, GPU or a hybrid architecture for maximum throughput. 
Finally, there is limited publicly reproducible evidence showing how sovereign solvers perform on mathematical benchmarks and Indian industrial optimization models

We are building **PIPEPYE** to solve these gaps: a sovereign, high-performance optimization solver engineered and benchmarked for industrial optimization problems (LP, MILP, QP) across Indian infrastructure and manufacturing.

---

## Repository Structure

```text
pipepye/
├── CMakeLists.txt              # Root CMake configuration (C++20, CUDA sm_86, Ninja)
├── .gitignore                  # Git ignore rules for CMake, CUDA, profiler artifacts
├── .github/
│   └── workflows/
│       └── ci.yml              # GitHub Actions CI matrix (Ubuntu 24.04 CPU + CUDA sm_86)
├── include/
│   └── pipepye/
│       ├── core/               # Precision definitions (scalar_t, index_t), Status, Version
│       ├── cuda/               # CUDA_CHECK macros, CudaException, Device querying (RTX 3050)
│       └── utils/              # High-resolution CPUTimer, Logger, ScopedNvtxRange markers
├── src/                        # Core C++20 library implementations (libpipepye_core.a)
├── cuda/                       # CUDA kernels (DAXPY/SAXPY) & hardware probe (pipepye_device_probe)
├── tests/                      # GoogleTest suite run via CTest (69 unit and integration tests)
├── benchmarks/                 # CPU sparse & CUDA micro-benchmark harnesses
├── scripts/
│   └── profile.sh              # One-command NVIDIA Nsight Systems profiling script
└── docs/                       # Comprehensive project documentation
    ├── environment.md          # Hardware & toolchain specification (RTX 3050, GCC 16, CUDA 13.3)
    ├── build.md                # Build instructions & CMake options
    ├── profiling.md            # Nsight Systems workflow & timeline analysis
    ├── ci.md                   # Continuous integration pipeline details
    ├── algorithm-hardware-feasibility.md # Algorithm × hardware feasibility & LP architecture blueprint
    ├── mps-spec.md             # MPS parser specification & internal model mapping
    ├── benchmark.md            # CPU & GPU micro-benchmarks, SpMV scaling & bandwidth analysis
    └── tests.md                # Comprehensive test inventory (69 tests) & numerical verification
```

---

## Quick Start

### 1. Build the Project
```bash
cmake -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DPIPEPYE_ENABLE_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES=86

ninja -C build
```

### 2. Run the Test Suite
```bash
ctest --test-dir build --output-on-failure
```

### 3. Run Hardware Detection & CUDA Kernel Verification
```bash
./build/bin/pipepye_device_probe
```

### 4. Run CPU vs GPU Micro-Benchmark
```bash
./build/bin/pipepye_microbench_cuda
```

### 5. Profile with NVIDIA Nsight Systems
```bash
./scripts/profile.sh
```

---

## Detailed Documentation

- [Hardware & Software Environment](docs/environment.md)
- [Build & Configuration Guide](docs/build.md)
- [Profiling Workflow & Timeline](docs/profiling.md)
- [CI Pipeline Specification](docs/ci.md)
- [Algorithm × Hardware Feasibility & LP Architecture](docs/algorithm-hardware-feasibility.md)
- [MPS Ingestion Specification & Model Mapping](docs/mps-spec.md)
- [CPU & GPU Performance Benchmarking Report](docs/benchmark.md)
- [Comprehensive Test Verification Report](docs/tests.md)
