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
├── tests/                      # GoogleTest suite run via CTest (120 unit and integration tests)
├── benchmarks/                 # CPU sparse, CUDA SpMV, scaling, and numerical robustness benchmarks
├── tools/                      # CLI utilities (pipepye_inspect unified model analyzer)
├── scripts/
│   └── profile.sh              # One-command NVIDIA Nsight Systems profiling script
└── docs/                       # Comprehensive project documentation
    ├── architecture.md         # Master optimization solver architecture & development roadmap
    ├── phase1summary.md        # Phase 1 sparse numerical core empirical findings & implications
    ├── phase2summary.md        # Phase 2 presolve, scaling & characterization findings & implications
    ├── presolve.md             # Modular presolve pipeline, 5 reduction passes & postsolve reconstruction
    ├── scaling.md              # Ruiz equilibration, Pock-Chambolle scaling & unscaling
    ├── characterization.md     # Problem analyzer, topological moments, Gini & conditioning proxies
    ├── numerical_robustness.md # First-order PDHG downstream solver robustness experiment
    ├── benchmark.md            # CPU/GPU micro-benchmarks, SpMV scaling & bandwidth analysis
    ├── tests.md                # Comprehensive test inventory (120 tests) & numerical verification
    ├── algorithm-hardware-feasibility.md # Algorithm × hardware feasibility & LP architecture blueprint
    ├── mps-spec.md             # MPS parser specification & internal model mapping
    ├── environment.md          # Hardware & toolchain specification (RTX 3050, GCC 16, CUDA 13.3)
    ├── build.md                # Build instructions & CMake options
    ├── profiling.md            # Nsight Systems workflow & timeline analysis
    └── ci.md                   # Continuous integration pipeline details
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

### 2. Run the Full Test Suite
```bash
ctest --test-dir build --output-on-failure
```

### 3. Inspect LP Models (Single-Line Banner or Full Report)
```bash
# Print canonical one-line dispatch summary:
./build/bin/pipepye_inspect tests/data/mps/netlib/beaconfd.mps --one-line

# Full before/after Phase 2 pipeline comparison:
./build/bin/pipepye_inspect tests/data/mps/netlib/beaconfd.mps --before-after
```

### 4. Run Automated Before/After Presolve & Scaling Benchmark
```bash
./build/bin/pipepye_bench_presolve_scaling
```

### 5. Run Downstream Numerical Robustness Experiment (PDHG Simulation)
```bash
./build/bin/pipepye_bench_numerical_robustness
```

### 6. Run Hardware Detection & CUDA Micro-Benchmarks
```bash
./build/bin/pipepye_device_probe
./build/bin/pipepye_microbench_cuda
```

### 7. Profile with NVIDIA Nsight Systems
```bash
./scripts/profile.sh
```

---

## Detailed Documentation

- **Architecture & Summaries**:
  - [Architecture & Development Roadmap](docs/architecture.md)
  - [Phase 1 Summary & Findings](docs/phase1summary.md)
  - [Phase 2 Summary & Preconditioning Findings](docs/phase2summary.md)
- **Phase 2 Pipeline & Algorithms**:
  - [Presolve Pipeline Architecture](docs/presolve.md)
  - [Matrix Scaling & Equilibration](docs/scaling.md)
  - [Problem Characterization Layer](docs/characterization.md)
  - [Numerical Robustness Experiment](docs/numerical_robustness.md)
- **Benchmarks & Numerical Verification**:
  - [CPU & GPU Performance Benchmarking Report](docs/benchmark.md)
  - [Comprehensive Test Verification Report (120 Tests)](docs/tests.md)
  - [Algorithm × Hardware Feasibility & LP Architecture](docs/algorithm-hardware-feasibility.md)
  - [MPS Ingestion Specification & Model Mapping](docs/mps-spec.md)
- **Environment & Engineering**:
  - [Hardware & Software Environment](docs/environment.md)
  - [Build & Configuration Guide](docs/build.md)
  - [Profiling Workflow & Timeline](docs/profiling.md)
  - [CI Pipeline Specification](docs/ci.md)
