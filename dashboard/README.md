# PipePye Demonstration Web Application

A minimal, locally hosted demonstration frontend for the **PipePye C++/CUDA Optimization Engine**.

This dashboard allows users to select any `.mps` file directly from their Linux filesystem using the browser's native file picker, submit the formulation to the real compiled C++/CUDA solver pipeline, and examine technical diagnostics across all five phases of the project.

---

## Key Principles & Architecture

1. **Zero External Dependencies**:
   - Backend: Python 3 standard library (`http.server`, `urllib`, `subprocess`, `tempfile`). No `pip install`, no Flask, no node/npm bloat.
   - Frontend: Modern vanilla HTML5, CSS3 (slate dark mode), and JavaScript with native SVG chart rendering. No CDN dependencies, fully functional offline.

2. **Real Solver Pipeline Execution (No Mocks)**:
   - Directly executes `./build/bin/pipepye_dashboard_runner`.
   - Ingests the formulation with `pipepye::model::MPSParser`.
   - Executes Phase 1 sparse analysis (COO, CSR, CSC memory footprints, matrix density, kernel suitability).
   - Applies Phase 2 `ModelPipeline::prepare` (Presolve 5 reduction passes + Ruiz equilibration).
   - Computes Phase 5 structural moments (staircase score, row degree Gini index, half-bandwidth, degree statistics).
   - Emits structure-aware solver recommendation with algorithmic rationale.
   - Executes Phase 3 First-Order PDHG on CPU and CUDA GPU (recording exact residual convergence logs).
   - Executes Phase 4 Sparse Dual Revised Simplex (pivots, bound flips, LU refactorizations, PFI eta updates).
   - Executes Phase 4 PDHG $\to$ Simplex basis crossover.
   - Executes Phase 5 MILP Branch-and-Bound warm-start vs cold-start ablation (if discrete variables exist).
   - Audits solution feasibility directly on the original model via `SolutionVerifier`.
   - Evaluates empirical hypothesis confirmation (`CONFIRMED`, `PARTIALLY_CONFIRMED`, `REFUTED`).

---

## Quick Start Guide

### 1. Build the Pipeline Runner Binary
Ensure CMake and Ninja are configured with CUDA enabled:
```bash
cmake --build build --target pipepye_dashboard_runner -j$(nproc)
```

### 2. Launch the Local Demonstration Server
```bash
python3 dashboard/server.py --port 8080
```
Output:
```text
======================================================================
  PipePye Demonstration Web Server Running
  URL: http://127.0.0.1:8080
  Runner binary: /home/satyansh/pipepye/build/bin/pipepye_dashboard_runner
======================================================================
```

### 3. Open the Dashboard in Your Browser
Navigate to:
```text
http://127.0.0.1:8080
```

### 4. Run an Optimization Task
You can either:
- **Select any `.mps` file from your Linux filesystem** using the native file picker or drag-and-drop.
- **Select a built-in benchmark model** from the dropdown menu (e.g. Netlib `afiro.mps`, `beaconfd.mps`, or Phase 5 Industrial Workloads like Case A Crude Blending, Case B Multi-Period Planning, Case C Refinery Scheduling MILP, Case D Unit Commitment MILP).
- Click **Run PipePye Pipeline**.

---

## Technical Features Demonstrated

| Phase | Component | Key Metrics & Telemetry |
| :--- | :--- | :--- |
| **Phase 1** | Sparse Representation | $m, n, \text{NNZ}$, density, variable types, row senses, COO/CSR/CSC footprints, SpMV suitability |
| **Phase 2** | Model Preparation | Eliminated rows/cols/NNZ, Ruiz dynamic range compression ($\max |A_{ij}| / \min |A_{ij}|$), prep latency |
| **Phase 3** | PDHG Solvers | CPU vs CUDA GPU solve times, H2D/Kernel/D2H transfer breakdowns, real SVG residual convergence curves ($\log_{10}$) |
| **Phase 4** | Dual Revised Simplex | Exact pivots, bound flips, LU refactorizations, PFI eta updates, FTRAN/BTRAN counts, solve time |
| **Phase 4** | Basis Crossover | Active bounds detected, crashed basis partition (structural vs slack), simplex cleanup pivots, vertex recovery |
| **Phase 5** | Structure & Prediction | Staircase score, row Gini index, half-bandwidth, calibrated solver recommendation and rationale |
| **Phase 5** | MILP Branch-and-Bound | Dual warm-start vs cold-start tree exploration pivots, 88%–98% pivot reduction percentage |
| **Audit** | Solution Verification | Independent check against original unpresolved model constraints ($\|Ax - b\|_\infty$, primal bounds) |
