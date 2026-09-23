# PipePye Industrial Optimization Benchmark Suite

This directory contains the **PipePye Industrial Benchmark Suite**—a collection of authentic, structurally diverse linear and mixed-integer linear programming formulations grounded in real-world energy, refining, and manufacturing systems.

---

## Workload Catalog

| Case Directory | Problem Description | Class | Characteristic Structure | Primary Algorithmic Challenge |
| :--- | :--- | :---: | :--- | :--- |
| [`case_a_crude_blending/`](file:///home/sleepytiger/PIPEPYE/workloads/case_a_crude_blending/README.md) | Crude Oil Blending | **LP** | Dense quality rows, compact ($m \le 155, n \le 260$), density $9\% - 30\%$ | Dense LU factorization, exact vertex pivot tracking vs. gradient slowdown |
| [`case_b_multi_period_planning/`](file:///home/sleepytiger/PIPEPYE/workloads/case_b_multi_period_planning/README.md) | Multi-Period Production & Inventory Planning | **LP** | Block-angular / staircase sparse ($\text{Staircase} > 0.996$), scalable to $40\text{k NNZ}$ | Parallel GPU SpMV throughput vs. superlinear Simplex pivot scaling |
| [`case_c_refinery_scheduling/`](file:///home/sleepytiger/PIPEPYE/workloads/case_c_refinery_scheduling/README.md) | Refinery Unit Scheduling | **MILP** | $40\% - 43\%$ binary mode selectors, semicontinuous ranges, tank mass balances | Branch-and-bound tree search, dual simplex basis warm-starting |
| [`case_d_unit_commitment/`](file:///home/sleepytiger/PIPEPYE/workloads/case_d_unit_commitment/README.md) | Power System Unit Commitment & Economic Dispatch | **MILP** | Exactly $50\%$ binary commitment variables, dynamic ramping constraints, spinning reserves | Massive B&B search space ($2^{960}$ states), $> 90\%$ pivot reduction via warm-start |

---

## Standardized Package Structure

Each case directory conforms to a unified layout:
```text
workloads/case_<id>/
├── README.md              # Complete formulation, real-world context, literature provenance, and predictions
├── metadata.json          # Machine-readable dimensions, structural metrics, predictions, and verified references
├── model.mps              # Canonical baseline instance in standard MPS format
├── <Scale>.mps            # Ladder instances (Small, Medium, Large or T10, T25, T50, T100)
├── reference/
│   └── solution.json      # Independent reference solver (HiGHS-1.8.1) verified solutions and metrics
└── benchmark/
    └── results.csv        # Empirical solver timings, iterations, pivots, and outcomes
```

---

## Verification & Reproducibility
All instances are solved and independently verified against:
1. **PipePye Native Solvers**: Dual Revised Simplex, PDHG First-Order Solver, Branch-and-Bound Engine, and Presolve/Scaling Pipeline.
2. **Independent Reference Solver**: HiGHS 1.8.1 (world-class open-source simplex & MIP solver).
3. **Independent Solution Verifier**: Constraint violation, variable bound violation, and objective discrepancy checks.
