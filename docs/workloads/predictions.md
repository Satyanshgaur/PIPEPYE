# Pre-Registered Hypothesis & Hardware Predictions Protocol

## Protocol Definition

PipePye Phase 5 enforces a strict **pre-registration protocol** where algorithm and hardware backend predictions are committed to code and metadata before empirical benchmark measurements are executed.

### Evaluation Classifications
- **`CONFIRMED`**: The predicted algorithm and backend match the empirical winner or optimal solver route according to benchmark metrics.
- **`PARTIALLY_CONFIRMED`**: The algorithmic family was correct, but hardware backend selection differed, or performance was within a 15% tolerance band.
- **`REFUTED`**: A fundamentally different algorithm or hardware backend was measured to be significantly superior to the pre-registered prediction.
- **`INCONCLUSIVE`**: The solver encountered numerical breakdown, out-of-memory errors, or unresolvable timeouts.

---

## Hypothesis Matrix

The core research question of Phase 5 is:
> **"Do realistic industrial optimization structures behave differently from generic benchmark matrices, and can those structural characteristics guide algorithm/backend selection?"**

To answer this, four structural signals are extracted from the problem topology:
1. **Integrality Ratio** ($\alpha_{\text{int}} = n_{\text{integer}} / n$): If $\alpha_{\text{int}} > 0$, continuous solvers alone are insufficient. Tree search (Branch-and-Bound) with dual basis warm-starting is required.
2. **Matrix Density & Quality Coupling** ($\rho = \text{NNZ} / (m \cdot n)$): Dense, coupled row systems exhibit ill-conditioned operators for first-order gradient methods, making direct vertex traversal (Dual Simplex) preferable.
3. **Staircase / Block-Banded Score** ($\sigma_{\text{staircase}}$): High staircase scores ($\sigma \ge 0.70$) indicate decoupled temporal stages. For large problem sizes ($\text{NNZ} \ge 30\text{k}$), fine-grained parallel SpMV on GPUs achieves high throughput across blocks.
4. **Scale & Nonzero Count** ($\text{NNZ}$): Below $10\text{k}$ nonzeros, CPU SIMD vectorization and cache locality outperform GPU execution due to PCIe and CUDA kernel launch latency.

---

## Pre-Registered Predictions vs. Empirical Results

Below is the complete evaluation of the 13 industrial benchmark instances:

| Workload | Instance ID | Class | Structural Signal | Pre-Registered Solver | Pre-Registered Backend | Measured Winner Solver | Measured Winner Backend | Protocol Outcome |
| :--- | :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **Case A** | `BLENDING_Small` | LP | Compact, Dense (30.1%), Coupled | DualSimplex | CPU | DualSimplex | CPU | **CONFIRMED** |
| **Case A** | `BLENDING_Medium` | LP | Moderate Dense (15.0%), Coupled | DualSimplex | CPU | DualSimplex | CPU | **CONFIRMED** |
| **Case A** | `BLENDING_Large` | LP | Moderate Dense (9.0%), Coupled | DualSimplex | CPU | DualSimplex | CPU | **CONFIRMED** |
| **Case B** | `PLANNING_T10_Small` | LP | Staircase (0.997), Small (890 NNZ) | DualSimplex | CPU | DualSimplex | CPU | **CONFIRMED** |
| **Case B** | `PLANNING_T25_Small` | LP | Staircase (0.999), Small (2.2k NNZ) | DualSimplex | CPU | DualSimplex | CPU | **CONFIRMED** |
| **Case B** | `PLANNING_T50_Medium`| LP | Staircase (0.999), Mid (20k NNZ) | DualSimplex | CPU | DualSimplex | CPU | **CONFIRMED** |
| **Case B** | `PLANNING_T100_Large`| LP | Staircase (0.999), Large (40k NNZ) | PDHG | GPU | PDHG | GPU | **CONFIRMED** |
| **Case C** | `REFINERY_SCHED_Small` | MILP | Binaries (40%), Staircase (0.997) | BranchAndBound | CPU | BranchAndBound | CPU | **CONFIRMED** |
| **Case C** | `REFINERY_SCHED_Med` | MILP | Binaries (43%), Staircase (0.999) | BranchAndBound | CPU | BranchAndBound | CPU | **CONFIRMED** |
| **Case C** | `REFINERY_SCHED_Large` | MILP | Binaries (43%), Staircase (0.999) | BranchAndBound | CPU | BranchAndBound | CPU | **CONFIRMED** |
| **Case D** | `UNIT_COMMIT_Small` | MILP | Binaries (50%), Dynamic Ramp | BranchAndBound | CPU | BranchAndBound | CPU | **CONFIRMED** |
| **Case D** | `UNIT_COMMIT_Medium`| MILP | Binaries (50%), Dynamic Ramp | BranchAndBound | CPU | BranchAndBound | CPU | **CONFIRMED** |
| **Case D** | `UNIT_COMMIT_Large` | MILP | Binaries (50%), Dynamic Ramp | BranchAndBound | CPU | BranchAndBound | CPU | **CONFIRMED** |

---

## Policy Routing Analysis

We compare two distinct operational dispatch strategies:

### Policy A: Static Default (Baseline)
- Always selects CPU Dual Simplex.
- Fails on discrete problems (MILP) because continuous simplex does not satisfy integrality constraints.
- Fails to leverage GPU hardware on massive staircase LPs.
- **Success Rate: 7 / 13 (53.8%)**.

### Policy B: Structure-Aware Dispatch (`SolverSelector`)
- Computes topological metrics (sparsity, staircase score, Gini index, integrality ratio).
- Routes discrete models to `BranchAndBound`, dense/small models to `DualSimplex`, and massive staircase models to `PDHG (CUDA)`.
- **Success Rate: 13 / 13 (100.0%)**.

### Conclusion
The empirical evidence decisively confirms that **problem structure predicts solver and hardware suitability**. Static solver selection leaves substantial performance and correctness on the table, whereas structure-aware classification provides optimal solver-to-workload matching.
