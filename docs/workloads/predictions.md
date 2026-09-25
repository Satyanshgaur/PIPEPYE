# Structure-Aware Hardware Predictions & Policy Dispatch Protocol

## Protocol Definition & Methodological Framing

PipePye's hardware and algorithm dispatch system investigates how topological matrix structures dictate optimal execution paths across CPU and GPU hardware backends.

### Methodological Scope: In-Sample Calibration vs. True Pre-Registration
To maintain absolute scientific transparency:
- **In-Sample Rule Calibration:** The deterministic dispatch thresholds embedded in PipePye's policy (`NNZ >= 30,000`, `density > 0.08`, `staircase_score >= 0.70`) were directly parameterized from the microbenchmark crossover boundaries observed in Experiments 1 and 2. Consequently, evaluating the 13 industrial instances represents *in-sample rule calibration*, not a blind trial. On an $n=13$ sample where rules are parameterized against the observed dataset, 100% accuracy is mathematically expected and carries minimal inferential weight on its own.
- **Out-of-Sample Held-Out Evaluation:** To verify whether these calibrated rules generalize beyond the 13 training instances, the dispatch engine was evaluated on a held-out suite of **19 external public benchmark models** (12 Netlib LP instances and 7 MIPLIB 3 combinatorial instances).
- **Non-Strawman Baseline Evaluation:** Comparing against a monolithic "Always CPU Simplex" baseline (53.8%) is a strawman because continuous simplex naturally fails on MILP models due to lack of integrality support. We therefore evaluate two realistic, class-aware baselines alongside the structure-aware engine.

### Evaluation Classifications
- **`CONFIRMED`**: The predicted algorithm and backend match the empirical winner or optimal solver route according to benchmark metrics.
- **`PARTIALLY_CONFIRMED`**: The algorithmic family was correct, but hardware backend selection differed, or performance was within a 15% tolerance band.
- **`REFUTED`**: A fundamentally different algorithm or hardware backend was measured to be significantly superior to the predicted route.
- **`INCONCLUSIVE`**: The solver encountered numerical breakdown, out-of-memory errors, or unresolvable timeouts.

---

## Hypothesis Matrix & Structural Signals

The core research question is:
> **"Do realistic industrial optimization structures behave differently from generic benchmark matrices, and can those structural characteristics guide algorithm/backend selection to prevent computational blowups?"**

To answer this, four structural signals are extracted from the problem topology:
1. **Integrality Ratio** ($\alpha_{\text{int}} = n_{\text{integer}} / n$): If $\alpha_{\text{int}} > 0$, continuous solvers alone are insufficient. Tree search (Branch-and-Bound) with dual basis warm-starting is required.
2. **Matrix Density & Quality Coupling** ($\rho = \text{NNZ} / (m \cdot n)$): Dense, coupled row systems exhibit ill-conditioned operators for first-order gradient methods, making direct vertex traversal (Dual Simplex) preferable.
3. **Staircase / Block-Banded Score** ($\sigma_{\text{staircase}}$): High staircase scores ($\sigma \ge 0.70$) indicate decoupled temporal stages. For large problem sizes ($\text{NNZ} \ge 30\text{k}$), fine-grained parallel SpMV on GPUs achieves high throughput across blocks ($204.3\times$ speedup at $T=100$).
4. **Scale & Nonzero Count** ($\text{NNZ}$): Below $15\text{k}$ nonzeros, CPU SIMD vectorization and cache locality outperform GPU execution due to PCIe and CUDA kernel launch latency (CPU 1-thread is up to $14.1\times$ faster).

---

## Calibrated Policy Routes vs. Empirical Results (13 Industrial Workloads)

Below is the complete evaluation of the 13 industrial benchmark instances under the calibrated policy:

| Workload | Instance ID | Class | Structural Signal | Calibrated Solver | Calibrated Backend | Measured Winner Solver | Measured Winner Backend | Evaluation Outcome |
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

## Policy Routing Analysis & Baseline Comparisons

To dismantle misleading strawman baselines, we evaluate three distinct operational dispatch strategies:

### 1. Class-Aware Monolithic CPU Baseline (Standard SOTA Practice)
- Routes all MILP models to CPU Branch-and-Bound.
- Routes all continuous LP models to CPU Dual Simplex.
- **Overall Industrial Accuracy:** **12 / 13 (92.3%)**.
- **Continuous LP Sub-Suite ($n=7$):** **6 / 7 (85.7%)**.
- **Failure Mode & Latency Penalty:** Fails on `PLANNING_T100`, where CPU Simplex takes 40,039 ms vs. GPU PDHG 196 ms—a **$204.3\times$ latency blowup**.

### 2. Class-Aware Monolithic GPU Baseline (Naive GPU-First Practice)
- Routes all MILP models to CPU Branch-and-Bound.
- Routes all continuous LP models to CUDA GPU PDHG.
- **Overall Industrial Accuracy:** **7 / 13 (53.8%)**.
- **Continuous LP Sub-Suite ($n=7$):** **1 / 7 (14.3%)**.
- **Failure Mode & Latency Penalty:** Fails on 6 of 7 continuous LPs. Dense blending models and small planning horizons run **up to $200\times$ slower** on GPU due to PCIe transfer, kernel launch latency, and slow first-order gradient convergence on ill-conditioned equations.

### 3. Structure-Aware Adaptive Policy (PipePye Decision Engine)
- Extracts structural features: integrality ratio, sparsity density, staircase score, and nonzero scale.
- Routes MILP $\to$ B&B, compact/dense LPs $\to$ CPU Dual Simplex, and massive staircase LPs $\to$ CUDA GPU PDHG.
- **Overall Industrial Accuracy:** **13 / 13 (100.0% calibrated)**.
- **Continuous LP Sub-Suite ($n=7$):** **7 / 7 (100.0% calibrated)**.
- **Advantage:** Unlocks the $204\times$ speedup on large staircase models while preventing GPU performance degradation on compact and dense models.

---

## Held-Out Generalization Evaluation (19 Public Benchmark Instances)

To test generalization out-of-sample on external reference instances with known global optima:

### 1. Netlib Linear Programming Suite (12 Held-Out Instances)
- Models: `afiro`, `adlittle`, `blend`, `sc50a`, `sc50b`, `kb2`, `share2b`, `stocfor1`, `lotfi`, `beaconfd`, `e226`, `bandm`.
- **Topological Characteristics:** All 12 instances have $\text{NNZ} < 3,500$.
- **Policy Decision:** **12 / 12 (100.0%) dispatched to CPU Dual Simplex**, correctly avoiding GPU launch overhead.
- **Empirical Validation:** Prepared Simplex solves 12/12 to certified optimality (zero KKT violations), outperforming or matching HiGHS 1.15.1 on 8 of 12 instances.

### 2. MIPLIB 3 / Mittelmann Combinatorial Suite (7 Held-Out Instances)
- Models: `p0033`, `flugpl`, `stein27`, `egout`, `mod008`, `bell3a`, `mas74`.
- **Topological Characteristics:** Integrality ratio $\alpha_{\text{int}} > 0$.
- **Policy Decision:** **7 / 7 (100.0%) dispatched to CPU Branch-and-Bound**.
- **Generalization Boundary:** While class-aware routing succeeds, structural metrics alone cannot predict combinatorial tree stagnation from missing polyhedral cutting planes (e.g. `egout.mps`), isolating root cut separation as the primary frontier for sovereign MILP scaling.
