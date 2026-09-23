# Case A: Refinery Crude Blending (LP)

## Overview & Industrial Context

Crude oil blending is a core daily planning operation in downstream petroleum refining. Refineries procure diverse crude feedstocks with varying chemical assays (API gravity, sulfur percentage, Reid vapor pressure, viscosity, octane index) and blend them continuously to feed distillation units and meet rigorous commercial product specifications for gasoline, jet fuel, diesel, and residual fuel oil.

The linear blending formulation models volumetric conservation and linear blend indices for product qualities. It serves as an archetypal **densely coupled, compact linear program**.

---

## Mathematical Formulation

Let:
- $\mathcal{C} = \{1, \dots, C\}$ denote the set of crude feedstocks.
- $\mathcal{P} = \{1, \dots, P\}$ denote the set of refined products.
- $\mathcal{Q} = \{1, \dots, Q\}$ denote the set of chemical quality specifications.

### Decision Variables
- $x_{c, p} \ge 0$: Volume of crude feedstock $c \in \mathcal{C}$ allocated to product $p \in \mathcal{P}$.

### Objective Function
Minimize net procurement cost minus product sales revenue:
$$\min_{x \ge 0} \sum_{c \in \mathcal{C}} \sum_{p \in \mathcal{P}} \left( \text{cost}_c - \text{price}_p \right) x_{c, p}$$

### Constraints
1. **Crude Feedstock Availability**:
   $$\sum_{p \in \mathcal{P}} x_{c, p} \le S_c \quad \forall c \in \mathcal{C}$$

2. **Product Demand Range**:
   $$D^{\min}_p \le \sum_{c \in \mathcal{C}} x_{c, p} \le D^{\max}_p \quad \forall p \in \mathcal{P}$$

3. **Product Quality Attribute Bounds**:
   For each quality attribute $q \in \mathcal{Q}$ with crude assay value $A_{c, q}$ and allowable product limits $[Q^{\min}_{p, q}, Q^{\max}_{p, q}]$:
   $$\sum_{c \in \mathcal{C}} \left( A_{c, q} - Q^{\max}_{p, q} \right) x_{c, p} \le 0 \quad \forall p \in \mathcal{P}, q \in \mathcal{Q}$$
   $$\sum_{c \in \mathcal{C}} \left( Q^{\min}_{p, q} - A_{c, q} \right) x_{c, p} \le 0 \quad \forall p \in \mathcal{P}, q \in \mathcal{Q}$$

---

## Instance Ladder

The workload generator parameterizes instances deterministically using a fixed pseudo-random seed to generate authentic crude assay distributions:

| Scale | Crudes ($C$) | Products ($P$) | Qualities ($Q$) | Rows ($m$) | Columns ($n$) | Nonzeros ($\text{NNZ}$) | Density | Gini Index | Staircase Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Small** | 6 | 3 | 3 | 26 | 18 | 141 | 30.13% | 0.086 | 0.306 |
| **Medium** | 13 | 6 | 4 | 66 | 78 | 774 | 15.04% | 0.089 | 0.193 |
| **Large** | 26 | 10 | 6 | 155 | 260 | 3,630 | 9.01% | 0.092 | 0.128 |

---

## Structural Characteristics

1. **High Matrix Density**: Unlike network flow or scheduling matrices which typically exhibit $< 0.1\%$ nonzero density, crude blending matrices have $9\% - 30\%$ density due to dense quality equations where every crude enters every quality constraint for that product.
2. **Dense Cross-Coupling**: Column operations affect multiple shared quality constraints simultaneously, leading to high condition numbers for first-order gradient methods.
3. **Compact Dimension**: Problem dimensions are modest (tens to hundreds of variables), characteristic of real-time blending control blocks.
4. **Low Gini Index**: Row and column degree distributions are near-uniform ($\text{Gini} \approx 0.08 - 0.09$), showing no hub or power-law structure.

---

## Pre-Registered Hypothesis & Hardware Predictions

- **Hypothesis**: The compact dimensionality and dense coupling make crude blending problems ideal for **CPU Dual Revised Simplex**. The basis matrix $B$ is small ($m \le 155$) and factorizes rapidly via LU decomposition. Pivot steps directly trace vertices to find exact active quality bounds.
- **Counter-Prediction**: First-order methods like **PDHG** will struggle due to slow convergence on coupled quality constraints (requiring thousands of steps) and excessive GPU kernel launch/transfer overhead relative to problem size.
- **Predicted Winner**: **Dual Simplex (CPU)**.

---

## Empirical Benchmark Results

Benchmarked using [`benchmarks/bench_industrial.cpp`](file:///home/sleepytiger/PIPEPYE/benchmarks/bench_industrial.cpp) on an AMD Ryzen 9 7900X (CPU) and NVIDIA GeForce RTX 3080 (GPU):

| Instance | Simplex Time | Simplex Pivots | PDHG CPU Time | PDHG CPU Iters | PDHG GPU Time | PDHG GPU Iters | Winner | Prediction Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **BLENDING_Small** | **< 0.01 ms** | 23 | 2.16 ms | 3,000 | 236.19 ms | 3,000 | **DualSimplex (CPU)** | **CONFIRMED** |
| **BLENDING_Medium** | **< 0.01 ms** | 71 | 7.49 ms | 3,000 | 61.10 ms | 3,000 | **DualSimplex (CPU)** | **CONFIRMED** |
| **BLENDING_Large** | **< 0.01 ms** | 90 | 12.37 ms | 3,000 | 58.46 ms | 3,000 | **DualSimplex (CPU)** | **CONFIRMED** |

### Key Observations
- Dual Simplex terminates in fewer than 100 pivots with exact vertex precision in sub-millisecond execution time.
- PDHG on CPU consumes 2.16–12.37 ms to reach tolerance, throttled by the dense quality couplings.
- PDHG on GPU is heavily bottlenecked by GPU initialization, small thread block allocations, and PCI-e transfer costs, executing up to orders of magnitude slower.
- The pre-registered prediction **Dual Simplex (CPU)** is **100% CONFIRMED**.
