# Case B: Multi-Period Production & Inventory Planning (LP)

## 1. Real-World Problem & Industrial Context
Multi-period production and inventory planning models the medium-to-long term scheduling of manufacturing capacity, raw materials, workforce shifts, intermediate product staging, and warehouse inventory across discrete operational periods $t \in \{1, \dots, T\}$. This formulation is ubiquitous in continuous process industries, petrochemical supply chains, assembly manufacturing, and enterprise Manufacturing Resource Planning (MRP) suites.

In these systems, decisions made in earlier periods ripple forward through time via inventory conservation: goods produced in period $t$ can satisfy demand in period $t$, or be stored in inventory to buffer against anticipated demand spikes, seasonal turnarounds, or plant maintenance shutdowns in period $t+1$. The objective is to minimize the sum of production costs (feedstocks, machine energy, labor overtime) and inventory carrying costs (storage, capital cost of working inventory, spoilage/degradation) across the entire operational horizon.

---

## 2. Literature Provenance & Source Formulation
The multi-period planning structure is one of the classic benchmarks in operations research and mathematical programming:
- **Foundational Formulation**: Manne, A. S. (1958). *"Programming of Economic Lot Sizes"*, Management Science, 4(2), 115–135.
- **Textbook Standard**: Johnson, L. A., & Montgomery, D. C. (1974). *Operations Research in Production Planning, Scheduling, and Inventory Control*. John Wiley & Sons.
- **Staircase LP Theoretical Foundation**: Fourer, R. (1982). *"Solving Staircase Linear Programs by the Simplex Method, 1: Inversion"*, Mathematical Programming, 23(1), 274–313.
- **Pricing in Staircase LPs**: Fourer, R. (1983). *"Solving Staircase Linear Programs by the Simplex Method, 2: Pricing"*, Mathematical Programming, 25(3), 251–292.
- **Public Domain Benchmark**: Netlib Staircase LP Suite (`SC205`, `SCAGR7`, `SCSD1`, `SCTAP1`).

---

## 3. Mathematical Formulation

### Sets & Indices
- $p \in \mathcal{P} = \{1, \dots, P\}$: Set of manufactured commodities or finished products.
- $m \in \mathcal{M} = \{1, \dots, M\}$: Set of shared processing units, reactors, or machine lines.
- $t \in \mathcal{T} = \{1, \dots, T\}$: Discrete time periods (e.g., shifts, days, weeks).

### Decision Variables
- $x_{p, t} \ge 0$: Production volume of product $p$ during period $t$.
- $s_{p, t} \ge 0$: Warehouse inventory volume of product $p$ held at the end of period $t$ and carried into $t+1$.

### Parameters
- $c_{p, t}$: Unit production cost of product $p$ in period $t$ (\$/unit).
- $h_{p, t}$: Unit holding/storage cost of product $p$ from period $t$ to $t+1$ (\$/unit).
- $D_{p, t}$: Customer delivery commitment / demand for product $p$ in period $t$.
- $A_{m, p}$: Capacity absorption rate (hours or machine units required per unit of product $p$ on unit $m$).
- $C_{m, t}$: Total available machine capacity of unit $m$ in period $t$.
- $S^{\max}_{p, t}$: Maximum allowable warehouse storage capacity for product $p$ in period $t$.
- $S^{\text{init}}_p$: Initial inventory available at the start of period 1 ($s_{p, 0}$).

### Objective Function
Minimize the discounted sum of production and inventory carrying expenses:
$$\min_{x, s \ge 0} \sum_{t=1}^T \sum_{p=1}^P \left( c_{p, t} x_{p, t} + h_{p, t} s_{p, t} \right)$$

### Constraints
1. **Dynamic Inventory Conservation Balances**:
   $$s_{p, t-1} + x_{p, t} - s_{p, t} = D_{p, t} \quad \forall p \in \mathcal{P}, t \in \mathcal{T}$$
   where $s_{p, 0} = S^{\text{init}}_p$ is a fixed boundary parameter.

2. **Resource & Unit Capacity Limits**:
   $$\sum_{p=1}^P A_{m, p} x_{p, t} \le C_{m, t} \quad \forall m \in \mathcal{M}, t \in \mathcal{T}$$

3. **Storage Warehouse Capacity Bounds**:
   $$0 \le s_{p, t} \le S^{\max}_{p, t} \quad \forall p \in \mathcal{P}, t \in \mathcal{T}$$

---

## 4. Assumptions & Simplifications in PipePye
1. **Deterministic Exogenous Demand**: Demand $D_{p, t}$ is assumed known with certainty across the entire planning window.
2. **Instantaneous Production Within Slices**: Products completed within period $t$ are available immediately to meet period $t$ demand or enter end-of-period storage $s_{p, t}$.
3. **Linear Cost Coefficients**: Costs are modeled linearly; step-function overtime premiums or convex quadratic holding penalties are linearized into upper-bounded variable increments.

---

## 5. Mathematical Structure & Pre-Registered Predictions

### Structural Classification
- **Near-Perfect Staircase Banded Matrix**: The matrix exhibits an authentic block-angular staircase structure with $\text{Staircase Score} > 0.996$ across all scales. Nonzero couplings across adjacent time steps occur strictly through the two nonzeros per inventory conservation balance ($+s_{p, t-1}$ and $-s_{p, t}$).
- **High Sparsity at Scale**: As the horizon expands from $T=10$ to $T=100$, nonzero density drops precipitously from $2.78\%$ to $0.21\%$.
- **High Gini Row Dispersion**: Inventory rows have exactly 3 nonzeros, while machine capacity rows have $P$ nonzeros, producing a distinct bimodal row degree distribution ($\text{Gini} \approx 0.30 - 0.47$).

### Pre-Registered Predictions
- **Small Horizons ($T \le 25$, $\text{NNZ} < 3,000$)**:
  - **Predicted Winner**: **Dual Revised Simplex (CPU)**.
  - **Rationale**: Small basis dimensions ($m \le 400$) invert rapidly using sparse LU factorization. First-order GPU algorithms suffer from memory allocation and kernel invocation latency.
- **Large Horizons ($T \ge 100$, $\text{NNZ} \ge 30,000$)**:
  - **Predicted Winner**: **PDHG First-Order Solver (CUDA GPU)**.
  - **Rationale**: The decoupled time blocks allow massive parallel thread occupancy during sparse matrix-vector multiplications ($Ax$ and $A^T y$). While Simplex pivot counts scale superlinearly with horizon length ($126 \to 333 \to 1,674 \to 3,428$ pivots), GPU PDHG executes iterations with uniform $O(\text{NNZ})$ parallel memory bandwidth, making it the preferred engine for large-scale production planning models.
