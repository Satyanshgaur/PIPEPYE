# Case D: Power System Unit Commitment & Economic Dispatch (MILP)

## Overview & Industrial Context

Unit commitment (UC) and economic dispatch (ED) form the computational backbone of wholesale electricity markets managed by Independent System Operators (ISOs) and Regional Transmission Organizations (RTOs) such as PJM, ERCOT, CAISO, and MISO.

The problem requires deciding which generating units (nuclear, coal, combined cycle, open cycle gas turbines, hydro) to turn on or off over a multi-period planning horizon (24 to 168 hours), and at what power output level to dispatch them, such that total operational cost is minimized while satisfying system-wide hourly demand, spinning reserve requirements, and physical generator technical limits (minimum generation, ramping rates, minimum run/down times).

Unit commitment is historically one of the most economically impactful mixed-integer linear programming applications in modern industry.

---

## Mathematical Formulation

Let:
- $\mathcal{G} = \{1, \dots, G\}$ denote the fleet of dispatchable generators.
- $\mathcal{T} = \{1, \dots, T\}$ denote the operational time periods (typically 1-hour intervals).

### Decision Variables
- $u_{g, t} \in \{0, 1\}$: Binary commitment status of generator $g$ at hour $t$ (1 if online, 0 if offline).
- $p_{g, t} \ge 0$: Power output dispatch (MW) of generator $g$ at hour $t$.

### Objective Function
Minimize the sum of no-load fixed operating costs, variable fuel/generation costs, and startup costs:
$$\min \sum_{t=1}^T \sum_{g=1}^G \left( C^{\text{fixed}}_g u_{g, t} + C^{\text{var}}_g p_{g, t} \right)$$

### Constraints
1. **System Power Balance**:
   $$\sum_{g=1}^G p_{g, t} \ge D_t \quad \forall t \in \mathcal{T}$$

2. **System Spinning Reserve Requirement**:
   $$\sum_{g=1}^G P^{\max}_g u_{g, t} \ge D_t + R_t \quad \forall t \in \mathcal{T}$$
   where $R_t$ is the operating reserve requirement to safeguard against unexpected generator trips.

3. **Generator Technical Operating Limits**:
   $$P^{\min}_g u_{g, t} \le p_{g, t} \le P^{\max}_g u_{g, t} \quad \forall g \in \mathcal{G}, t \in \mathcal{T}$$

4. **Dynamic Ramping Constraints**:
   Inter-temporal ramping limits between consecutive hours:
   $$p_{g, t} - p_{g, t-1} \le \text{RU}_g u_{g, t-1} + P^{\min}_g (u_{g, t} - u_{g, t-1}) \quad \forall g \in \mathcal{G}, t \in \{2, \dots, T\}$$
   $$p_{g, t-1} - p_{g, t} \le \text{RD}_g u_{g, t} + P^{\min}_g (u_{g, t-1} - u_{g, t}) \quad \forall g \in \mathcal{G}, t \in \{2, \dots, T\}$$

---

## Instance Ladder

The generator ladder spans small municipal grids up to large regional transmission operator subregions:

| Instance ID | Generators ($G$) | Horizon ($T$) | Rows ($m$) | Columns ($n$) | Binaries ($n_{\text{bin}}$) | Nonzeros ($\text{NNZ}$) | Density | Staircase Score | Gini Index |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **UNIT_COMMIT_Small** | 5 | 24 | 254 | 120 | 60 (50.0%) | 580 | 1.90% | 0.9934 | 0.112 |
| **UNIT_COMMIT_Medium** | 20 | 24 | 988 | 480 | 240 (50.0%) | 2,360 | 0.50% | 0.9984 | 0.155 |
| **UNIT_COMMIT_Large** | 40 | 48 | 3,896 | 1,920 | 960 (50.0%) | 9,520 | 0.13% | 0.9996 | 0.177 |

---

## Structural Characteristics

1. **High Combinatorial Dimension**: Exactly $50\%$ of all variables are binary commitment flags. For the Large instance, the state space encompasses $2^{960} \approx 10^{289}$ binary combinations.
2. **Coupled System Demand & Reserve Rows**: Unlike purely decoupled unit models, the hourly demand balance and reserve equations couple all active generators simultaneously.
3. **Temporal Dynamic Links**: Ramp rate bounds create tight sequential couplings across time slices, generating a pronounced block-banded matrix with staircase score $> 0.993$.
4. **Low Degree Dispersion**: Row degrees are homogeneous ($\text{Gini} \le 0.18$).

---

## Pre-Registered Hypothesis & Hardware Predictions

- **Integrality Requirement**: Due to $50\%$ binary variables, non-integer solutions are useless for operational scheduling. A rigorous Branch-and-Bound solver is mandatory.
- **Dual Simplex Warm-Start Dominance**: At each tree node, branching on generator commitment $u_{g, t} \in \{0, 1\}$ imposes a single-variable bound modification. Because bound tightenings maintain dual feasibility, warm-started Dual Simplex solves each child node in only 5 to 25 pivots, whereas solving from scratch requires complete Phase I/Phase II basis construction.
- **Predicted Winner**: **Branch-and-Bound with Warm-Started Dual Simplex (CPU)**.

---

## Empirical Benchmark Results

Measured across the Unit Commitment ladder:

| Instance | Tree Nodes | Warm Pivots | Cold Pivots | Pivots / Node (Warm) | Pivots / Node (Cold) | Pivot Reduction | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **UNIT_COMMIT_Small** | 49 | 320 | 2,735 | 6.53 | 55.82 | **88.30%** | **CONFIRMED** |
| **UNIT_COMMIT_Medium** | 500 | 4,209 | 110,541 | 8.42 | 221.08 | **96.19%** | **CONFIRMED** |
| **UNIT_COMMIT_Large** | 200 | 4,546 | 195,936 | 22.73 | 979.68 | **97.68%** | **CONFIRMED** |

### Key Observations
- On the large 40-generator, 48-hour model ($m=3,896, n=1,920$), cold simplex requires nearly **1,000 pivots per node**, totaling **195,936 pivots** across 200 nodes.
- Warm-started Dual Simplex resolves each child LP in an average of only **22.7 pivots**, completing 200 nodes in **4,546 total pivots** — a massive **97.68% pivot reduction** ($43.1\times$ fewer pivots).
- This confirms that tree exploration without basis warm-starting is computationally infeasible for realistic power system commitment models.
