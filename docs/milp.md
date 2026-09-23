# Mixed-Integer Linear Programming (MILP) Architecture

PipePye's Phase 7 delivers a high-performance, sovereign **Mixed-Integer Linear Programming (MILP)** branch-and-bound engine. This document provides the mathematical foundation, algorithmic architecture, and software design of PipePye's MILP components.

---

## 1. Mathematical Problem Formulation

A Mixed-Integer Linear Program (MILP) in canonical bounded form is formulated as:

$$
\min_{x} \; c^T x + \text{offset}
$$

subject to:

$$
l_r \le A x \le u_r \quad (m \text{ constraints})
$$

$$
l_c \le x \le u_c \quad (n \text{ variables})
$$

$$
x_j \in \mathbb{Z} \quad \forall j \in \mathcal{I}
$$

$$
x_j \in \{0, 1\} \quad \forall j \in \mathcal{B} \subseteq \mathcal{I}
$$

where $\mathcal{I} \subseteq \{0, \dots, n-1\}$ indexes general integer variables, $\mathcal{B} \subseteq \mathcal{I}$ indexes binary variables, and variables in $\{0, \dots, n-1\} \setminus \mathcal{I}$ are continuous.

---

## 2. The MILP Solving Pipeline

PipePye executes a coordinated multi-stage pipeline:

```
  [MPS Input] 
       │
       ▼
  [Problem Analyzer] ──► Characterizes integrality ratio, sparsity, staircase block structure
       │
       ▼
  [Integer Root Presolve] ──► Bounds tightening: l_j = ceil(l_j), u_j = floor(u_j)
       │                  ──► Singleton row reductions
       ▼
  [Root Relaxation Solver] ──► Solves continuous LP via Sparse Dual Revised Simplex
       │
       ├──► [Gomory Cuts] ──► Generates GMI fractional cuts from tableau rows via BTRAN
       │
       ├──► [Primal Heuristics] ──► Simple Rounding & Fractional Diving for early incumbents
       │
       ▼
  [Branch-and-Bound Tree Search]
       ├── Queue Management: BestBound (min-heap), DepthFirst (stack), BestEstimate
       ├── Branching Variable: PseudoCost scoring or Most-Fractional selection
       ├── Child Node Evaluation: Dual Simplex warm-started with parent basis
       └── Incumbent & Gap Milestones: Records time to first incumbent, 10%, 1%, 0.1% gap
       │
       ▼
  [Solution Verification] ──► Independent verification of bounds, integrality, and rows
```

---

## 3. Mathematical Foundations of Key Components

### 3.1 Dual Revised Simplex Basis Warm-Starting
When branching on variable $x_k$ at value $x_k^* \notin \mathbb{Z}$, two child nodes are created:
- **Left branch**: $x_k \le \lfloor x_k^* \rfloor$ (new upper bound $u_k' = \lfloor x_k^* \rfloor$)
- **Right branch**: $x_k \ge \lceil x_k^* \rceil$ (new lower bound $l_k' = \lceil x_k^* \rceil$)

In the parent node, the basis $B$ was dual feasible. Modifying only the bound of variable $x_k$ preserves dual feasibility:
$$d_j = c_j - A_j^T y \ge 0 \quad (\forall j \text{ at lower bound})$$
$$d_j = c_j - A_j^T y \le 0 \quad (\forall j \text{ at upper bound})$$

The Dual Simplex algorithm can immediately restore primal feasibility using a small number of dual pivots (typically 1 to 5), bypassing the costly Phase I setup required by cold-start factorizations.

### 3.2 Gomory Mixed-Integer (GMI) Fractional Cuts
From the optimal simplex tableau, any basic integer variable $x_{B_i}$ with fractional value $x_{B_i}^*$ satisfies:

$$
x_{B_i} + \sum_{v \in N} \bar{a}_v x_v = 0
$$

where $\pi = (B^{-1})^T e_i$ is computed via **BTRAN** ($B^T \pi = e_i$), and $\bar{a}_v = \pi^T A_v^{aug}$.

For nonbasic variables $v \in N$ held at their lower or upper bounds, define the non-negative distance $\Delta x_v \ge 0$:
- If $v$ is at lower bound: $\Delta x_v = x_v - l_v$
- If $v$ is at upper bound: $\Delta x_v = u_v - x_v$

Let $f_0 = x_{B_i}^* - \lfloor x_{B_i}^* \rfloor \in (0, 1)$. The Gomory Mixed-Integer cut is:

$$
\sum_{v \in N} \gamma_v \Delta x_v \ge f_0 (1 - f_0)
$$

where the coefficient $\gamma_v$ is defined as:
- **If variable $v$ is integer**:
  $$\text{Let } f_v = \alpha_v - \lfloor \alpha_v \rfloor$$
  $$\gamma_v = \begin{cases} f_v (1 - f_0), & \text{if } f_v \le f_0 \\ (1 - f_v) f_0, & \text{if } f_v > f_0 \end{cases}$$
- **If variable $v$ is continuous**:
  $$\gamma_v = \begin{cases} \alpha_v (1 - f_0), & \text{if } \alpha_v \ge 0 \\ -\alpha_v f_0, & \text{if } \alpha_v < 0 \end{cases}$$

Slack variables $s_k = (A x)_k$ are substituted back into the original structural variables $x$, producing a normalized cut $C^T x \ge \text{RHS}$ that strictly separates the fractional relaxation point without cutting off any integer feasible solutions.

### 3.3 Pseudocost Branching
Pseudocost branching tracks the historical per-unit objective degradation observed when rounding fractional variables:

$$
q_j^- = \frac{\Delta z^-}{\lfloor x_j^* \rfloor - x_j^*}, \quad q_j^+ = \frac{\Delta z^+}{\lceil x_j^* \rceil - x_j^*}
$$

The composite score balances down-branch and up-branch degradations:

$$
\text{score}(j) = 0.8 \cdot \min(q_j^- f_j^-, q_j^+ f_j^+) + 0.2 \cdot \max(q_j^- f_j^-, q_j^+ f_j^+)
$$

Variables with larger scores are prioritized for branching, as they produce larger dual bound gains that trigger rapid subtree pruning.

---

## 4. Software API & Usage Example

### Solving a MILP Model

```cpp
#include <pipepye/milp/branch_and_bound.hpp>
#include <pipepye/model/mps_parser.hpp>
#include <iostream>

using namespace pipepye;
using namespace pipepye::milp;

int main() {
    // 1. Load MILP model
    auto lp = model::MPSParser::parse_file("refinery_scheduling.mps").value();

    // 2. Configure solver with advanced capabilities
    MILPConfig config = MILPConfig::Advanced();
    config.time_limit_sec = 60.0;
    config.mip_gap_tol = 1e-4; // 0.01% optimality gap

    // 3. Solve
    BranchAndBoundSolver solver(config);
    MILPResult result = solver.solve(lp);

    // 4. Output results
    std::cout << result.format_summary() << "\n";
    return 0;
}
```

### Measuring Warm vs. Cold Start Efficiency

```cpp
BranchAndBoundSolver solver;
auto [warm_res, cold_res] = solver.solve_warm_vs_cold(lp);

std::cout << "Warm Pivots: " << warm_res.warm_start_pivots << "\n"
          << "Cold Pivots: " << cold_res.cold_start_pivots << "\n"
          << "Pivot Reduction: " << (warm_res.pivot_reduction_ratio() * 100.0) << " %\n";
```

---

## 5. Verification Standards

Every candidate solution found during tree search or primal heuristics must pass three levels of verification:
1. **Integrality Invariant**: For every $j \in \mathcal{I}$, $|x_j - \text{round}(x_j)| \le 10^{-5}$.
2. **Primal Bound Feasibility**: $l_j - 10^{-6} \le x_j \le u_j + 10^{-6}$ for all $j \in \{0, \dots, n-1\}$.
3. **Constraint Feasibility**: $l_i^{row} - 10^{-5} \le (A x)_i \le u_i^{row} + 10^{-5}$ for all $i \in \{0, \dots, m-1\}$.
