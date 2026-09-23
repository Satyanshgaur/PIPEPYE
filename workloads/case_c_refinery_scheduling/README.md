# Case C: Refinery Unit Scheduling (MILP)

## 1. Real-World Problem & Industrial Context
Refinery unit scheduling coordinates short-term operational execution (typically 12-hour shifts or daily operational windows) across interconnected conversion and separation units:
- Atmospheric & Vacuum Crude Distillation Units (CDU / VDU)
- Fluid Catalytic Crackers (FCC)
- Hydrotreating & Hydrocracking Units (HTU / HCU)
- Catalytic Reformers, Isomerization, and Alkylation Units

Refinery units are highly flexible chemical reactors that operate in discrete, mutually exclusive operating modes (e.g., maximum diesel mode vs. maximum gasoline mode, sweet crude diet vs. sour crude diet, or mild vs. severe cracking). Each mode is characterized by:
- Semicontinuous operating ranges: a unit must either process throughput above a minimum hydraulic turndown limit ($F^{\min}$) up to its maximum hydraulic capacity ($F^{\max}$), or shut down entirely ($f = 0$).
- Mode-dependent product yield coefficients ($Y_{u, m, s}$).
- Variable utility costs, chemical catalyst consumption, and fixed shift setup/mode costs.

Intermediate buffer tanks decouple units and prevent downstream shutdowns during upstream mode shifts. The optimization problem decides the binary mode choices, continuous feed flow rates, and tank levels to minimize operational, feed, and inventory holding costs.

---

## 2. Literature Provenance & Source Formulation
The formulation implemented in PipePye is directly based on premier publications in chemical process systems engineering and mathematical programming:
- **Refinery Operational Planning**: Pinto, J. M., Joly, M., & Moro, L. F. (2000). *"Planning and Scheduling Models for Refinery Operations"*, Computers & Chemical Engineering, 24(9-10), 2259–2276.
- **Pipeline and Tank Transfer Optimization**: Moro, L. F. L., & Pinto, J. M. (1998). *"Mixed-Integer Optimization for the Scheduling of Pipeline Transfers in Petroleum Refineries"*, Computers & Chemical Engineering, 22, S725–S728.
- **Process Scheduling Survey**: Floudas, C. A., & Lin, X. (2004). *"Continuous-Time Versus Discrete-Time Approaches for Scheduling of Chemical Processes: a Review"*, Computers & Chemical Engineering, 28(11), 2109–2129.
- **Crude Scheduling Benchmarks**: Shah, N. (1996). *"Mathematical Programming Techniques for Crude Oil Scheduling"*, Computers & Chemical Engineering, 20, S1227–S1232.

---

## 3. Mathematical Formulation

### Sets & Indices
- $u \in \mathcal{U} = \{1, \dots, U\}$: Set of refinery conversion and separation processing units.
- $m \in \mathcal{M}_u = \{1, \dots, M_u\}$: Discrete operational operating modes for unit $u$.
- $s \in \mathcal{S} = \{1, \dots, S\}$: Intermediate and product storage tanks.
- $t \in \mathcal{T} = \{1, \dots, T\}$: Discrete planning time periods (operational shifts).

### Decision Variables
- $y_{u, m, t} \in \{0, 1\}$: Binary decision variable indicating whether unit $u$ operates in mode $m$ during shift $t$ ($1$) or is idle/operating in another mode ($0$).
- $f_{u, m, t} \ge 0$: Continuous feed flow rate processed by unit $u$ under mode $m$ during shift $t$ (bbl/shift).
- $v_{s, t} \ge 0$: Continuous volume of material stored in tank $s$ at the end of shift $t$ (bbl).

### Parameters
- $c^{\text{op}}_{u, m}$: Variable operating and energy cost per unit feed processed in mode $m$ (\$/bbl).
- $c^{\text{mode}}_{u, m}$: Fixed shift operating/startup cost incurred when unit $u$ is active in mode $m$ (\$/shift).
- $c^{\text{hold}}_s$: Inventory holding cost per unit volume in tank $s$ (\$/bbl/shift).
- $F^{\min}_{u, m}, F^{\max}_{u, m}$: Minimum turndown flow rate and maximum hydraulic processing capacity for unit $u$ in mode $m$.
- $Y_{u, m, s}$: Fractional volumetric yield of tank component $s$ produced per barrel of feed processed in unit $u$ under mode $m$.
- $C_{u, m, s}$: Fractional consumption rate of tank component $s$ consumed as feed to unit $u$ in mode $m$.
- $\text{Demand}_{s, t}$: External finished product withdrawal or downstream pipeline export requirement from tank $s$ in shift $t$.
- $V^{\min}_s, V^{\max}_s$: Minimum safety heel and maximum working storage capacity of tank $s$.

### Objective Function
Minimize the sum of variable operating costs, discrete mode activation expenses, and intermediate inventory holding costs:
$$\min \sum_{t \in \mathcal{T}} \left[ \sum_{u \in \mathcal{U}} \sum_{m \in \mathcal{M}_u} \left( c^{\text{op}}_{u, m} f_{u, m, t} + c^{\text{mode}}_{u, m} y_{u, m, t} \right) + \sum_{s \in \mathcal{S}} c^{\text{hold}}_s v_{s, t} \right]$$

### Constraints
1. **Mutually Exclusive Operating Modes**:
   $$\sum_{m \in \mathcal{M}_u} y_{u, m, t} \le 1 \quad \forall u \in \mathcal{U}, t \in \mathcal{T}$$

2. **Big-M Semicontinuous Operating Ranges**:
   $$f_{u, m, t} \ge F^{\min}_{u, m} y_{u, m, t} \quad \forall u \in \mathcal{U}, m \in \mathcal{M}_u, t \in \mathcal{T}$$
   $$f_{u, m, t} \le F^{\max}_{u, m} y_{u, m, t} \quad \forall u \in \mathcal{U}, m \in \mathcal{M}_u, t \in \mathcal{T}$$

3. **Dynamic Tank Material Balances**:
   $$v_{s, t-1} + \sum_{(u, m) \in \text{Prod}(s)} Y_{u, m, s} f_{u, m, t} - \sum_{(u, m) \in \text{Cons}(s)} C_{u, m, s} f_{u, m, t} - \text{Demand}_{s, t} = v_{s, t} \quad \forall s \in \mathcal{S}, t \in \mathcal{T}$$

4. **Physical Tank Capacity Limits**:
   $$V^{\min}_s \le v_{s, t} \le V^{\max}_s \quad \forall s \in \mathcal{S}, t \in \mathcal{T}$$

---

## 4. Assumptions & Simplifications in PipePye
1. **Discrete Uniform Time Grid**: Operational scheduling is discretized into uniform shifts ($t = 1, \dots, T$).
2. **Linear Yield Vectors**: Component yields $Y_{u, m, s}$ are represented as fixed fractional slates per mode, avoiding bilinear temperature/pressure non-convexities.
3. **Instantaneous Mode Switching**: Setup costs $c^{\text{mode}}_{u, m}$ penalize mode operations; mechanical cleaning and changeover times are absorbed within shift boundaries.

---

## 5. Mathematical Structure & Pre-Registered Predictions

### Structural Classification
- **High Integrality Fraction**: $40\% - 43\%$ of all decision columns are discrete binary variables ($y_{u, m, t}$).
- **Disjunctive Polyhedra**: The semicontinuous processing limits enforce disjunctive unions of $\{0\}$ and $[F^{\min}, F^{\max}]$, making the continuous LP relaxation strictly fractional.
- **Temporal Coupling**: Intermediate tank conservation links consecutive shifts into a staircase band ($\text{Staircase Score} > 0.997$).

### Pre-Registered Predictions
- **Algorithm Prediction**: **Branch-and-Bound with Dual Simplex Basis Warm-Starting**. Pure continuous solvers (Simplex alone or PDHG) cannot resolve binary decisions. At each B&B child node, branching on fractional mode indicators ($y_j \le 0$ or $y_j \ge 1$) preserves dual feasibility. A warm-started Dual Simplex solver resolves each child node in $< 5$ pivots, saving $> 90\%$ of total pivots compared to cold-starting from Phase I.
- **Hardware Prediction**: **CPU Execution**. Branch-and-bound search requires dynamic priority queues, irregular tree branching, and repeated sparse LU basis updates on small matrices, which are inherently irregular and non-coalesced, making CPU single-thread execution vastly superior to GPU SIMT execution.
