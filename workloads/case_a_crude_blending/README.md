# Case A: Crude Oil Blending (LP)

## 1. Real-World Problem & Industrial Context
Crude oil blending is a mission-critical operational scheduling task executed daily in downstream petroleum refineries. Refineries receive shipments of crude oils from diverse geological reservoirs across the globe—such as light sweet crudes (Brent, Bonny Light), heavy sour crudes (Arab Heavy, Maya), and intermediate synthetic crudes. Each crude feed exhibits vastly different chemical assays, characterized by API gravity, sulfur weight percentage, Reid vapor pressure (RVP), viscosity, and distillation fractions (light naphtha, middle distillates, vacuum gas oil).

To satisfy environmental mandates (e.g., Euro-VI / Bharat Stage VI clean fuel standards with sulfur < 10 ppm) and operational requirements of catalytic cracking and atmospheric distillation units, refineries blend incoming crudes prior to distillation or blend intermediate refinery streams into commercial products (gasoline grades, jet A-1, ultra-low sulfur diesel, heavy fuel oil). The goal is to maximize refining gross margin (product sales revenue minus crude procurement and transport costs) while strictly enforcing physical volume conservation and all product quality bounds.

---

## 2. Literature Provenance & Source Formulation
The formulation implemented in PipePye is derived directly from classical and contemporary chemical engineering literature:
- **Foundational Benchmark**: Haverly, C. A. (1978). *"Studies of Compromise Solutions for the Pooling Problem"*, ACM SIGMAP Bulletin, 25, 19–28.
- **Industrial LP Formulation**: Baker, T. E., & Lasdon, L. S. (1985). *"Successive Linear Programming at Exxon"*, Management Science, 31(3), 264–274.
- **Standard Textbook Reference**: Williams, H. P. (2013). *Model Building in Mathematical Programming* (5th ed., Chapter 12: "Refinery Blending"). John Wiley & Sons.
- **Engineering Specification Basis**: Gary, J. H., Handwerk, G. E., & Kaiser, M. J. (2007). *Petroleum Refining: Technology and Economics* (5th ed.). CRC Press.

---

## 3. Mathematical Formulation

### Sets & Indices
- $c \in \mathcal{C} = \{1, \dots, C\}$: Set of available crude oil feedstocks.
- $p \in \mathcal{P} = \{1, \dots, P\}$: Set of refined commercial blend products.
- $q \in \mathcal{Q} = \{1, \dots, Q\}$: Set of chemical quality assay metrics (e.g., sulfur content, API gravity, octane number, Reid vapor pressure).

### Decision Variables
- $x_{c, p} \ge 0$: Volume of crude feedstock $c \in \mathcal{C}$ allocated to blend product $p \in \mathcal{P}$ (in thousands of barrels/day).

### Parameters
- $\text{Cost}_c$: Procurement and transport cost per unit volume of crude $c$ (\$/bbl).
- $\text{Price}_p$: Wholesale market selling price per unit volume of product $p$ (\$/bbl).
- $S_c$: Total available supply limit of crude feedstock $c$ (bbl/day).
- $D^{\min}_p, D^{\max}_p$: Minimum and maximum commercial demand commitments for product $p$.
- $A_{c, q}$: Measured assay concentration or property value of quality metric $q$ in crude $c$.
- $Q^{\min}_{p, q}, Q^{\max}_{p, q}$: Minimum and maximum allowable commercial specification limits for quality metric $q$ in finished product $p$.

### Objective Function
Minimize total net procurement cost minus revenue (standard LP minimization format):
$$\min_{x \ge 0} \sum_{c \in \mathcal{C}} \sum_{p \in \mathcal{P}} \left( \text{Cost}_c - \text{Price}_p \right) x_{c, p}$$

### Constraints
1. **Crude Supply Availability**:
   $$\sum_{p \in \mathcal{P}} x_{c, p} \le S_c \quad \forall c \in \mathcal{C}$$

2. **Commercial Product Demand Windows**:
   $$\sum_{c \in \mathcal{C}} x_{c, p} \ge D^{\min}_p \quad \forall p \in \mathcal{P}$$
   $$\sum_{c \in \mathcal{C}} x_{c, p} \le D^{\max}_p \quad \forall p \in \mathcal{P}$$

3. **Product Quality Attribute Specifications**:
   $$\frac{\sum_{c \in \mathcal{C}} A_{c, q} x_{c, p}}{\sum_{c \in \mathcal{C}} x_{c, p}} \le Q^{\max}_{p, q} \implies \sum_{c \in \mathcal{C}} \left( A_{c, q} - Q^{\max}_{p, q} \right) x_{c, p} \le 0 \quad \forall p \in \mathcal{P}, q \in \mathcal{Q}$$
   $$\frac{\sum_{c \in \mathcal{C}} A_{c, q} x_{c, p}}{\sum_{c \in \mathcal{C}} x_{c, p}} \ge Q^{\min}_{p, q} \implies \sum_{c \in \mathcal{C}} \left( Q^{\min}_{p, q} - A_{c, q} \right) x_{c, p} \le 0 \quad \forall p \in \mathcal{P}, q \in \mathcal{Q}$$

---

## 4. Assumptions & Simplifications in PipePye
1. **Linear Quality Blending**: Volumetric linear additivity is assumed for all chemical attributes. In physical refining, properties such as octane and RVP exhibit nonlinear blending characteristics; in industrial linear programming models (such as Aspen PIMS or Haverly GRTMPS), these are transformed via empirical linear blending index numbers prior to solving.
2. **Decoupled Intermediate Pooling**: Bilinear pooling terms (stream concentration multiplied by flow rate) are absent because feeds flow directly from dedicated crude tanks to product blenders.
3. **Single Operational Shift Horizon**: The model represents steady-state operation over a fixed shift, isolating pure blending economics from multi-period inventory dynamics.

---

## 5. Mathematical Structure & Pre-Registered Predictions

### Structural Classification
- **Dense Quality Coupling**: Nonzero density ranges from $9.01\%$ to $30.13\%$. Because every crude candidate potentially enters every quality equation for a product, the constraint matrix contains dense rows of length $C$.
- **Compact Matrix Footprint**: Total rows $m \in [26, 155]$, columns $n \in [18, 260]$. Fits entirely in CPU L1/L2 cache.
- **Low Gini Inequality**: Row length distribution is uniform ($\text{Gini} \approx 0.086 - 0.092$).

### Pre-Registered Predictions
- **Algorithm Prediction**: **Dual Revised Simplex (CPU)**. The dense quality constraints produce ill-conditioned operators for first-order gradient methods (PDHG), requiring thousands of damping iterations. Dual Simplex factorizes the small basis matrix in microseconds and traces vertices directly, finding active quality constraints in $< 100$ pivots.
- **Hardware Prediction**: **CPU Execution**. GPU acceleration is counter-productive because the entire matrix fits in CPU cache (4 KB to 80 KB). PCIe transfer and CUDA kernel launch overhead (> 20 ms) exceed CPU simplex solve time by multiple orders of magnitude.
