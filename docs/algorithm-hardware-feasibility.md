# PipePye — Algorithm × Hardware Feasibility Memo & LP MVP Candidate Path

## 1. Executive Summary: The Candidate Path for the LP MVP

Based on **Section 2.5 ("Development order is not research order")** and **Sections 3, 11, and 12 of `updated_architecture.md`**, PipePye does not attempt to solve every problem with one algorithm, nor does it attempt to build every complex solver simultaneously before an end-to-end pipeline exists.

### The Candidate Path Sequence

```text
Phase 1: Sparse Core Substrate (CSR/CSC/COO, SpMV, SpMVᵀ, reductions)
    ↓
Phase 2: Presolve, Scaling (Ruiz/equilibration) & Problem Characterization
    ↓
Phase 3: LP Path 1 — First-Order PDHG (CPU baseline → CUDA resident solver)
    ↓
Phase 4: LP Path 2 — Dual Revised Simplex (Dense LU → Sparse Markowitz LU → Devex)
    ↓
Phase 5: Algorithm × Hardware Empirical Crossover Study (PDHG vs. Simplex on CPU/GPU)
```

### Why this Sequence is the Optimal Candidate Path for the MVP

1. **Fastest Path to an End-to-End Measurable Solver (De-risking Phase 3)**:
   A textbook revised simplex solver with sparse LU factorization, Markowitz pivoting, Forrest-Tomlin updates, and anti-degeneracy mechanisms is a 4–6 month undertaking before a single LP solves cleanly.
   In contrast, **Primal-Dual Hybrid Gradient (PDHG / PDLP style)** requires only:
   - Sparse matrix-vector multiplications ($A x$ and $A^T y$);
   - Vector AXPY operations and element-wise projections onto bounds $[l, u]$;
   - Reductions and Euclidean norms for residual checking.
   
   By choosing PDHG as the **first operational LP solver**, the team builds an end-to-end pipeline (MPS ingestion $\to$ Scaling $\to$ Solver $\to$ Independent Verification) within weeks, directly exercising the GPU toolchain built in Phase 0.

2. **GPU Residency Proof-of-Concept**:
   The initial Phase 0 DAXPY microbenchmark proved that moving data across PCIe every iteration cripples throughput ($0.07\times$ end-to-end vs. $3.0\times$ kernel speedup). PDHG keeps the problem matrices ($A, A^T$) and iterate vectors ($x, y, \bar{x}, \bar{y}$) **100% resident in GPU VRAM**, amortizing host-device transfer over thousands of fast iterations.

3. **Dual Simplex as the Essential Precision & Warm-Start Anchor (Phase 4)**:
   PDHG cannot warm-start efficiently from single-variable bound modifications and struggles beyond $10^{-4} - 10^{-6}$ tolerances on degenerate problems. **Dual Revised Simplex** is implemented immediately following PDHG to provide:
   - Exact vertex / basic solutions;
   - Unambiguous certificates of primal/dual infeasibility;
   - High-throughput warm-starting required for Branch-and-Bound MILP nodes in Phase 7.

4. **Independent Verification Eliminates Confirmation Bias**:
   Neither solver trusts its own internal metrics. Both feed their final candidate solutions to an independent **KKT Verification Oracle** that evaluates primal residual $\|Ax - b\|_\infty$, dual residual $\|A^T y + s - c\|_\infty$, and complementarity gap on the original untransformed model.

---

## 2. Algorithm × Hardware Literature Matrix & Hypotheses

| Dimension | Dual Revised Simplex (Phase 4) | First-Order PDHG / PDLP (Phase 3) | Interior Point Method (IPM) (Research) |
|---|---|---|---|
| **Dominant Operations** | Basis solve $B x_B = b$, dual solve $B^T y = c_B$, rank-1 basis updates ($B_{k+1} = B_k + u v^T$), pricing dot products. | Sparse Matrix-Vector ($A x$, $A^T y$), vector additions (`axpy`), vector norms, box projections. | Normal equations $(A \Theta A^T) \Delta y = r$, Sparse Cholesky factorization ($L L^T$), dense triangular solves. |
| **Parallelism Profile** | Highly sequential (one pivot per step); limited parallel pricing; irregular sparse access. | Massively parallel (embarrassingly parallel SpMV and coordinate-wise projections across all NNZ). | Coarse-grain parallel linear algebra, but sparse Cholesky factorization tree has sequential critical paths. |
| **Sparse Matrix Behavior** | Factorization fill-in; requires Markowitz pivoting and hyper-sparse techniques (Gilbert-Peierls). | Matrix-free; static sparsity pattern; zero fill-in during execution; predictable memory layout. | Massive fill-in in $A A^T$; requires AMD/ColAMD reordering; fill-in can exceed original NNZ by $10\times - 100\times$. |
| **Memory Footprint** | Low to Medium: Matrix $A$ + Basis factors $L, U$ + update buffers. Fits in CPU L3 cache for small LPs. | Extremely Low: Exactly $O(\text{NNZ} + m + n)$ static memory. Easily fits within 6GB VRAM on RTX 3050. | Extremely High: Cholesky factor $L$ can easily exhaust 6GB VRAM on large industrial models. |
| **Warm-Start Capability** | **Exceptional**: 5–50 dual pivots to re-optimize after bound changes. Essential for MILP. | **Poor**: Can warm-start initial iterate $(x_0, y_0)$, but convergence rate does not dramatically improve. | **Very Poor**: Initial points must remain strictly interior; warm-starts close to the boundary destroy conditioning. |
| **Numerical Precision** | High (exact basis representation, $10^{-8} - 10^{-12}$). Exact active set identified. | Moderate ($10^{-4}$ easily, $10^{-6}$ with restarts/scaling, $10^{-8}$ very difficult). | High ($10^{-8} - 10^{-10}$ in 30–60 iterations if factorization does not break down). |
| **Hardware Affinity** | **CPU-First**: High single-thread clock speed, low latency memory cache, branch predictability. | **GPU-First**: High memory bandwidth (GDDR6), SIMT streaming multiprocessors, high arithmetic throughput. | **Hybrid**: Factorization on CPU / iterative solves on GPU; matrix-free iterative Krylov on GPU. |
| **Key Literature References** | Huangfu & Hall (2018), *Parallelizing the dual revised simplex method*; Bixby (2002). | Applegate et al. (Google, 2021), *Practical First-Order Method for Large-Scale LP*; cuPDLPx (Lu, 2025). | Smith, Gondzio & Hall (2023), *GPU acceleration of matrix-free IPM*; Wright (1997), *Primal-Dual IPM*. |

---

## 3. Known Risks and Mitigation Strategies

### Risk 1: PDHG Convergence Stagnation & Tail Tail-Off
- **Mechanism**: First-order methods exhibit asymptotic $O(1/k)$ convergence. On ill-conditioned problems, residuals oscillate or plateau at $10^{-3}$, failing strict industrial tolerance targets.
- **Mitigation**:
  1. **Ruiz Equilibration**: Rescale rows and columns to unit $\ell_\infty$ norm iteratively before passing to GPU.
  2. **Pock-Chambolle Preconditioning**: Compute diagonal scaling matrices $T = \text{diag}(\tau_j)$ and $\Sigma = \text{diag}(\sigma_i)$ where $\tau_j = 1 / \sum_i |A_{ij}|$ and $\sigma_i = 1 / \sum_j |A_{ij}|$, ensuring step sizes satisfy $\sigma_i \tau_j A_{ij}^2 < 1$.
  3. **Adaptive Restarts**: Detect normalized duality gap stagnation and restart the algorithm from the current ergodic average $(\bar{x}, \bar{y})$.
  4. **Active-Set Crossover / Simplex Polish**: Use PDHG to rapidly identify the candidate active set within $\varepsilon = 10^{-4}$, then pass non-basic bounds to Track A Simplex for clean basis cleanup.

### Risk 2: Simplex Basis Matrix Singularities & Numerical Drift
- **Mechanism**: In revised simplex, the basis inverse $B^{-1}$ is updated via rank-1 operations (Product Form of the Inverse or Forrest-Tomlin). Floating-point roundoff accumulates, leading to loss of feasibility and basis singularity.
- **Mitigation**:
  1. Start development with **Dense LU with partial pivoting** on Netlib subsets ($n \le 1000$) to validate pricing and ratio logic independently of sparse factorization bugs.
  2. Migrate to **Sparse LU with Markowitz threshold pivoting** (threshold parameter $u = 0.1$).
  3. Enforce **periodic reinversion**: discard accumulated update factors and recompute a fresh sparse LU decomposition every 50–100 iterations or whenever $\|B x_B - b\|_\infty > 10^{-8}$.
  4. Implement **Harris's two-pass ratio test** and **bound flipping (BFRT)** to navigate degenerate vertices without cycling.

### Risk 3: PCIe Bandwidth Bottleneck & Host Synchronization
- **Mechanism**: Frequent kernel launches with implicit host synchronizations (`cudaDeviceSynchronize()`, scalar downloads for termination checks) can degrade GPU compute throughput by $10\times$.
- **Mitigation**:
  1. Evaluate termination criteria only every $K = 64$ or $128$ iterations.
  2. Implement termination check reductions inside a GPU reduction kernel; only transfer a single 64-bit convergence flag to host.
  3. Keep all iterate vectors, step size parameters, and matrix structures persistently allocated on the GPU during the entire solve loop.

---

## 4. Empirical Hypotheses to Test in Phase 5

1. **Hypothesis 1 (Structure × Hardware)**:
   *Staircase and block-angular multi-period planning instances ($N > 10^5$, NNZ $> 10^6$) will achieve $>5\times$ speedup on the RTX 3050 via PDHG compared to CPU Dual Simplex at moderate accuracy ($\varepsilon = 10^{-4}$).*
2. **Hypothesis 2 (Degeneracy × Algorithm)**:
   *Highly degenerate network-flow and crude blending instances will stall under naive PDHG, whereas CPU Dual Simplex with Devex pricing and bound-flipping will resolve to exact optimality in fewer iterations and lower wall-clock time.*
3. **Hypothesis 3 (Scale Crossover)**:
   *For small problems ($N < 5,000$ variables), CPU execution will consistently beat GPU execution due to PCIe initialization and kernel launch overheads, defining the low-end switching boundary of our Adaptive Execution Policy.*
