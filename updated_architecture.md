# PipePye — Optimization Solver Architecture & Development Roadmap

## 0. Project North Star

Build a **from-scratch, sovereign mathematical optimization engine** for large, sparse industrial optimization workloads, initially targeting **LP**, then **MILP** and **convex QP**.

The central research question is not:

> Which optimization algorithm is universally fastest?

It is:

> **Which algorithm is best suited to a given problem structure, workload, and hardware architecture?**

The project therefore treats **problem structure × algorithm × hardware** as the core design space.

The solver must ultimately be able to demonstrate:

- mathematical correctness;
- numerical robustness;
- scalability on large sparse problems;
- measurable CPU/GPU performance;
- relevance to refinery, blending, planning, logistics, and energy workloads;
- reproducibility against recognized public benchmarks;
- a transparent, extensible implementation with no dependency on an existing optimization solver library for the core.

---

# 1. What We Are Actually Building

This is **not** a CPLEX clone.

It is also not merely:

- a CUDA implementation of an LP algorithm;
- a web dashboard around an existing solver;
- an "AI optimizer";
- a collection of disconnected mathematical demos.

The intended system is a **hybrid optimization engine** in which multiple computational strategies can coexist and eventually be selected according to workload characteristics.

Conceptually:

```text
                         INPUT MODEL
                        MPS / LP / API
                              │
                              ▼
                    ┌──────────────────┐
                    │ Parse + Validate │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Presolve + Scale │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Characterization │
                    │                  │
                    │ m, n, nnz        │
                    │ sparsity         │
                    │ structure        │
                    │ conditioning     │
                    │ memory footprint │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Execution Policy │
                    │                  │
                    │ method + backend │
                    └────────┬─────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              ▼              ▼
        Dual Simplex      PDHG/PDLP       IPM
            CPU          CPU / GPU     CPU / GPU study
              │              │              │
              └──────────────┼──────────────┘
                             ▼
                    ┌──────────────────┐
                    │ Solver / Search  │
                    │                  │
                    │ LP               │
                    │ MILP             │
                    │ QP               │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Verification     │
                    │                  │
                    │ primal residual  │
                    │ dual residual    │
                    │ KKT / gap        │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Benchmark Logger │
                    │ + Profiling      │
                    └──────────────────┘
```

---

# 2. Core Design Principles

## 2.1 Correctness before performance

A fast incorrect solver has no value.

Every optimization result should be independently checked for:

- primal feasibility;
- dual feasibility where applicable;
- complementarity / optimality gap;
- objective recomputation;
- numerical status;
- tolerance satisfaction.

Reference solvers such as HiGHS are used as **secondary evidence**, not as the sole correctness mechanism.

---

## 2.2 Benchmarking is part of the product

Every major subsystem should produce reproducible measurements.

The project should eventually support:

```bash
pipepye benchmark --suite netlib
pipepye benchmark --suite mittelmann
pipepye benchmark --suite miplib
pipepye benchmark --suite qplib
pipepye benchmark --suite industrial
```
Every run should record machine-readable results.
---

## 2.3 Never assume that GPU = faster

GPU acceleration is useful only when enough useful parallel work exists to amortize:

- kernel launch overhead;
- host-device transfers;
- synchronization;
- irregular memory access;
- GPU memory limitations.

This means CPU/GPU selection itself is a research problem.

---

## 2.4 From-scratch core

We explicitly require the solver core to be built from mathematical foundations rather than simply wrapping an existing optimization library.

Therefore:

**Allowed:**

- reading papers;
- studying existing solver architecture;
- using standard system/CUDA libraries;
- comparing against HiGHS, Gurobi, etc.;
- using public benchmark datasets.

**Not allowed in the solver core:**

- linking to CBC/HiGHS/SCIP/GLPK as the optimization engine;
- copying an existing solver implementation

The external solvers are **references and benchmarks**, not implementation dependencies.

---

## 2.5 Development order is not research order

We will research several algorithm families at the same time, but implement them in an order that de-risks the engineering.

### Research question

```text
Dual Simplex
      ↕
     IPM
      ↕
   PDHG/PDLP
```

### Practical development order

```text
Sparse core
    ↓
PDHG-style LP path
    ↓
Dual Simplex
    ↓
MILP
    ↓
QP
    ↓
Advanced IPM / MIQP / other extensions
```

This prevents the team from spending months implementing several difficult solver families before any useful end-to-end system exists.

---

# 3. Algorithm Strategy

## 3.1 Dual / Revised Simplex

### Why it matters

Dual simplex is highly relevant to:

- sparse LPs;
- repeated solves;
- warm starts;
- MILP node re-solves.

Related LPs frequently differ only slightly, making basis reuse valuable.

### Why it is difficult

A serious implementation requires:

- basis management;
- sparse factorization;
- pivoting;
- pricing;
- basis updates;
- numerical stability;
- hyper-sparsity handling.

Parallelization is possible, but the computational pattern is relatively irregular.

### Planned role

**Primary CPU LP path and future MILP node solver.**

Initial implementation should be deliberately narrower than a commercial solver.

### Reference

Huangfu & Hall, *Parallelizing the dual revised simplex method*:

https://link.springer.com/article/10.1007/s12532-017-0130-5

---

## 3.2 PDHG / PDLP-style first-order methods

### Why it matters

First-order primal-dual methods expose operations such as:

- `A x`;
- `Aᵀ y`;
- vector updates;
- reductions.

These map naturally to sparse GPU execution.

PDLP research demonstrates that first-order methods can handle very large sparse LPs and can be strengthened using:

- diagonal/Ruiz-style preconditioning;
- adaptive step sizes;
- restarts;
- practical termination criteria;
- polishing/refinement.

### Important limitation

A naive PDHG implementation is **not** enough.

Tight tolerances can require many iterations, and practical performance depends heavily on scaling, restart policy and termination logic.

Therefore the PDHG path should be based on the literature rather than a textbook-only implementation.

### Planned role

**Primary GPU-oriented LP research path.**

### References

Google, *Practical First-Order Method for Large-Scale Linear Programming*:

https://proceedings.neurips.cc/paper_files/paper/2021/hash/a8fbbd3b11424ce032ba813493d95ad7-Abstract.html

Restarted Halpern/PDHG work:

https://arxiv.org/abs/2407.16144

cuPDLPx:

https://arxiv.org/abs/2507.14051

Repository:

https://github.com/MIT-Lu-Lab/cuPDLPx

---

## 3.3 Interior-Point Methods

### Why it matters

Interior-point methods offer:

- strong continuous optimization performance;
- high-accuracy solutions;
- a powerful alternative to first-order methods.

### Why they are not the initial MVP

Classic large sparse IPM implementations can be dominated by:

- KKT systems;
- sparse factorization;
- fill-in;
- conditioning;
- memory pressure.

These are difficult problems on a small consumer GPU.

However, research on matrix-free IPM shows that large parts of the method can be expressed through operations such as `A x`, `Aᵀ y`, and iterative linear algebra.

### Planned role

**Research / secondary path initially.**

Do not commit to CPU-only IPM. Investigate matrix-free and iterative approaches before deciding.

### Reference

Smith, Gondzio & Hall, *GPU acceleration of the matrix-free interior point method*:

https://www.research.ed.ac.uk/en/publications/gpu-acceleration-of-the-matrix-free-interior-point-method/

---

# 4. Why the Initial Architecture Is Hybrid

The expected architecture is:

```text
                        Problem
                           │
                     Characterize
                           │
            ┌──────────────┼──────────────┐
            │              │              │
            ▼              ▼              ▼
      Dual Simplex       PDHG           IPM
          CPU           GPU-first      CPU/GPU
            │              │              │
            └──────────────┼──────────────┘
                           ▼
                    Best available path
```

The project does not assume that every LP should run using the same algorithm.

The eventual execution policy should answer questions such as:

- Is the model small enough that GPU overhead dominates?
- Is it large enough to keep the GPU busy?
- Is the matrix highly sparse?
- What sparsity structure does it have?
- Does the workload require repeated warm-started solves?
- Does it appear numerically difficult?
- Is a high-accuracy solution required?
- Is the model a MILP node relaxation?

The initial implementation can use deterministic heuristics. Machine learning is **not** required to make the policy useful.

---

# 5. Numerical Foundation

A serious solver requires more than an algorithm loop.

The numerical foundation should contain:

```text
core/
├── sparse/
│   ├── CSR
│   ├── CSC
│   ├── COO
│   └── permutations
│
├── numerics/
│   ├── dot
│   ├── norm
│   ├── axpy
│   ├── reductions
│   ├── scaling
│   └── residuals
│
└── linear_algebra/
    └── iterative solvers / PCG candidate
```

The most important primitives are:

- sparse matrix-vector multiply;
- transpose sparse matrix-vector multiply;
- vector operations;
- reductions;
- norms;
- scaling;
- permutations.

These operations are shared by multiple algorithm families.

---

# 6. Presolve and Scaling

Presolve belongs **before** the solver, not as a late-stage optimization.

Initial reductions:

- empty rows/columns;
- singleton constraints;
- fixed variables;
- bound tightening;
- forcing constraints;
- simple dominated constraints;
- coefficient-range analysis.

Scaling:

- Ruiz-style equilibration;
- diagonal scaling;
- coefficient-range reporting.

Pipeline:

```text
Raw model
   ↓
Validate
   ↓
Presolve
   ↓
Scale
   ↓
Characterize
   ↓
Solve
```

Reason:

Presolve can substantially reduce problem size and improve the numerical characteristics of the model. Scaling can make iterative methods more effective and reduce avoidable numerical difficulties.

Presolve is therefore both:

- a robustness feature;
- a performance feature.

---

# 7. MPS and Model Representation

MPS is the first important interchange format.

Conceptually:

```text
MPS
 │
 ├── objective
 ├── constraints
 ├── coefficients
 ├── RHS
 └── bounds
        ↓
Internal LinearProgram
        ↓
Sparse Matrix
```

The internal representation must **not** be tied to MPS.

Example conceptual representation:

```cpp
struct LinearProgram {
    SparseMatrix A;
    Vector objective;
    Vector rhs;
    Vector lower_bounds;
    Vector upper_bounds;
};
```

This lets the system later support:

```text
MPS ─────┐
LP  ─────┤
API ─────┼──→ Internal Model → Solver
future ──┘
```

### MPS test strategy

Have two separate datasets:

```text
tests/data/mps/
    tiny_min.mps
    bounds.mps
    equality.mps
    inequality.mps
    ranged.mps
```

These test the parser.

Then:

```text
benchmarks/netlib/
```

contains real benchmark instances.

---

# 8. Phase 0 — Foundations & Decisions

### Objective

Make the development environment reproducible and replace guesswork with an evidence-backed algorithm decision process.

### Deliverables

- C++20 project;
- CMake;
- GoogleTest/Catch2;
- CTest;
- GitHub Actions CI;
- CUDA toolchain;
- CUDA error handling;
- Nsight Systems workflow;
- reproducible environment documentation;
- literature matrix;
- algorithm-hardware feasibility memo;
- MPS specification;
- tiny MPS test corpus;
- initial LP API design;
- risk register.

### The algorithm memo must answer

For every candidate:

```text
Algorithm
↓
Dominant operations
↓
Parallelism
↓
Sparse behavior
↓
Memory behavior
↓
Warm-start ability
↓
Numerical behavior
↓
GPU suitability
↓
Implementation difficulty
↓
Research evidence
↓
Our hypothesis
↓
How we will test it
```

### Important

The memo should present **hypotheses**, not pretend that the final CPU/GPU mapping is already known.

### Exit criterion

You can explain, with primary research citations:

> which algorithm families are promising for which workloads, what the limitations are, and exactly how you will experimentally test the claims.

---

# 9. Phase 1 — Sparse Numerical Core

### Objective

Build the computational substrate before implementing a serious solver.

### CPU

Implement and test:

```text
CSR / CSC / COO
SpMV
SpMVᵀ
dot
axpy
norm
reductions
permutations
scaling
```

Start with single-threaded correctness, then multithreading.

### CUDA

Implement:

```text
CUDA SpMV
CUDA SpMVᵀ
warp/block reductions
vector kernels
```

Different SpMV strategies should be compared rather than assuming one is best.

Possible candidates:

- basic CSR;
- row-adaptive CSR;
- merge-path style methods.

### Optional

Add a minimal preconditioned-CG skeleton only if the sparse core is already stable. Do not let PCG derail the core.

### Deliverable

**CPU/GPU microbenchmark report.**

Sweep:

```text
problem size
NNZ
sparsity
matrix structure
```

### First serious graph

> **Runtime vs NNZ, CPU vs GPU, stratified by sparsity/structure.**

This is the first point where the project starts producing research evidence.

---

# 10. Phase 2 — Presolve, Scaling & Characterization

### Objective

Transform arbitrary input models into better-conditioned, smaller and characterized problems.

Implement:

```text
Presolve
 ├── empty rows/columns
 ├── singleton handling
 ├── fixed-variable elimination
 ├── simple bound tightening
 └── forcing/dominance reductions

Scaling
 └── Ruiz/equilibration

Characterization
 ├── m
 ├── n
 ├── nnz
 ├── density
 ├── coefficient range
 ├── structure class
 └── conditioning proxy
```

### Deliverable

For selected instances:

```text
Without presolve/scaling
vs
With presolve/scaling

model size
iterations
residuals
runtime
```

This creates direct evidence for the numerical-robustness story.

---

# 11. Phase 3 — First LP Solver: PDHG-style

### Objective

Get the first complete LP path working quickly enough to begin genuine CPU/GPU experimentation.

### Development sequence

```text
CPU PDHG
   ↓
correctness
   ↓
scaling
   ↓
adaptive step
   ↓
restart
   ↓
KKT-based termination
   ↓
CUDA port
```

The GPU version should aim to keep solver state resident on the device:

```text
CPU
 │
 │ initial model/state transfer
 ▼
GPU
 ├── SpMV
 ├── SpMVᵀ
 ├── vector updates
 ├── reductions
 ├── iteration
 ├── iteration
 └── iteration
 │
 │ final solution transfer
 ▼
CPU
```

This avoids the mistake demonstrated by the initial DAXPY benchmark: a very fast GPU kernel can still yield poor end-to-end performance if data is repeatedly moved across PCIe.

### Accuracy path

Treat polishing/refinement as a planned research component.

The first-order solver should be able to obtain a useful solution, while a later refinement step can target stricter numerical accuracy.

### Exit criterion

- verified LP solutions;
- CPU/GPU timing decomposition;
- convergence logs;
- benchmark results;
- first CPU/GPU crossover map.

---

# 12. Phase 4 — Dual Revised Simplex

### Objective

Add a second fundamentally different LP strategy.

Progressive implementation:

```text
bounded-variable revised simplex
        ↓
dense LU baseline
        ↓
sparse LU / Markowitz
        ↓
dual pricing
        ↓
basis updates
        ↓
hyper-sparsity
        ↓
warm starts
```

Do not attempt to reproduce every feature of a commercial simplex engine.

The important milestone is:

> A real, numerically credible dual-simplex implementation capable of warm-started related solves.

### Why this matters

It gives the architecture a fundamentally different computational regime:

```text
PDHG
→ matrix-free
→ iterative
→ parallel
→ GPU-friendly

Dual Simplex
→ basis-oriented
→ warm-startable
→ high-accuracy
→ CPU-friendly
```

That contrast is central to the project's research thesis.

---

# 13. Phase 5 — Algorithm × Hardware Study

This is the **core research phase**.

Now we have enough machinery to compare:

```text
                 CPU                  GPU
                  │                    │
         ┌────────┼────────┐     ┌─────┴─────┐
         │        │        │     │           │
    Dual Simplex   PDHG    IPM  PDHG       IPM*
```

*where IPM experiments are sufficiently mature.

Measure:

- wall time;
- time to target gap;
- iterations;
- memory;
- GPU utilization;
- achieved bandwidth;
- host-device transfer time;
- kernel time;
- numerical residuals;
- solution quality.

### Important

Record **wins and losses**.

The goal is not to manufacture:

> "GPU = 5× faster."

The goal is to discover:

> **under what conditions each method/backend wins.**

---

# 14. Adaptive Execution Policy

Once benchmark data exists:

```text
Problem characterization
        ↓
Execution policy
        ↓
method
backend
        ↓
solver
```

Initially use deterministic heuristics:

```text
small + warm-startable → Dual Simplex / CPU

huge + sparse + matrix-free
                       → PDHG / GPU

high-accuracy / difficult
                       → simplex or IPM path
```

These are **hypotheses** until validated.

Only after enough empirical data exists should you investigate a statistical or ML-based selector.

### Final interface

```bash
vajra solve model.mps --device auto --method auto
```

The output should explain its decision:

```text
Problem:
  variables: 1.2M
  constraints: 82K
  NNZ: 9.4M
  structure: staircase sparse

Policy:
  method: PDHG
  device: CUDA

Reason:
  large sparse matrix
  sufficient GPU residency
  no warm-start state
```

The explanation is part of the transparency story.

---

# 15. Phase 6 — Industrial Benchmark Suite

This should run in parallel with later LP development.

The project statement asks for industrial relevance, so build a small but defensible suite based on public formulations/literature.

Recommended initial cases:

| Case | Class | Candidate structure | Purpose |
|---|---|---|---|
| Crude blending | LP | blending/pooling | numerical/quality constraints |
| Multi-period planning | LP | staircase/block-angular | very large sparse GPU candidate |
| Refinery scheduling | MILP | time × unit × product | branch-and-bound |
| Unit commitment / dispatch | MILP | unit × period | binary decisions + repeated LP relaxations |

Each case should contain:

```text
model.mps
generator / instance builder
formulation README
reference solution
metadata
benchmark result
```

### Important

For every industrial case, record the **prediction before measuring**:

> "We expect structure X to favor method Y on hardware Z because..."

Then test it.

This is what converts a showcase application into a research experiment.

---

# 16. Phase 7 — MILP

Only begin serious MILP work after LP is reliable.

Architecture:

```text
MILP
 │
 ▼
Branch-and-Bound
 │
 ├── node
 │    ↓
 │  LP relaxation
 │    ↓
 │  LP solver
 │
 └── node
      ↓
    LP relaxation
```

### Initial implementation

1. LP relaxation.
2. Depth-first branch-and-bound.
3. Incumbent tracking.
4. Basic branching.
5. Warm-started dual simplex node solves.
6. Root presolve.

### Later

- pseudocost branching;
- rounding/diving;
- best-bound/best-estimate search;
- Gomory/MIR cuts;
- stronger heuristics;
- parallel node processing.

### Metrics

For MILP, don't only report:

> solved / not solved.

Also report:

- MIP gap vs time;
- time to first feasible solution;
- time to 10%, 1%, 0.1% gap;
- nodes explored;
- nodes/sec;
- LP time;
- memory.

For industrial use, a strong feasible solution quickly can matter more than proving exact optimality.

---

# 17. Phase 8 — QP

Once the LP and sparse numerical infrastructure are mature, add **convex QP**.

A practical initial candidate is an ADMM/operator-splitting approach inspired by OSQP.

Why this path:

- warm-start friendly;
- useful for repeated solves;
- reuses vector and sparse-matrix operations;
- shares conceptual infrastructure with first-order LP methods.

### Initial scope

**Convex QP only.**

Do not immediately attempt nonconvex QP.

Reference:

Stellato et al., *OSQP: An Operator Splitting Solver for Quadratic Programs*:

https://stanford.edu/~boyd/papers/osqp.html

---

# 18. Future Scope

Only after the core system is credible:

```text
LP
 ↓
MILP
 ↓
QP
 ↓
MIQP
 ↓
advanced IPM
 ↓
parallel B&B
 ↓
stronger cuts / heuristics
 ↓
NLP
 ↓
MINLP
```

Possible longer-term research surfaces:

- structure-aware presolve;
- advanced warm starts;
- matrix-free IPM;
- adaptive algorithm selection;
- parallel B&B;
- multi-GPU execution;
- industrial-specific model transformations;
- stronger decomposition methods.

These are **future directions**, not SIH MVP commitments.

---

# 19. Benchmark Architecture

Use several benchmark layers.

## Layer 1 — Unit tests

Purpose:

> Does the implementation work?

Examples:

```text
CSR construction
SpMV
bounds
parser
scaling
residual calculation
```

---

## Layer 2 — Microbenchmarks

Purpose:

> How does a primitive behave?

Examples:

```text
SpMV
SpMVᵀ
dot
axpy
reduction
```

Sweep:

```text
size × NNZ × structure × hardware
```

---

## Layer 3 — Standard LP benchmarks

Primary candidates:

- Netlib;
- Mittelmann.

Purpose:

> Does the LP solver work on recognized problems?

---

## Layer 4 — MILP benchmarks

Primary candidate:

- MIPLIB.

Purpose:

> Does the complete MILP architecture work?

---

## Layer 5 — QP benchmarks

Primary candidate:

- QPLIB convex subset.

---

## Layer 6 — Industrial benchmarks

Purpose:

> Does the solver behave meaningfully on workloads relevant to the target problem?

---

# 20. Benchmark Protocol

Every comparison must specify:

```text
Hardware
CPU
GPU
RAM
VRAM

Software
OS
compiler
CUDA
solver version

Execution
threads
GPU device
precision
time limit
termination tolerance
random seed

Metrics
solve time
time-to-target-gap
objective
primal residual
dual residual
duality gap
iterations
memory
GPU utilization
transfer time
```

For repeated measurements, use a fixed protocol and report an appropriate aggregate such as geometric mean for performance ratios.

Do not hide failures.

---

# 21. Correctness Architecture

Verification should be independent of the optimization implementation as much as possible.

Conceptually:

```text
                Solver
                  │
                  ▼
              Candidate x
                  │
                  ▼
         Independent verifier
                  │
       ┌──────────┼──────────┐
       ▼          ▼          ▼
    primal       dual       gap/KKT
    checks      checks       checks
       │          │          │
       └──────────┼──────────┘
                  ▼
              VALID / FAIL
```

Reference-solver agreement is additional evidence:

```text
Vajra objective
      vs
reference objective
```

but not the sole correctness criterion.

---

# 22. Performance Metrics

Do not report only one number.

For GPU work, decompose:

```text
Total time
=
preprocess
+
H→D transfer
+
kernel execution
+
synchronization
+
D→H transfer
+
verification
```

For iterative algorithms, also report:

```text
time / iteration
iterations to tolerance
time to target gap
```

This helps distinguish:

> slow hardware execution

from:

> slow mathematical convergence.

---

# 23. The First Three Headline Experiments

## Experiment 1 — Sparse kernel crossover

Question:

> When does GPU SpMV outperform CPU SpMV?

Sweep:

```text
NNZ
matrix dimensions
sparsity
structure
```

Output:

> CPU/GPU crossover map.

---

## Experiment 2 — Algorithm crossover

Question:

> When does PDHG outperform dual simplex, and vice versa?

Sweep:

```text
problem size
sparsity
structure
warm-start availability
accuracy target
```

Output:

> Algorithm × workload performance map.

---

## Experiment 3 — Hardware-aware industrial workload study

Question:

> Do Indian-industry-shaped sparse models exhibit the same performance regimes as generic benchmark instances?

Compare:

```text
blending
planning
scheduling
dispatch
```

Output:

> Industry workload → algorithm → hardware map.

This is the experiment that ties the entire project back to the SIH problem statement.

---

# 24. What the Final SIH Demo Should Show

The final system should not be a passive dashboard.

A judge should be able to watch the actual solver work.

Example:

```text
                VAJRA

Problem: Crude Blending
Variables: 248,000
Constraints: 42,300
NNZ: 1,870,000

Characterization:
  structure: block-sparse
  density: 0.018%
  estimated VRAM: 1.9 GB

Execution policy:
  method: PDHG
  backend: CUDA
```

Then a live convergence plot:

```text
Residual
  │\
  │ \
  │  \
  │   \____
  │        \___
  └────────────── Iterations
```

Show live:

```text
iteration
objective
primal residual
dual residual
duality gap
GPU utilization
VRAM
elapsed time
```

Then show a CPU replay:

```text
CPU: 8.42 s
GPU: 1.87 s

GPU speedup: 4.50×
```

Then deliberately make the problem smaller:

```text
CPU: 0.31 s
GPU: 0.58 s

GPU loses.
```

Then explain:

> GPU overhead dominates at this workload size.

Finally:

```text
--device auto
```

selects CPU for the small case and GPU for the large case.

This demonstrates the central research idea live instead of merely showing a benchmark table.

---

# 25. Claim → Evidence Map

| Claim | Evidence |
|---|---|
| Sovereign | From-scratch repository; no solver-library dependency; transparent internals |
| Correct | Independent primal/dual/KKT checks; reference-solver agreement |
| Numerically robust | difficult-instance suite; scaling/presolve experiments; failure classification |
| Scalable | runtime/memory scaling curves; large sparse instances |
| GPU useful | CPU/GPU crossover map and kernel profiling |
| Hardware-aware | adaptive execution policy |
| Industrially relevant | open-literature industrial models + structural analysis |
| Reproducible | public benchmark runner + machine-readable outputs + fixed protocol |
| Extensible | modular LP → MILP → QP architecture |

---

# 26. Repository Architecture

A mature repository should eventually look approximately like:

```text
vajra/
│
├── CMakeLists.txt
├── README.md
│
├── include/
│   └── vajra/
│       ├── core/
│       ├── model/
│       ├── sparse/
│       ├── numerics/
│       ├── algorithms/
│       ├── backend/
│       ├── presolve/
│       ├── verification/
│       └── benchmark/
│
├── src/
│   ├── core/
│   ├── model/
│   ├── sparse/
│   ├── numerics/
│   ├── algorithms/
│   │   ├── lp/
│   │   ├── milp/
│   │   └── qp/
│   ├── presolve/
│   └── verification/
│
├── cuda/
│   ├── sparse/
│   ├── reductions/
│   ├── vector/
│   └── algorithms/
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── data/
│
├── benchmarks/
│   ├── netlib/
│   ├── mittelmann/
│   ├── miplib/
│   ├── qplib/
│   ├── synthetic/
│   └── industrial/
│
├── experiments/
│   ├── spmv/
│   ├── pdhg/
│   ├── simplex/
│   └── policy/
│
├── scripts/
│
└── docs/
    ├── architecture/
    ├── algorithms/
    ├── benchmarking/
    ├── experiments/
    └── decisions/
```

Do not create all of this immediately. Let the architecture grow with the system.

---

# 27. Development Gates

The project should use hard gates rather than dates alone.

### Gate 0 — Toolchain

```text
C++20
CMake
tests
CI
CUDA
Nsight
```

---

### Gate 1 — Sparse core

```text
CSR/CSC
SpMV
reductions
CPU/GPU validation
```

---

### Gate 2 — LP correctness

```text
MPS
presolve
scaling
LP solver
independent verification
```

---

### Gate 3 — GPU viability

```text
GPU LP path
end-to-end timing
crossover map
```

---

### Gate 4 — Algorithm comparison

```text
PDHG
Dual Simplex
measured comparison
```

---

### Gate 5 — Industrial relevance

```text
industrial models
structure characterization
benchmark results
```

---

### Gate 6 — MILP

Only after LP is solid.

---

### Gate 7 — QP

Only after LP infrastructure is solid.

---

# 28. Immediate Next Steps

The current foundational toolchain is already in place:

```text
C++20 / CMake        ✓
GoogleTest / CTest   ✓
CI                   ✓
CUDA                 ✓
CUDA error handling  ✓
Nsight Systems       ✓
environment docs     ✓
```

The next sequence should therefore be:

### 1. Finish Phase 0 documentation

Produce:

```text
docs/algorithm-hardware-feasibility.md
docs/mps-spec.md
tests/data/mps/
```

The algorithm memo should explicitly mark conclusions as **hypotheses to test**.

### 2. Build SpMV

Do not jump into the LP solver yet.

Implement:

```text
CPU SpMV
CUDA SpMV
```

and validate identical numerical results within a defined tolerance.

### 3. Build the microbenchmark harness

Run sweeps over:

```text
NNZ
matrix size
sparsity
structure
```

### 4. Produce the first real research plot

> **CPU vs GPU SpMV crossover map.**

This becomes the first empirical evidence behind the architecture.

---

# 29. The Core Philosophy

Three sentences define the project:

> **1. The MVP is not "an LP solver with GPU." It is a measured investigation of which LP computational strategy is appropriate for which workload and hardware.**

> **2. Presolve, scaling and verification are core numerical infrastructure, not optional polish.**

> **3. The industrial benchmark suite turns the sovereignty argument into something measurable: the solver is not merely open, it is evaluated and tuned against workloads relevant to the target domain.**

The strongest eventual contribution is therefore:

```text
             PROBLEM STRUCTURE
                    ×
                ALGORITHM
                    ×
                 HARDWARE
                    ↓
          OPTIMAL EXECUTION POLICY
```

That is the architecture around which PipePye/Vajra should be developed.
