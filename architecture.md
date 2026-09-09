# PipePye — Analytical Review + Engineering Progression Plan

You've asked two things across your messages: (1) critique my understanding against the problem statement, and (2) give a concrete progression/pipeline. I'll do both — first the critique (because a few of your assumptions will hurt you if uncorrected), then the full build plan.

---

## Part 1 — Critique of Your Current Understanding

### Point 1: "The gap is Indian industry use cases" — Partially right, but you're framing it wrong

**Critique:** The gap is *not* that CPLEX/Gurobi can't solve refinery problems. They can, very well. The defensible gap is threefold:

1. **Sovereignty/auditability** — no Indian refinery can inspect, modify, or certify a proprietary solver's internals. For strategic sectors (refining, power dispatch), that's a national-infrastructure argument, and it's the one the PS actually makes.
2. **Customizability** — industrial solvers benefit enormously from structure exploitation (network matrices in blending, block-angular structure in multi-period planning, staircase structure in scheduling). You cannot inject structure-specific pivoting, presolve, or decomposition rules into a closed solver. An owned core can.
3. **Cost at scale** — enterprise solver licenses are priced per-core/cloud-use; a refinery running large MILPs daily across many users pays heavily.

**Add to your understanding:** the phrase "developed, validated and optimized specifically for Indian industrial use cases" is best interpreted as: *industrial-shape benchmark suite + tuning for those structures*, not "algorithmically different math." The math is universal; the validation and tuning are not. Your industry-shaped benchmark suite is therefore not a nice-to-have — it IS the differentiator. Prioritize it more than your current doc does (it's buried at §21/§45).

### Point 2: GPU acceleration — You have the right skepticism, but you're underestimating one thing

**Critique:** Correct instinct that GPU doesn't always win. But here's the hard truth you must internalize *now*:

- **Simplex is fundamentally hard to GPU-accelerate.** It's sequential (one basis change at a time), sparse, and irregular — the worst case for GPUs. Revised simplex belongs on CPU, period.
- **Interior-point methods (IPM) are GPU-friendly at the linear-algebra level** — but the classic IPM bottleneck is the **normal equations / factorization step**, which is sparse Cholesky — again irregular and hard on GPU.
- **First-order methods (ADMM / PDLP-style, like Google's lifted PDHG)** map beautifully to GPU (SpMV, axpy, dot products) — but they converge slowly and struggle to reach the 1e-6…1e-8 accuracy judges and benchmarks expect. This is exactly why Google's cuPDLP is interesting but hasn't displaced Gurobi.

**What to add:** Your MVP claim should be structured as a **three-way algorithm-hardware mapping study**:

| Algorithm | GPU affinity | Accuracy ceiling | Role |
|---|---|---|---|
| Revised simplex | Low | High (exact basis) | CPU backbone, warm-starts, MILP nodes |
| Interior-point | Medium | High | CPU for accuracy; GPU experiments on kernels |
| First-order (ADMM/PDHG) | High | Moderate | GPU path for huge problems / good warm starts |

This mapping IS your research contribution. Don't pick one algorithm before building this table experimentally.

### Point 3: LP → MILP → QP ordering — Right, but add presolve much earlier

**Critique:** Your phasing puts presolve in Phase 8 territory (MILP). That's a mistake. **Presolve is an LP technology first** (singleton rows/columns, forcing constraints, dominated columns, bound tightening, free variable substitution). It's cheap to implement, dramatically improves conditioning, and directly supports your numerical-robustness USP. Move basic presolve into the LP phase.

Also: **scaling** (equilibration, geometric mean) is arguably the single highest-leverage numerical-robustness feature in any LP solver. It deserves to be a named subsystem, not a recovery mechanism (your §18 lists scaling as a "recovery" fallback — it should be default preprocessing).

### Point 4: Benchmark suite — Right, but your correctness metric is incomplete

**Critique:** `|f_calc − f_ref|` is insufficient. Two problems:
1. Against what reference? For LP you should use **KKT/optimality certificate checks** (primal residual, dual residual, complementarity gap), which need no external solver. Reference-objective comparison is secondary evidence.
2. For degenerate problems, multiple optimal solutions exist — objective comparison can mislead. Report **duality gap / KKT residual** as primary, objective-vs-reference as secondary.

Also add **Mittelmann's own methodology discipline**: fixed time limits, fixed tolerance, single-thread pinning, repeated runs, report geometric means — judges will respect a half-page "benchmark protocol" section.

### Broader gaps in your doc (things missing entirely)

1. **No dual/simplex warm-starting plan.** Essential for MILP (every B&B node re-solves an LP) and for industrial re-solves (refinery re-optimized hourly with small data changes). This is a genuine advantage a sovereign core can offer.
2. **No numerical linear algebra strategy for IPM.** Will you form normal equations? Use iterative Krylov (CG on normal equations, preconditioned)? For GPU, an iterative approach with a good preconditioner is far more realistic than sparse Cholesky on a 6GB consumer card. Name this decision explicitly.
3. **No license/legal note.** "From scratch" means no linking to GLPK/COIN-OR code. State it — it protects the sovereignty claim.
4. **No fallback accuracy policy.** First-order methods give you a solution with gap ε. Define a documented pipeline: first-order fast solution → polish/refinement step (e.g., crossover to simplex, or active-set cleanup) to recover high accuracy. This is exactly what cuPDLP-style systems do and it's your most credible GPU story.
5. **Power dispatch (UC/ED) is a MILP** (unit commitment has binary on/off variables). It's actually your best MILP case study because the LP relaxation is tight-ish and B&B succeeds — good for demos. Add it to Phase 8 case studies.

---

## Part 2 — The Progression & Pipeline (Detailed)

### Phase 0 — Foundations & Decisions (Weeks 1–3)

Deliverables:
- C++20 skeleton, CMake, Catch2/GoogleTest, CI.
- CUDA toolchain + one profiling tool (Nsight Systems / Compute Sanitizer) working.
- **Written algorithm-hardware feasibility memo** (the table above) — this is a graded-looking artifact for SIH.
- MPS parser spec (fixed + free format), test corpus.

**Exit criterion:** you can state, with citations, which LP algorithm family runs on which hardware and why.

### Phase 1 — Sparse Core (Weeks 3–6)

```
core/sparse:  CSR/CSC/COO, row/column views, permutations, scaling utilities
core/num:     SpMV, SpMVᵀ, dot, axpy, norms, reductions  (CPU: multithreaded)
              CUDA: SpMV (CSR-adaptive + merge-path), warp reductions
benchmark:    micro-bench harness for kernels alone (NNZ sweep, structure sweep)
```

Deliverable: **kernel-level CPU/GPU crossover plot** before any solver exists. This de-risks everything downstream and is your first real result.

### Phase 2 — Presolve & Scaling (Weeks 6–8)

```
presolve:     empty rows/cols, singleton constraints, forcing/dominated rows,
              bound tightening, coefficient range analysis
scaling:      geometric mean + equilibration (default ON), report condition estimates
```

Deliverable: same instance solved with/without presolve+scaling — residual and iteration-count comparison table. Direct evidence for your "numerical stability" USP.

### Phase 3 — LP Solver, CPU (Weeks 8–14)

Two parallel tracks, decide by data:

- **Track A (primary, correctness):** Revised simplex — dense-lu start → sparse LU with Markowitz pivoting → Forrest-Tomlin or product-form updates. Devolve: this is hard; a simplified bound-flipping + Devex-pricing version is acceptable for MVP.
- **Track B (parallel, GPU thesis):** PDHG/ADMM first-order method with GPU kernels + **crossover/polish step** for accuracy.

**Verification layer (build here, not later):**
```
verify(): primal feasibility, dual feasibility, complementarity,
          objective recomputation, status classification
          (optimal / feasible / infeasible-certificate / numerically-failed)
```

**Exit criterion:** ≥90% of a chosen Netlib subset solved to KKT tolerance ≤1e-6, verified independently.

### Phase 4 — GPU LP End-to-End (Weeks 14–20)

- Port first-order solver fully to CUDA (persistent kernels, fused SpMV+vector ops, pinned-memory transfers, batching to hide transfer).
- Hybrid policy: presolve/factorization on CPU, iterative core on GPU.
- Full metric stack: wall time decomposed into preprocess / transfer / kernel / sync; occupancy, bandwidth vs peak (compare to `% of HBM/Achieved BW` from Nsight).

**Deliverable: the CPU/GPU crossover map** (runtime vs NNZ, stratified by sparsity structure). This is headline figure #1.

### Phase 5 — Benchmark & Reproducibility Infrastructure (build continuously, harden weeks 16–20)

```
benchmark/
  suites:  netlib (subset ~50), mittelmann LP set, MIPLIB "easy" subset (for Phase 7),
           synthetic stress suite (10 edge-case generators)
  protocol: fixed seeds, time limits, thread pinning, 5-run geometric mean,
            machine-readable JSON per run
  runner:  ./bench --suite X --runs 5 → CSV + plots
  refs:    HiGHS (primary open ref), Gurobi (if license available, as ceiling marker)
```

**Deliverable:** one-command reproduction. Publish the results including failures.

### Phase 6 — Industrial Benchmark Suite (Weeks 18–24, parallel with Phase 5)

Build 4 generator-based case studies with documented formulations:

| Case | Structure | Class | Why |
|---|---|---|---|
| Crude blending | pooling/blending network | LP (→MILP with mode binaries later) | dense-ish columns, quality bounds → numerically tricky |
| Refinery scheduling | time × unit × product | MILP, weak-ish relaxation | tests B&B |
| Multi-period planning | staircase block-angular | LP, huge & sparse | GPU-favorable structure |
| Unit commitment / dispatch | unit × hour, binaries + ramps | MILP, tight relaxation | winnable MILP showcase |

**Deliverable:** generator scripts + reference solutions from HiGHS + a "characteristics table" per instance (m, n, nnz, density, degeneracy indicators).

### Phase 7 — MILP (Weeks 22–30)

Order inside the phase:

1. LP relaxation + depth-first B&B + best-bound node storage
2. Branching: most-infeasible → pseudocost
3. Incumbent heuristics: rounding, diving
4. **Warm-started dual simplex for node re-solves** (this is why Phase 3 Track A matters)
5. Presolve at root; cut generation (Gomory mixed-integer cuts) as stretch

Metrics: MIP gap, nodes/sec, time-to-first-incumbent, time-to-1% gap (don't promise proven optimality — industrial MILP practice is "good gap in practical time").

### Phase 8 — QP (Weeks 30+)

Scope decision upfront: **convex QP only.** Path: interior-point with the same KKT machinery, or an active-set method reusing simplex infrastructure. QPLIB convex subset + portfolio-optimization-shaped industrial case study.

### Phase 9+ — Stretch / Future

MIQP (reuse B&B + QP relaxations), NLP (SQP — likely out of scope, keep architecture hooks only), parallel B&B, advanced cuts, adaptive `--device auto` policy trained on your own benchmark corpus.

---

### Master Pipeline (the diagram to keep)

```
 MPS/LP file
     │
     ▼
 Ingest + validate
     │
     ▼
 Presolve ──► Scale ──► Characterize (size/nnz/structure/conditioning)
     │                        │
     │                        ▼
     │              ┌── Execution policy ──┐
     ▼              ▼                      ▼
   CPU track                             GPU track
   (simplex / IPM)              (first-order + polish)
     │              └──────────┬───────────┘
     ▼                         ▼
            Solver engine
                  │
                  ▼
         Independent verification (KKT / residuals / gap)
                  │
        ┌─── fail ─┤ pass
        ▼          ▼
   Recovery      Benchmark log → metrics → plots → policy update
   (rescale,
    re-solve,
    alt path)
```

---

## Part 3 — Claims Mapping (the five claims, and what proves each)

| Claim | Proof artifact |
|---|---|
| A — Sovereignty | Repo, from-scratch code, no solver linking, license note, ability to demo a custom presolve rule added live |
| B — Correctness | KKT verification layer + 0 discrepancies vs HiGHS on Netlib subset + failure report |
| C — Scalability | Runtime/memory scaling curves to ≥1M variables on staircase-structured instances |
| D — Industrial relevance | Industrial benchmark suite with generators, formulation docs, structural analysis |
| E — Performance | Crossover map + speedup table on defined regime + honest loss table on small/degenerate problems |

---

## Three sentences to internalize

1. **Your MVP is not "an LP solver with GPU" — it's a measured answer to "which LP algorithm on which hardware, and where's the crossover."**
2. **Presolve and scaling belong in the LP phase; they are your numerical-robustness story, not an afterthought.**
3. **The industrial benchmark suite is the sovereignty claim made concrete — treat it as a first-class deliverable, not an appendix.**

If you want, I can next draft the Phase 3 technical design (simplex architecture + the PDHG GPU formulation in detail), or the industrial benchmark generator specs — those are the two highest-uncertainty components.
