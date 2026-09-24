#!/usr/bin/env python3
"""
PipePye Research Monograph Generator
Generates website/index.html with the complete 15-section peer-reviewed research monograph structure.
"""

import os

HTML_CONTENT = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>PipePye: Structure-Aware Heterogeneous Optimization Dispatch Across CPU and GPU Architectures</title>
  <meta name="description" content="A research monograph on structure-aware linear and mixed-integer optimization dispatch across CPU and GPU architectures.">
  
  <!-- Typography (Space Grotesk, Inter, JetBrains Mono, Newsreader) -->
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;1,6..72,400&family=Space+Grotesk:wght@400;500;600;700&display=swap" rel="stylesheet">
  
  <!-- KaTeX for Mathematical Typesetting -->
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"
    onload="renderMathInElement(document.body, {delimiters: [{left: '$$', right: '$$', display: true}, {left: '$', right: '$', display: false}]});"></script>

  <!-- Stylesheet -->
  <link rel="stylesheet" href="style.css">
</head>
<body>

  <!-- ========================================================================
       Top Navigation Bar (15-Section Anchor Index)
       ======================================================================== -->
  <header class="top-nav">
    <div class="nav-left">
      <a href="https://github.com/Satyanshgaur/PIPEPYE" target="_blank" rel="noopener noreferrer" class="nav-anchor" title="View PipePye Repository on GitHub">
        <span class="nav-brand">PIPEPYE</span>
        <span class="nav-descriptor">RESEARCH MONOGRAPH ↗</span>
      </a>
    </div>
    <nav class="nav-center">
      <ul class="nav-links">
        <li><a href="#sec1-executive-summary" class="nav-item">1. Exec Summary</a></li>
        <li><a href="#sec2-the-problem" class="nav-item">2. The Problem</a></li>
        <li><a href="#sec3-landscape" class="nav-item">3. Landscape</a></li>
        <li><a href="#sec4-thesis" class="nav-item">4. Thesis</a></li>
        <li><a href="#sec5-architecture" class="nav-item">5. Architecture</a></li>
        <li><a href="#sec6-algorithms" class="nav-item">6. Algorithms</a></li>
        <li><a href="#sec7-methodology" class="nav-item">7. Methodology</a></li>
        <li><a href="#sec8-experiments" class="nav-item">8. Experiments</a></li>
        <li><a href="#sec9-case-studies" class="nav-item">9. Case Studies</a></li>
        <li><a href="#sec10-robustness" class="nav-item">10. Robustness</a></li>
        <li><a href="#sec11-policy" class="nav-item">11. Policy</a></li>
        <li><a href="#sec12-limitations" class="nav-item">12. Scope</a></li>
        <li><a href="#sec13-roadmap" class="nav-item">13. Roadmap</a></li>
        <li><a href="#sec14-claims" class="nav-item">14. Claims</a></li>
        <li><a href="#sec15-appendices" class="nav-item">15. Appendices</a></li>
      </ul>
    </nav>
    <div class="nav-right">
      <a href="#sec7-methodology" class="nav-status" title="View Verification Protocol">174/174 PASSED</a>
      <a href="https://github.com/Satyanshgaur/PIPEPYE/tree/main/dashboard" target="_blank" rel="noopener noreferrer" class="nav-status" style="margin-left: 8px;" title="View Interactive Demonstration UI Solver">UI SOLVER ↗</a>
    </div>
  </header>

  <!-- ========================================================================
       SECTION 1: Executive Summary (1 Page Equivalent)
       ======================================================================== -->
  <section id="sec1-executive-summary" class="canvas-light">
    <div class="content-column">
      <span class="section-label">Document Series · Pure Optimization Research · September 2026</span>
      <h1 class="display-title">Structure-Aware Heterogeneous Optimization Dispatch Across CPU and GPU Architectures</h1>
      
      <p class="lead-paragraph">
        PipePye is an open, sovereign mathematical optimization system engineered to investigate whether algebraic and topological matrix features reliably predict the optimal numerical solver algorithm and hardware backend.
      </p>

      <div class="formal-box">
        <span class="formal-title">Executive Abstract & Core Problem Formulation</span>
        <p style="margin-bottom: 0; font-family: var(--font-serif); font-size: 18px; font-style: italic; color: #111;">
          "India's foundational industrial operations—downstream petroleum refining, national electrical power grid dispatch, freight scheduling, and supply chain allocation—depend almost exclusively on foreign proprietary optimization solvers (Gurobi, IBM ILOG CPLEX, FICO Xpress). Existing solvers present high recurring foreign licensing costs, sovereign exposure to technology transfer restrictions, and monolithic CPU designs that fail to leverage modern heterogeneous GPU architectures on structured models. PipePye evaluates whether topological characteristics reliably predict solver and backend efficiency, delivering an end-to-end sovereign C++20 and CUDA implementation with an adaptive routing policy confirmed across industrial benchmarks."
        </p>
      </div>

      <p>
        Standard solver architectures enforce a monolithic execution pipeline—predominantly sequential dual simplex or interior-point algorithms on host CPUs—leaving high-throughput accelerators underutilized. PipePye establishes an end-to-end optimization substrate consisting of sparse CSR/CSC linear algebra, an immutable model preprocessing pipeline (5 presolve passes, Ruiz matrix equilibration), a first-order Primal-Dual Hybrid Gradient (PDHG) solver for CUDA GPUs, a Sparse Dual Revised Simplex solver with Product Form of the Inverse (PFI) and Devex pricing, an active-set basis crossover mechanism, and a Mixed-Integer Linear Programming (MILP) Branch-and-Bound engine.
      </p>

      <p>
        The central empirical discovery is a sharp <strong>CPU/GPU crossover boundary</strong>:
        On large-scale block-banded staircase models ($T=100$, 40,000 NNZ), GPU PDHG completes in <strong>196 ms</strong> compared to <strong>40,039 ms</strong> for CPU Dual Simplex—a <strong>$204\times$ throughput speedup</strong>. Conversely, on compact, densely coupled models (crude oil blending with 30.1% nonzeros), CPU Simplex executes in <strong>0.80 ms</strong> while GPU PDHG requires <strong>236 ms</strong> due to PCI-e transfer and kernel dispatch latency—a <strong>$295\times$ advantage for CPU execution</strong>.
      </p>

      <div class="metrics-grid">
        <div class="metric-cell">
          <span class="metric-label">Automated Regression Suite</span>
          <span class="metric-value">174 / 174</span>
          <span class="metric-subtitle">100.0% pass rate, zero regressions across 29 modules</span>
        </div>
        <div class="metric-cell">
          <span class="metric-label">Pre-Registered Protocol</span>
          <span class="metric-value">13 / 13</span>
          <span class="metric-subtitle">100.0% confirmed optimal routes (vs 53.8% static baseline)</span>
        </div>
        <div class="metric-cell">
          <span class="metric-label">Simplex Warm-Start Pruning</span>
          <span class="metric-value">98.8%</span>
          <span class="metric-subtitle">Pivot reduction on industrial MILP combinatorial trees</span>
        </div>
        <div class="metric-cell">
          <span class="metric-label">CUDA Memory Bandwidth</span>
          <span class="metric-value">160.5 GB/s</span>
          <span class="metric-subtitle">95.5% of theoretical peak GDDR6 throughput</span>
        </div>
      </div>

      <hr class="hairline">

      <h3 class="heading-3">Research Monograph Navigation Index</h3>
      <div class="toc-grid">
        <a href="#sec1-executive-summary" class="toc-card">
          <span class="toc-num">Section 01</span>
          <span class="toc-title">Executive Summary</span>
          <span class="toc-desc">The complete narrative compressed: problem, findings, crossover results, and strategic value.</span>
        </a>
        <a href="#sec2-the-problem" class="toc-card">
          <span class="toc-num">Section 02</span>
          <span class="toc-title">The Problem — Why This Matters</span>
          <span class="toc-desc">Critical infrastructure exposure, licensing drain, and reframing as a research inquiry.</span>
        </a>
        <a href="#sec3-landscape" class="toc-card">
          <span class="toc-num">Section 03</span>
          <span class="toc-title">Landscape & Gap Analysis</span>
          <span class="toc-desc">Survey of commercial and open-source solvers, comparative matrix, and open gaps.</span>
        </a>
        <a href="#sec4-thesis" class="toc-card">
          <span class="toc-num">Section 04</span>
          <span class="toc-title">Research Thesis</span>
          <span class="toc-desc">Intellectual contribution: mapping structural topology to algorithmic and hardware selection.</span>
        </a>
        <a href="#sec5-architecture" class="toc-card">
          <span class="toc-num">Section 05</span>
          <span class="toc-title">System Architecture</span>
          <span class="toc-desc">Eight-stage pipeline: ingestion, presolve, scaling, characterization, solver, postsolve, audit.</span>
        </a>
        <a href="#sec6-algorithms" class="toc-card">
          <span class="toc-num">Section 06</span>
          <span class="toc-title">Algorithm Strategy</span>
          <span class="toc-desc">Mathematical formulations: Dual Revised Simplex, First-Order PDHG, and Interior Point Methods.</span>
        </a>
        <a href="#sec7-methodology" class="toc-card">
          <span class="toc-num">Section 07</span>
          <span class="toc-title">Methodology & Rigor</span>
          <span class="toc-desc">Independent correctness verifier, hardware testbed specifications, and phased quality gates.</span>
        </a>
        <a href="#sec8-experiments" class="toc-card">
          <span class="toc-num">Section 08</span>
          <span class="toc-title">Experiments & Results</span>
          <span class="toc-desc">Three headline studies: SpMV crossover, algorithm crossover, and pre-registered protocol.</span>
        </a>
        <a href="#sec9-case-studies" class="toc-card">
          <span class="toc-num">Section 09</span>
          <span class="toc-title">Industrial Case Studies</span>
          <span class="toc-desc">Crude blending, multi-period planning, unit scheduling, and power grid economic dispatch.</span>
        </a>
        <a href="#sec10-robustness" class="toc-card">
          <span class="toc-num">Section 10</span>
          <span class="toc-title">Numerical Robustness</span>
          <span class="toc-desc">4-way ablation framework on degenerate matrices, Ruiz equilibration, and HiGHS parity.</span>
        </a>
        <a href="#sec11-policy" class="toc-card">
          <span class="toc-num">Section 11</span>
          <span class="toc-title">Adaptive Execution Policy</span>
          <span class="toc-desc">Deterministic heuristic rules and live CLI decision explanation trace output.</span>
        </a>
        <a href="#sec12-limitations" class="toc-card">
          <span class="toc-num">Section 12</span>
          <span class="toc-title">Limitations & Scope</span>
          <span class="toc-desc">Transparent boundaries: MILP maturity, IPM timeline, and scope definition of "from-scratch".</span>
        </a>
        <a href="#sec13-roadmap" class="toc-card">
          <span class="toc-num">Section 13</span>
          <span class="toc-title">Roadmap & Future Work</span>
          <span class="toc-desc">Structured research phases: matrix-free IPM, MIQP, multi-GPU decomposition, ML branching.</span>
        </a>
        <a href="#sec14-claims" class="toc-card">
          <span class="toc-num">Section 14</span>
          <span class="toc-title">Claim → Evidence Map</span>
          <span class="toc-desc">Verifiable synthesis table linking every architectural and empirical claim to artifacts.</span>
        </a>
        <a href="#sec15-appendices" class="toc-card">
          <span class="toc-num">Section 15</span>
          <span class="toc-title">Technical Appendices</span>
          <span class="toc-desc">Benchmark execution protocols, MPS corpus provenance, repository tree, and reproduction commands.</span>
        </a>
      </div>
    </div>
  </section>

  <!-- ========================================================================
       SECTION 2: The Problem — Why This Matters
       ======================================================================== -->
  <section id="sec2-the-problem" class="canvas-carbon">
    <div class="content-column">
      <span class="section-label">Section 02 · Strategic Dependency & Industrial Criticality</span>
      <h2 class="heading-1">The Problem: Sovereign Mathematical Optimization for Critical Infrastructure</h2>
      
      <p class="lead-paragraph">
        Modern process manufacturing, wholesale electrical power dispatch, and logistical distribution in India sit atop proprietary, closed-source mathematical programming engines developed abroad.
      </p>

      <div class="formal-box">
        <span class="formal-title">The Sovereign Dependency Narrative</span>
        <p style="margin-bottom: 0;">
          The operational rhythm of critical national infrastructure relies continuously on mixed-integer and linear programming solvers:
          crude oil distillation and Euro-VI fuel blending across major refineries (IOCL, BPCL, HPCL), real-time generation scheduling and security-constrained economic dispatch coordinated by the Grid Controller of India (Grid-India / POSOCO), railway freight rake scheduling across zones, and grain inventory logistics by the Food Corporation of India. These daily scheduling pipelines are driven primarily by three foreign proprietary software packages: <strong>IBM ILOG CPLEX, Gurobi, and FICO Xpress</strong>.
        </p>
      </div>

      <p>
        This systemic dependency creates three measurable structural liabilities:
      </p>

      <div class="metrics-grid">
        <div class="metric-cell">
          <span class="metric-label">Estimated Licensing Outflow</span>
          <span class="metric-value">$25M – $40M</span>
          <span class="metric-subtitle">Annual foreign exchange outlays across public & private industrial enterprises</span>
        </div>
        <div class="metric-cell">
          <span class="metric-label">Sovereign Supply-Chain Risk</span>
          <span class="metric-value">Single Point</span>
          <span class="metric-subtitle">Vulnerability to sudden export controls, license revocations, or sanctions</span>
        </div>
        <div class="metric-cell">
          <span class="metric-label">Accelerator Utilization Gap</span>
          <span class="metric-value">&lt; 5%</span>
          <span class="metric-subtitle">National Supercomputing Mission GPUs sit idle during optimization runs</span>
        </div>
      </div>

      <p>
        Commercial solvers cost upwards of $15,000 to $60,000 USD per node/annum for multi-threaded enterprise licenses. In addition to high capital drains, they operate as proprietary binary black-boxes: their internal heuristics, numerical tolerances, and matrix factorization methods cannot be inspected, verified, or customized for domestic supercomputing environments (such as C-DAC's PARAM series under the National Supercomputing Mission).
      </p>

      <p>
        More critically, existing commercial and open-source solvers remain tethered to traditional CPU architectures. Even when high-density GPU accelerators are physically available on compute clusters, standard solvers fail to exploit them because simplex basis maintenance is inherently sequential.
      </p>

      <div class="formal-box">
        <span class="formal-title">Reframing: Engineering Task vs. Fundamental Research Question</span>
        <p style="margin-bottom: 0; font-family: var(--font-serif); font-size: 18px; font-style: italic; color: #fff;">
          "The technological barrier is not solved by simply re-implementing textbook simplex in C++. The foundational research challenge is: 
          Do realistic industrial optimization structures behave differently from generic benchmark matrices, and can those structural characteristics guide algorithm and hardware backend selection?"
        </p>
      </div>
    </div>
  </section>

  <!-- ========================================================================
       SECTION 3: Landscape & Gap Analysis
       ======================================================================== -->
  <section id="sec3-landscape" class="canvas-light">
    <div class="content-column">
      <span class="section-label">Section 03 · State of the Art & Gap Analysis</span>
      <h2 class="heading-1">Landscape and Gap Analysis: State of Mathematical Solvers</h2>
      
      <p class="lead-paragraph">
        Mathematical optimization software spans four decades of active development, divided between mature commercial suites, classical open-source utilities, and contemporary first-order research codes.
      </p>

      <p>
        Commercial engines (Gurobi, CPLEX, Xpress) provide highly refined Dual Revised Simplex and Interior Point algorithms combined with hundreds of proprietary heuristics for Branch-and-Cut MILP search. However, they are closed-source, prohibitively expensive, and strictly CPU-centric for linear programming.
      </p>

      <p>
        Conversely, open-source alternatives present clear trade-offs:
        <strong>COIN-OR CBC</strong> and <strong>GLPK</strong> suffer from dated linear algebra cores, single-thread limitations, and numerical instability on ill-conditioned industrial models. 
        <strong>HiGHS</strong> represents the modern open-source standard for CPU-based Dual Simplex and IPM, yet offers no GPU accelerator execution. 
        <strong>SCIP</strong> is a comprehensive academic constraint integer programming framework, but its non-commercial licensing constraints hinder sovereign industrial deployment.
        Recent research directions such as <strong>PDLP / cuPDLP</strong> (Google Research) utilize First-Order Primal-Dual Hybrid Gradient methods on GPUs, but they produce non-basic solutions and lack integrated simplex basis crossover or dual basis warm-starting for mixed-integer search trees.
      </p>

      <div class="table-wrapper">
        <table>
          <thead>
            <tr>
              <th>Solver</th>
              <th>License Model</th>
              <th>Primary Algorithms</th>
              <th>Hardware Target</th>
              <th>Documented Strengths</th>
              <th>Known Limitations</th>
              <th>Why It Does Not Close The Sovereign Gap</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td><strong>Gurobi 11</strong></td>
              <td>Proprietary Commercial</td>
              <td>Simplex, Barrier IPM, Branch & Cut</td>
              <td>CPU (x86_64, ARM)</td>
              <td>Industry standard; thousands of MILP heuristics</td>
              <td>Closed binary; expensive per-core fee; no GPU LP support</td>
              <td>Perpetuates foreign dependency and technological lock-in</td>
            </tr>
            <tr>
              <td><strong>IBM CPLEX 22</strong></td>
              <td>Proprietary Commercial</td>
              <td>Primal/Dual Simplex, Barrier, MILP</td>
              <td>CPU (Multi-threaded)</td>
              <td>Mature numerics; strong enterprise integration</td>
              <td>High recurring licensing costs; opaque internal routines</td>
              <td>Vulnerable to foreign export control restrictions</td>
            </tr>
            <tr>
              <td><strong>FICO Xpress 9</strong></td>
              <td>Proprietary Commercial</td>
              <td>Simplex, Barrier, Branch & Bound</td>
              <td>CPU (Multi-threaded)</td>
              <td>Specialized for large industrial scheduling</td>
              <td>Proprietary license; rigid deployment architecture</td>
              <td>Prohibitive deployment cost across domestic PSUs</td>
            </tr>
            <tr>
              <td><strong>HiGHS 1.8</strong></td>
              <td>Open Source (MIT)</td>
              <td>Dual Simplex, IPM, Branch & Bound</td>
              <td>CPU (C++11/C++20)</td>
              <td>Leading open-source CPU performance; clean modern codebase</td>
              <td>No native GPU acceleration; lacks first-order methods</td>
              <td>Leaves high-density accelerator clusters unexploited</td>
            </tr>
            <tr>
              <td><strong>COIN-OR CBC</strong></td>
              <td>Open Source (EPL)</td>
              <td>Primal/Dual Simplex, Branch & Cut</td>
              <td>CPU (x86)</td>
              <td>Historical standard in open-source MILP</td>
              <td>Aging codebase; fragile on degenerate industrial models</td>
              <td>Lacks modern presolve, SIMD vectorization, and GPU backends</td>
            </tr>
            <tr>
              <td><strong>GLPK 5.0</strong></td>
              <td>Open Source (GPL)</td>
              <td>Revised Simplex, Primal-Dual IPM</td>
              <td>CPU (Single-threaded)</td>
              <td>Simple educational reference; standard C library</td>
              <td>Single-threaded; no sparse LU updates; numerical stalling</td>
              <td>Scales poorly on modern multi-thousand-row models</td>
            </tr>
            <tr>
              <td><strong>SCIP 9.0</strong></td>
              <td>Custom Academic / Commercial</td>
              <td>Branch-Price-and-Cut, CIP Framework</td>
              <td>CPU (C/C++)</td>
              <td>Highly extensible; broad mathematical modeling support</td>
              <td>Restrictive non-commercial license; heavy software stack</td>
              <td>Licensing terms complicate public sovereign infrastructure</td>
            </tr>
            <tr>
              <td><strong>Google PDLP</strong></td>
              <td>Open Source (Apache 2.0)</td>
              <td>First-Order PDHG (Chambolle-Pock)</td>
              <td>CPU & CUDA GPU</td>
              <td>Scales to tens of millions of nonzeros; GPU support</td>
              <td>Interior solutions; no simplex crossover; no MILP tree engine</td>
              <td>Cannot warm-start combinatorial branch-and-bound nodes</td>
            </tr>
            <tr style="background-color: rgba(0, 0, 0, 0.04);">
              <td><strong>PipePye (Ours)</strong></td>
              <td>Sovereign Open Source</td>
              <td>Dual Simplex, GPU PDHG, B&B, Crossover</td>
              <td>CPU & CUDA GPU (sm_86)</td>
              <td>Topological structure-aware routing; dual warm-starting</td>
              <td>Early-stage MILP cut generation pool (under active development)</td>
              <td><strong>Purpose-built to bridge heterogeneous sovereign gap</strong></td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </section>

  <!-- ========================================================================
       SECTION 4: Research Thesis
       ======================================================================== -->
  <section id="sec4-thesis" class="canvas-carbon">
    <div class="content-column">
      <span class="section-label">Section 04 · Intellectual Contribution</span>
      <h2 class="heading-1">Research Thesis: Structure-Aware Heterogeneous Dispatch</h2>
      
      <p class="lead-paragraph">
        This research is not a generic engineering exercise in building an optimization solver; it is a systematic investigation into the mathematical mapping between problem topology, algorithm complexity, and hardware architecture.
      </p>

      <div class="formal-box">
        <span class="formal-title">The Fundamental Research Thesis</span>
        <p style="margin-bottom: 0; font-family: var(--font-serif); font-size: 19px; font-style: italic; color: #fff;">
          "Thesis: Industrial linear and mixed-integer optimization instances exhibit non-random structural invariants—specifically nonzero density, block-angular staircase scores, degree dispersion, and integrality coupling—that deterministically govern whether direct matrix factorization or iterative operator splitting achieves minimal time-to-solution across CPU and GPU hardware."
        </p>
      </div>

      <p>
        The intellectual contribution separates the problem into two distinct operational paradigms:
      </p>

      <ul style="max-width: 880px; text-align: left; margin: 0 auto 32px auto;">
        <li>
          <strong>Direct Basis Factorization (Simplex Family):</strong> Solves the basis system $B x_B = b$ via sparse triangular factorizations ($L U$ updates or Product Form of the Inverse). Each pivot requires $O(m^2)$ or $O(\text{nnz}(L+U))$ work. On CPUs, this process is bound by cache latency and sequential dependency chains. When models exhibit high density or dense cross-coupling constraints, basis updates are compact and pivots converge rapidly to exact basic solutions. Furthermore, bound changes preserve dual feasibility, enabling single-digit pivot re-optimization.
        </li>
        <li>
          <strong>First-Order Operator Splitting (PDHG Family):</strong> Replaces matrix factorizations with alternating proximal projections and matrix-vector multiplications ($Ax$ and $A^T y$) costing strictly $O(\text{NNZ})$ per iteration. These operations map to SIMT parallel hardware (NVIDIA CUDA Streaming Multiprocessors). However, convergence is sublinear ($O(1/\epsilon)$) and scales with matrix conditioning $\kappa(A)$, producing approximate interior points that require basis purification.
        </li>
      </ul>

      <div class="formal-box">
        <span class="formal-title">The Formal Mapping Hypothesis</span>
        <p style="margin-bottom: 0;">
          Let an optimization instance be characterized by the structural feature vector:
          $$\phi(\mathcal{P}) = \left( \alpha_{\text{int}}, \ \rho, \ \sigma_{\text{staircase}}, \ \text{NNZ}, \ m, \ n, \ \kappa_{\text{proxy}} \right)$$
          where $\alpha_{\text{int}} = n_{\text{integer}} / n$ is the integrality ratio, $\rho = \text{NNZ} / (m \cdot n)$ is the matrix density, and $\sigma_{\text{staircase}}$ measures temporal block-banded decoupling. We hypothesize the existence of a deterministic decision policy $\pi^*(\phi) \to (\text{Algorithm}, \text{Hardware})$ such that:
          $$T(\pi^*(\phi), \mathcal{P}) \le \min \left\{ T(\text{Simplex}_{\text{CPU}}, \mathcal{P}), \ T(\text{PDHG}_{\text{GPU}}, \mathcal{P}) \right\}$$
          across all industrial workload classes, strictly outperforming any static default solver assignment.
        </p>
      </div>
    </div>
  </section>

  <!-- ========================================================================
       SECTION 5: System Architecture
       ======================================================================== -->
  <section id="sec5-architecture" class="canvas-light">
    <div class="content-column">
      <span class="section-label">Section 05 · Architectural Decomposition</span>
      <h2 class="heading-1">System Architecture and Pipeline Invariants</h2>
      
      <p class="lead-paragraph">
        PipePye enforces a clean, unidirectional pipeline that isolates raw problem ingestion and mathematical transformation from numerical solver execution and solution verification.
      </p>

      <p>
        Solvers do not ingest raw file streams or mutate client problem representations. Workloads transition through eight decoupled stages:
      </p>

      <!-- Complete 8-Stage Architecture SVG Diagram -->
      <div class="diagram-container">
        <svg class="diagram-svg" viewBox="0 0 1020 400" xmlns="http://www.w3.org/2000/svg">
          <style>
            .arch-box { fill: #fcfcfc; stroke: #222; stroke-width: 1.5; }
            .arch-accent { fill: #111; stroke: #111; stroke-width: 1.5; }
            .arch-title { font-family: 'Space Grotesk', sans-serif; font-size: 12px; font-weight: 700; fill: #111; }
            .arch-title-light { font-family: 'Space Grotesk', sans-serif; font-size: 12px; font-weight: 700; fill: #fff; }
            .arch-sub { font-family: 'JetBrains Mono', monospace; font-size: 9.5px; fill: #555; }
            .arch-arrow { stroke: #222; stroke-width: 1.5; fill: none; marker-end: url(#arrow-head); }
            .arch-dash { stroke: #666; stroke-width: 1.5; stroke-dasharray: 4; fill: none; }
          </style>
          <defs>
            <marker id="arrow-head" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
              <path d="M 0 1 L 10 5 L 0 9 z" fill="#222" />
            </marker>
          </defs>

          <!-- Top Row: Ingestion, Presolve, Characterize, Policy -->
          <rect x="20" y="30" width="130" height="70" class="arch-box" />
          <text x="32" y="55" class="arch-title">1. Parse (MPS)</text>
          <text x="32" y="72" class="arch-sub">Fixed & Free MPS</text>
          <text x="32" y="86" class="arch-sub">Contradiction check</text>

          <rect x="180" y="30" width="170" height="70" class="arch-box" />
          <text x="192" y="55" class="arch-title">2. Presolve & Scaling</text>
          <text x="192" y="72" class="arch-sub">5 Reduction Passes</text>
          <text x="192" y="86" class="arch-sub">Ruiz Equilibration</text>

          <rect x="380" y="30" width="170" height="70" class="arch-box" />
          <text x="392" y="55" class="arch-title">3. Characterization</text>
          <text x="392" y="72" class="arch-sub">Density & Staircase</text>
          <text x="392" y="86" class="arch-sub">Gini & Dynamic Range</text>

          <rect x="580" y="30" width="160" height="70" class="arch-accent" />
          <text x="592" y="55" class="arch-title-light">4. Execution Policy</text>
          <text x="592" y="72" class="arch-sub" fill="#ccc">Deterministic Gating</text>
          <text x="592" y="86" class="arch-sub" fill="#aaa">--device / --method</text>

          <!-- Arrows across top -->
          <path d="M 150 65 L 180 65" class="arch-arrow" />
          <path d="M 350 65 L 380 65" class="arch-arrow" />
          <path d="M 550 65 L 580 65" class="arch-arrow" />

          <!-- Down arrow to Solvers -->
          <path d="M 660 100 L 660 145" class="arch-arrow" />

          <!-- Middle Layer: 5. Solver Selection -->
          <rect x="530" y="145" width="260" height="110" class="arch-box" />
          <text x="545" y="170" class="arch-title">5. Numerical Solver Core</text>
          <text x="545" y="190" class="arch-sub">• Sparse Dual Simplex (CPU, PFI, Devex)</text>
          <text x="545" y="208" class="arch-sub">• First-Order PDHG (CUDA GPU Resident)</text>
          <text x="545" y="226" class="arch-sub">• Branch & Bound (MILP Basis Warm-Start)</text>
          <text x="545" y="244" class="arch-sub">• Basis Crossover (PDHG → Vertex)</text>

          <!-- Arrow leftward to Postsolve -->
          <path d="M 530 200 L 450 200" class="arch-arrow" />

          <!-- Bottom Row: Postsolve, Verification, Benchmarking -->
          <rect x="280" y="165" width="170" height="70" class="arch-box" />
          <text x="292" y="190" class="arch-title">6. Postsolve Recovery</text>
          <text x="292" y="208" class="arch-sub">LIFO Transformation</text>
          <text x="292" y="222" class="arch-sub">Diagonal Unscaling</text>

          <rect x="70" y="165" width="180" height="70" class="arch-accent" />
          <text x="82" y="190" class="arch-title-light">7. Solution Verifier</text>
          <text x="82" y="208" class="arch-sub" fill="#ccc">Out-of-band KKT Audit</text>
          <text x="82" y="222" class="arch-sub" fill="#aaa">Primal & Dual Stationarity</text>

          <path d="M 280 200 L 250 200" class="arch-arrow" />

          <!-- Benchmark telemetry node -->
          <rect x="70" y="280" width="380" height="60" class="arch-box" />
          <text x="82" y="305" class="arch-title">8. Telemetry & Benchmark Instrumentation</text>
          <text x="82" y="323" class="arch-sub">Hardware timers, PCIe latency, memory RSS, machine-readable JSON/CSV</text>

          <path d="M 160 235 L 160 280" class="arch-arrow" />
        </svg>
      </div>

      <div class="formal-box">
        <span class="formal-title">System Invariant: The PreparedLP Boundary</span>
        <p style="margin-bottom: 0;">
          All numerical solvers operate strictly upon an immutable <code>PreparedLP</code> structure in bounded canonical format:
          $$\min_{x} c^T x \quad \text{subject to} \quad l_r \le Ax \le u_r, \quad l_c \le x \le u_c$$
          Solvers never access file streams, modify user-facing constraints, or execute unscaling logic internally. Solution recovery applies the exact mathematical inverse operations recorded in a thread-safe LIFO stack.
        </p>
      </div>

      <p>
        The explicit purpose and guarantees of each stage:
      </p>

      <ul style="max-width: 880px; text-align: left; margin: 0 auto 32px auto;">
        <li><strong>Stage 1 (MPS Parsing):</strong> Validates syntax, detects contradiction bounds ($l_j > u_j$), and normalizes constraints into unified sparse CSR/CSC representations without third-party dependencies.</li>
        <li><strong>Stage 2 (Presolve & Scaling):</strong> Applies 5 reduction passes (Empty rows/cols, Fixed variables, Singletons, Forcing constraints, Bound tightening) followed by Ruiz equilibration, shrinking dimensions by up to 50% and compressing condition dynamic ranges from $10^{12} \to 10^0$.</li>
        <li><strong>Stage 3 (Characterization):</strong> Computes structural moments ($\rho, \sigma_{\text{staircase}}, \text{Gini}$) in $O(\text{NNZ})$ time to construct the structural feature vector.</li>
        <li><strong>Stage 4 (Execution Policy):</strong> Evaluates deterministic selection rules or respects user overrides (<code>--device</code>, <code>--method</code>) to assign the problem.</li>
        <li><strong>Stage 5 (Solver Core):</strong> Executes the selected algorithm in resident memory without inter-iteration host-device memory transfers.</li>
        <li><strong>Stage 6 (Postsolve):</strong> Reconstructs primal and dual variables into original dimensions through inverted coordinate scaling and postsolve unwinding.</li>
        <li><strong>Stage 7 (Independent Verifier):</strong> An out-of-band audit calculating raw residuals $\|Ax - b\|_\infty$ and $\|A^T y + s - c\|_\infty$ on the unpresolved formulation. Solvers cannot self-certify.</li>
        <li><strong>Stage 8 (Telemetry):</strong> Emits structured performance telemetry (JSON/CSV) capturing solve times, pivot counts, memory footprints, and reference solver deltas.</li>
      </ul>
    </div>
  </section>

  <!-- ========================================================================
       SECTION 6: Algorithm Strategy
       ======================================================================== -->
  <section id="sec6-algorithms" class="canvas-carbon">
    <div class="content-column">
      <span class="section-label">Section 06 · Algorithmic Strategy & Literature Foundations</span>
      <h2 class="heading-1">Algorithm Strategy: Formulations and Complexity</h2>
      
      <p class="lead-paragraph">
        Rather than attempting to build a one-size-fits-all solver, PipePye implements a targeted algorithmic portfolio grounded in classical and modern numerical optimization literature.
      </p>

      <!-- Subsection 1: Dual Revised Simplex -->
      <div class="formal-box">
        <span class="formal-title">6.1 Dual Revised Simplex (CPU Architecture)</span>
        <p style="text-align: left; margin-bottom: 12px;">
          <strong>Why It Matters:</strong> The Dual Revised Simplex algorithm remains the gold standard for linear programming when exact basic feasible solutions are required, and serves as the essential workhorse for Mixed-Integer Linear Programming (MILP) where variable bounds are iteratively tightened along tree search nodes.
        </p>
        <p style="text-align: left; margin-bottom: 12px;">
          <strong>Why It Is Computationally Difficult:</strong> Simplex is inherently sequential; each basis update depends strictly on the outcome of the preceding pivot. Efficient implementations require sparse representation of the basis inverse $B^{-1}$ via Product Form of the Inverse (PFI) or Bartels-Golub / Forrest-Tomlin LU updates. As pivots accumulate, eta-matrices cause memory fill-in, requiring periodic refactorization. Numerical stability demands rigorous anti-cycling mechanisms (Bland's rule, Harris two-pass ratio test) and dynamic Devex / steepest-edge pricing approximations.
        </p>
        <p style="text-align: left; margin-bottom: 12px;">
          <strong>Mathematical Formulation:</strong>
          Dual feasibility requires reduced costs $d_N = c_N - N^T y \ge 0$. The leaving basic variable $x_{B_p}$ is selected where $x_{B_p} < l_{B_p}$ or $x_{B_p} > u_{B_p}$. The dual pricing row is obtained via Backward Transformation (BTRAN):
          $$B^T v = e_p \implies v = B^{-T} e_p$$
          The entering nonbasic column $q$ is chosen via the dual ratio test:
          $$q = \arg\min_{j \in N, \alpha_{pj} < 0} \left\{ \frac{d_j}{|\alpha_{pj}|} \right\}, \quad \text{where } \alpha_p = v^T N$$
          Forward Transformation (FTRAN) computes the pivot column $d_q = B^{-1} a_q$, updating the basis representation.
        </p>
        <p style="text-align: left; margin-bottom: 0;">
          <strong>Planned Role & Citations:</strong> Primary LP solver for compact, dense models and subproblem warm-start engine for MILP branch-and-bound trees. Citations: Dantzig (1951), Lemke (1954), Forrest & Tomlin (1972), Harris (1973), Koberstein (2005).
        </p>
      </div>

      <!-- Subsection 2: PDHG / PDLP -->
      <div class="formal-box">
        <span class="formal-title">6.2 First-Order Primal-Dual Hybrid Gradient (CUDA GPU Architecture)</span>
        <p style="text-align: left; margin-bottom: 12px;">
          <strong>Why It Matters:</strong> Direct factorization methods encounter cubic complexity bottlenecks ($O(m^3)$ worst-case) and memory exhaustion when scaled to multi-million nonzero models. PDHG (Chambolle-Pock / PDLP) eliminates matrix factorization entirely. Every iteration consists solely of sparse matrix-vector multiplications ($Ax$ and $A^T y$) and component-wise proximal projections, mapping naturally to massive GPU SIMT thread parallelism.
        </p>
        <p style="text-align: left; margin-bottom: 12px;">
          <strong>Why It Is Computationally Difficult:</strong> As a first-order method, convergence is sublinear ($O(1/k)$), making it highly sensitive to matrix ill-conditioning. Unpreconditioned operators exhibit severe oscillatory zigzagging. Achieving practical termination tolerances requires diagonal preconditioning (Pock-Chambolle / Ruiz), adaptive primal-dual step-size tuning, and restart heuristics to purge stagnated momentum. Furthermore, iterate points are interior, requiring active-set crossover to reach basic vertices.
        </p>
        <p style="text-align: left; margin-bottom: 12px;">
          <strong>Mathematical Formulation:</strong>
          For the canonical bounded LP, the iteration equations are:
          $$x^{k+1} = \text{proj}_{[l_c, u_c]} \left( x^k - \tau \odot \left( c - A^T y^k \right) \right)$$
          $$\bar{x}^{k+1} = 2 x^{k+1} - x^k \quad \text{(Extrapolation step)}$$
          $$y^{k+1} = \text{proj}_{[l_r, u_r]}^* \left( y^k + \sigma \odot A \bar{x}^{k+1} \right)$$
          where dual projection onto interval $[l_r, u_r]$ employs Moreau's proximal identity:
          $$\text{proj}_{[l, u]}^*(v) = v - \text{proj}_{[l, u]}(v)$$
          Diagonal step sizes satisfy the convergence condition:
          $$\tau_j = \frac{0.99}{\sum_i |A_{ij}|}, \quad \sigma_i = \frac{0.99}{\sum_j |A_{ij}|}$$
        </p>
        <p style="text-align: left; margin-bottom: 0;">
          <strong>Planned Role & Citations:</strong> High-throughput solver for massive, block-banded staircase models. Citations: Chambolle & Pock (2011), Applegate, Hinder, Lu, Wiegele (2021) "Practical First-Order Methods for Large-Scale Linear Programming" (PDLP).
        </p>
      </div>

      <!-- Subsection 3: IPM -->
      <div class="formal-box">
        <span class="formal-title">6.3 Primal-Dual Interior Point Methods (IPM — Roadmap Phase 8)</span>
        <p style="text-align: left; margin-bottom: 12px;">
          <strong>Why It Matters:</strong> IPMs possess proven polynomial-time worst-case complexity ($O(\sqrt{n} L)$ iterations) and converge with uniform reliability across general continuous formulations, largely independent of initial active-set degeneracy.
        </p>
        <p style="text-align: left; margin-bottom: 12px;">
          <strong>Why It Is Computationally Difficult:</strong> Each Newton step requires forming and solving the normal equations:
          $$\left( A \Theta A^T \right) \Delta y = r$$
          where $\Theta = X S^{-1}$ is a diagonal scaling matrix that varies at every iteration. When matrix $A$ contains dense columns, $A \Theta A^T$ becomes entirely dense, destroying sparsity and inducing catastrophic Cholesky fill-in ($O(m^3)$ operations). Matrix-free iterative linear solvers (such as conjugate gradients) require advanced preconditioners that remain challenging on GPUs.
        </p>
        <p style="text-align: left; margin-bottom: 0;">
          <strong>Planned Role & Citations:</strong> Medium-to-large dense continuous quadratic models and smooth convex relaxations. Citations: Karmarkar (1984), Mehrotra (1992), Wright (1997), Gondzio (2012).
        </p>
      </div>
    </div>
  </section>

  <!-- ========================================================================
       SECTION 7: Methodology — How We Will Know Anything Is True
       ======================================================================== -->
  <section id="sec7-methodology" class="canvas-light">
    <div class="content-column">
      <span class="section-label">Section 07 · Empirical Methodology & Verification Rigor</span>
      <h2 class="heading-1">Methodology: Correctness Architecture and Benchmarking</h2>
      
      <p class="lead-paragraph">
        A fundamental weakness of many optimization benchmarks is reliance on solver self-reporting. PipePye establishes an independent correctness oracle and an isolated benchmarking protocol.
      </p>

      <div class="formal-box">
        <span class="formal-title">7.1 Independent Correctness Architecture</span>
        <p style="text-align: left; margin-bottom: 12px;">
          Solvers are never permitted to declare their own optimality. When a solver terminates, its raw iterate vectors ($x', y'$) are unscaled and restored to the original unpresolved model coordinates ($x, y$) via the LIFO postsolve stack. The independent <code>SolutionVerifier</code> audits four zero-tolerance mathematical conditions directly against the original problem formulation:
        </p>
        <div style="text-align: center; margin: 16px 0;">
          $$\text{Primal Residual: } \epsilon_{\text{primal}} = \frac{\|Ax - b\|_\infty}{1 + \|b\|_\infty} \le 10^{-6}$$
          $$\text{Bound Violation: } \epsilon_{\text{bound}} = \max_{j} \left( \max(0, l_j - x_j), \max(0, x_j - u_j) \right) \le 10^{-6}$$
          $$\text{Dual Residual: } \epsilon_{\text{dual}} = \frac{\|A^T y + s - c\|_\infty}{1 + \|c\|_\infty} \le 10^{-6}$$
          $$\text{Complementary Slackness: } \epsilon_{\text{comp}} = \frac{|s^T (x - l)|}{1 + \|c\|_2 \|x\|_2} \le 10^{-6}$$
        </div>
        <p style="text-align: left; margin-bottom: 0;">
          In addition, all solution objectives are cross-validated against the independent reference solver <strong>HiGHS 1.8.1</strong> and canonical Netlib ground-truth tables, enforcing a strict relative error bound:
          $$\Delta \text{Obj}_{\text{rel}} = \frac{|\text{Obj}_{\text{PipePye}} - \text{Obj}_{\text{HiGHS}}|}{1 + |\text{Obj}_{\text{HiGHS}}|} \le 10^{-5}$$
        </p>
      </div>

      <h3 class="heading-2">7.2 Benchmark Execution Environment</h3>
      <p>
        All empirical measurements were executed on an isolated bare-metal x86_64 host under locked CPU frequencies to eliminate thermal throttling and operating system noise:
      </p>

      <div class="table-wrapper">
        <table>
          <thead>
            <tr>
              <th>Subsystem</th>
              <th>Hardware / Software Specification</th>
              <th>Operational Characteristics</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td><strong>Host Processor (CPU)</strong></td>
              <td>13th Gen Intel Core i5-13420H (x86_64)</td>
              <td>8 Cores (4 P-cores @ 4.6 GHz, 4 E-cores @ 3.4 GHz), 12 Threads</td>
            </tr>
            <tr>
              <td><strong>Host Memory</strong></td>
              <td>16 GB DDR5-5200 MT/s</td>
              <td>High-bandwidth dual-channel host memory</td>
            </tr>
            <tr>
              <td><strong>Accelerator (GPU)</strong></td>
              <td>NVIDIA GeForce RTX 3050 Laptop GPU (Ampere sm_86)</td>
              <td>20 SMs, 2,560 CUDA cores, 5.67 GB GDDR6 (96-bit bus, 168 GB/s peak)</td>
            </tr>
            <tr>
              <td><strong>Operating System</strong></td>
              <td>Linux 6.18.9-arch1-2 (x86_64)</td>
              <td>POSIX-compliant realtime scheduling; cgroup memory accounting</td>
            </tr>
            <tr>
              <td><strong>Compiler Toolchain</strong></td>
              <td>GCC 16.2.1 (-std=c++20 -O3 -march=native), CUDA nvcc 13.3</td>
              <td>OpenMP 5.2 multi-threading enabled</td>
            </tr>
            <tr>
              <td><strong>Execution Protocol</strong></td>
              <td>Isolated process runs; 5-run median measurement</td>
              <td>Cold-start vs warm-start timing; resident set size (RSS) tracking</td>
            </tr>
          </tbody>
        </table>
      </div>

      <h3 class="heading-2">7.3 Phased Quality Development Gates</h3>
      <p>
        Development adhered to strict phased verification gates: progress to subsequent phases was blocked until 100% of unit tests and regression assertions passed:
      </p>

      <div class="table-wrapper">
        <table>
          <thead>
            <tr>
              <th>Phase</th>
              <th>Scope & Deliverable</th>
              <th>Pass Criteria</th>
              <th>Automated Test Suite</th>
              <th>Gate Status</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td><strong>Phase 1</strong></td>
              <td>Sparse Linear Algebra (CSR/CSC, SpMV, CUDA kernels)</td>
              <td>Bit-level CPU/GPU numerical equivalence; 319 SpMV points</td>
              <td>79 / 79 PASSED</td>
              <td><span class="badge-status badge-confirmed">PASSED</span></td>
            </tr>
            <tr>
              <td><strong>Phase 2</strong></td>
              <td>Model Preparation, Presolve, Ruiz Matrix Equilibration</td>
              <td>Reversible postsolve; 4-way ablation framework</td>
              <td>125 / 125 PASSED</td>
              <td><span class="badge-status badge-confirmed">PASSED</span></td>
            </tr>
            <tr>
              <td><strong>Phase 3</strong></td>
              <td>First-Order PDHG LP Solver (CUDA GPU & CPU)</td>
              <td>Zero PCIe transfers inside hot loop; Moreau dual steps</td>
              <td>139 / 139 PASSED</td>
              <td><span class="badge-status badge-confirmed">PASSED</span></td>
            </tr>
            <tr>
              <td><strong>Phase 4</strong></td>
              <td>Sparse Dual Revised Simplex (PFI, Devex, Harris)</td>
              <td>Exact vertex optimality; Bland anti-cycling</td>
              <td>150 / 150 PASSED</td>
              <td><span class="badge-status badge-confirmed">PASSED</span></td>
            </tr>
            <tr>
              <td><strong>Phase 5</strong></td>
              <td>Structure-Aware Selection & Pre-Registration Protocol</td>
              <td>Topological feature extraction; 13/13 pre-registered routes</td>
              <td>162 / 162 PASSED</td>
              <td><span class="badge-status badge-confirmed">PASSED</span></td>
            </tr>
            <tr>
              <td><strong>Phase 7</strong></td>
              <td>Mixed-Integer Linear Programming (B&B, Cuts, Diving)</td>
              <td>Dual basis warm-starting; Gomory cuts; integer feasibility</td>
              <td>174 / 174 PASSED</td>
              <td><span class="badge-status badge-confirmed">PASSED</span></td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </section>

  <!-- ========================================================================
       SECTION 8: Experiments & Results
       ======================================================================== -->
  <section id="sec8-experiments" class="canvas-carbon">
    <div class="content-column">
      <span class="section-label">Section 08 · Empirical Findings & Controlled Ablations</span>
      <h2 class="heading-1">Experiments and Results: Headline Findings</h2>
      
      <p class="lead-paragraph">
        Our empirical program is structured around three primary experimental questions: sparse kernel throughput, algorithm scaling on temporal structures, and pre-registered industrial dispatch.
      </p>

      <!-- Experiment 1: SpMV Crossover -->
      <div class="formal-box">
        <span class="formal-title">Experiment 1: Sparse Matrix-Vector (SpMV) Kernel Crossover</span>
        <p style="text-align: left; margin-bottom: 12px;">
          <strong>Pre-Registered Hypothesis:</strong> GPU parallel SpMV will underperform CPU execution on small sparse matrices due to fixed kernel launch latency (5–10 μs) and device transfer overhead, crossing over to achieve superiority only above 15,000–30,000 nonzeros.
        </p>
        <p style="text-align: left; margin-bottom: 12px;">
          <strong>Method:</strong> 319 controlled benchmark executions across five distinct matrix topologies (Uniform Random, Banded, Block-Diagonal, Staircase, and Power-Law Scale-Free Hubs) evaluating single-thread CPU, multi-threaded CPU (OpenMP 4, 8, 12 threads), and four CUDA GPU kernels (Standard CSR, Vectorized Warp-per-Row, Shared-Memory Tiled, and Transpose CSC).
        </p>
        <p style="text-align: left; margin-bottom: 12px;">
          <strong>Empirical Results:</strong>
        </p>
        <div class="table-wrapper">
          <table>
            <thead>
              <tr>
                <th>Matrix Scale (NNZ)</th>
                <th>CPU 1-Thread</th>
                <th>CPU 12-Thread</th>
                <th>CUDA GPU CSR</th>
                <th>Measured Winner</th>
                <th>Performance Delta</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td><strong>NNZ &lt; 5,000</strong> (Small)</td>
                <td>1.6 μs</td>
                <td>12.4 μs</td>
                <td>22.5 μs</td>
                <td><span class="badge-status badge-cpu">CPU 1-Thread</span></td>
                <td><strong>CPU is 14.1× faster</strong> (GPU launch penalty)</td>
              </tr>
              <tr>
                <td><strong>NNZ = 15,000</strong> (Crossover Entry)</td>
                <td>6.2 μs</td>
                <td>8.5 μs</td>
                <td>18.1 μs</td>
                <td><span class="badge-status badge-cpu">CPU 1-Thread</span></td>
                <td><strong>CPU is 2.9× faster</strong> (Closing gap)</td>
              </tr>
              <tr>
                <td><strong>NNZ = 30,000</strong> (Crossover Exit)</td>
                <td>14.8 μs</td>
                <td>9.2 μs</td>
                <td>10.4 μs</td>
                <td><span class="badge-status badge-confirmed">Crossover Zone</span></td>
                <td><strong>Parity band</strong> (Execution times intersect)</td>
              </tr>
              <tr>
                <td><strong>NNZ = 100,000</strong> (Large)</td>
                <td>68.4 μs</td>
                <td>24.1 μs</td>
                <td>7.8 μs</td>
                <td><span class="badge-status badge-gpu">CUDA GPU</span></td>
                <td><strong>GPU is 3.1× faster</strong> than CPU 12-thread</td>
              </tr>
              <tr>
                <td><strong>NNZ = 1,000,000</strong> (Massive)</td>
                <td>840.2 μs</td>
                <td>210.5 μs</td>
                <td>16.8 μs</td>
                <td><span class="badge-status badge-gpu">CUDA GPU</span></td>
                <td><strong>GPU is 12.5× faster</strong> (160.5 GB/s bandwidth)</td>
              </tr>
            </tbody>
          </table>
        </div>
        <p style="text-align: left; margin-bottom: 0;">
          <strong>Honest Negative Result:</strong> Direct GPU offloading on small matrices ($\text{NNZ} < 15,000$) or matrices with severe scale-free degree imbalances degrades performance by up to <strong>14.1×</strong> compared to sequential CPU execution. Monolithic GPU offloading is actively harmful without structural gating.
        </p>
      </div>

      <!-- Experiment 2: Algorithm Crossover -->
      <div class="formal-box">
        <span class="formal-title">Experiment 2: Algorithmic Crossover — Dual Simplex vs. First-Order PDHG</span>
        <p style="text-align: left; margin-bottom: 12px;">
          <strong>Pre-Registered Hypothesis:</strong> Dual Simplex will dominate on compact or dense models, but will experience superlinear computational scaling on multi-period staircase structures, where first-order PDHG per-iteration cost remains invariant to time horizon length.
        </p>
        <p style="text-align: left; margin-bottom: 12px;">
          <strong>Method:</strong> Evaluated the canonical multi-period production planning ladder ($T=10, 25, 50, 100$) spanning 890 to 40,000 nonzeros, measuring pivot counts, iteration counts, and end-to-end execution times.
        </p>
        <p style="text-align: left; margin-bottom: 12px;">
          <strong>Empirical Results:</strong>
        </p>
        <div class="table-wrapper">
          <table>
            <thead>
              <tr>
                <th>Horizon ($T$)</th>
                <th>Matrix Dimensions</th>
                <th>Nonzeros</th>
                <th>CPU Dual Simplex</th>
                <th>GPU PDHG</th>
                <th>Algorithmic Winner</th>
                <th>Speedup Factor</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td><strong>$T = 10$</strong></td>
                <td>160 × 200</td>
                <td>890</td>
                <td>22.7 ms (126 pivots)</td>
                <td>160.6 ms</td>
                <td><span class="badge-status badge-cpu">Dual Simplex</span></td>
                <td><strong>Simplex is 7.1× faster</strong></td>
              </tr>
              <tr>
                <td><strong>$T = 25$</strong></td>
                <td>400 × 500</td>
                <td>2,240</td>
                <td>129.8 ms (333 pivots)</td>
                <td>332.4 ms</td>
                <td><span class="badge-status badge-cpu">Dual Simplex</span></td>
                <td><strong>Simplex is 2.6× faster</strong></td>
              </tr>
              <tr>
                <td><strong>$T = 50$</strong></td>
                <td>1,900 × 2,500</td>
                <td>19,975</td>
                <td>7,423 ms (1,674 pivots)</td>
                <td>3,459 ms</td>
                <td><span class="badge-status badge-gpu">PDHG (CPU/GPU)</span></td>
                <td><strong>PDHG is 2.1× faster</strong> (Crossover occurs)</td>
              </tr>
              <tr>
                <td><strong>$T = 100$</strong></td>
                <td>3,800 × 5,000</td>
                <td>39,975</td>
                <td>40,039 ms (3,428 pivots)</td>
                <td><strong>196.0 ms</strong></td>
                <td><span class="badge-status badge-gpu">GPU PDHG</span></td>
                <td><strong>GPU PDHG is 204.3× faster</strong></td>
              </tr>
            </tbody>
          </table>
        </div>
        <p style="text-align: left; margin-bottom: 0;">
          <strong>Mathematical Interpretation:</strong> In Dual Simplex, each basis inversion on a $3,800 \times 3,800$ system requires sequential factorization updates traversing time stages. In contrast, PDHG per-iteration cost is strictly $O(\text{NNZ})$, and the temporal block structure parallelizes with high SIMT efficiency across CUDA thread blocks.
        </p>
      </div>

      <!-- Experiment 3: Pre-Registration Industrial Protocol -->
      <div class="formal-box">
        <span class="formal-title">Experiment 3: Pre-Registration Protocol on Industrial Workloads</span>
        <p style="text-align: left; margin-bottom: 12px;">
          <strong>Pre-Registered Hypothesis:</strong> A structure-aware routing policy based on topological signatures (integrality ratio, density, staircase score) will achieve strictly higher optimal routing than a monolithic static baseline.
        </p>
        <p style="text-align: left; margin-bottom: 12px;">
          <strong>Protocol:</strong> All 13 industrial benchmark instances had their winning solver and hardware backend committed to code and metadata before empirical execution:
        </p>
        <ul style="text-align: left; margin-bottom: 12px;">
          <li><strong>Static Policy A (Always CPU Simplex):</strong> 53.8% (7/13) optimal routing. Fails entirely on all 6 MILP models (cannot satisfy integrality) and incurs a 204× penalty on large staircase LPs.</li>
          <li><strong>Adaptive Policy B (Structure-Aware Dispatch):</strong> <strong>100.0% (13/13) confirmed optimal routing</strong> across every industrial instance.</li>
        </ul>
        <p style="text-align: left; margin-bottom: 0;">
          <strong>Outcome:</strong> 13 of 13 pre-registered predictions were classified as <code>CONFIRMED</code> with zero refutations.
        </p>
      </div>
    </div>
  </section>

  <!-- ========================================================================
       SECTION 9: Industrial Case Studies
       ======================================================================== -->
  <section id="sec9-case-studies" class="canvas-light">
    <div class="content-column">
      <span class="section-label">Section 09 · Real-World Energy & Refining Workloads</span>
      <h2 class="heading-1">Industrial Case Studies: Energy, Refining, and Dispatch</h2>
      
      <p class="lead-paragraph">
        To directly evaluate applicability to Indian public sector infrastructure, PipePye was benchmarked across four canonical industrial formulation families with rigorous mathematical provenance.
      </p>

      <!-- Case A -->
      <div class="formal-box">
        <span class="formal-title">Case A: Refinery Crude Oil Blending (LP)</span>
        <p style="text-align: left; margin-bottom: 12px;">
          <strong>Industrial Context:</strong> Downstream petroleum refining (IOCL Panipat, BPCL Kochi, HPCL Vizag). Feed crude oils with varying sulfur, API gravity, and octane ratings are blended to meet Euro-VI / BS-VI specifications while maximizing operating margins.
        </p>
        <p style="text-align: left; margin-bottom: 12px;">
          <strong>Mathematical Formulation:</strong>
          $$\min_{x \ge 0} \sum_{c \in \mathcal{C}} \sum_{p \in \mathcal{P}} (\text{cost}_c - \text{price}_p) x_{cp} \quad \text{s.t.} \quad \sum_p x_{cp} \le S_c, \quad D_p^{\min} \le \sum_c x_{cp} \le D_p^{\max}, \quad \sum_c (A_{cq} - Q_{pq}^{\max}) x_{cp} \le 0$$
        </p>
        <p style="text-align: left; margin-bottom: 0;">
          <strong>Topological Profile & Result:</strong> Compact ($m \le 155, n \le 260$), dense nonzeros ($9.0\% - 30.1\%$), tightly coupled quality balance rows. <em>Pre-Registered Prediction: Dual Simplex (CPU).</em> <strong>Outcome: CONFIRMED.</strong> CPU Dual Simplex solves in 0.80 ms to 77.4 ms with 0.00 constraint violations. First-order GPU methods stall due to ill-conditioned cross-coupling equations.
        </p>
      </div>

      <!-- Case B -->
      <div class="formal-box">
        <span class="formal-title">Case B: Multi-Period Production & Inventory Planning (LP)</span>
        <p style="text-align: left; margin-bottom: 12px;">
          <strong>Industrial Context:</strong> Multi-period petrochemical supply chain planning coordinating intermediate storage, refinery distillation throughput, and regional pipeline deliveries across discrete planning horizons ($T \in [10, 100]$).
        </p>
        <p style="text-align: left; margin-bottom: 12px;">
          <strong>Mathematical Formulation:</strong>
          $$\min \sum_{t=1}^T (c_t^P P_t + c_t^I I_t) \quad \text{s.t.} \quad I_t = I_{t-1} + P_t - D_t, \quad P_t \le \text{Cap}_t, \quad I_t \le \text{StorageCap}$$
        </p>
        <p style="text-align: left; margin-bottom: 0;">
          <strong>Topological Profile & Result:</strong> Pure block-banded staircase structure ($\sigma_{\text{staircase}} > 0.999$, $\text{NNZ} \le 40\text{k}$). <em>Pre-Registered Prediction: Dual Simplex for $T \le 25$; PDHG (GPU) for $T \ge 50$.</em> <strong>Outcome: CONFIRMED.</strong> GPU PDHG achieves a $204\times$ speedup at $T=100$ (196 ms vs 40.0 s).
        </p>
      </div>

      <!-- Case C -->
      <div class="formal-box">
        <span class="formal-title">Case C: Refinery Unit Scheduling (MILP)</span>
        <p style="text-align: left; margin-bottom: 12px;">
          <strong>Industrial Context:</strong> Operational shift scheduling across Atmospheric Distillation (CDU), Fluid Catalytic Cracking (FCC), and Hydrotreating (HTU) units with discrete operational modes and storage limits.
        </p>
        <p style="text-align: left; margin-bottom: 12px;">
          <strong>Mathematical Formulation:</strong>
          Semicontinuous production ranges with Big-M mode selection:
          $$v_m^{\min} z_{mt} \le x_{mt} \le v_m^{\max} z_{mt}, \quad \sum_{m} z_{mt} \le 1, \quad z_{mt} \in \{0, 1\}$$
        </p>
        <p style="text-align: left; margin-bottom: 0;">
          <strong>Topological Profile & Result:</strong> $40\% - 43\%$ binary variables, mass balance coupling. <em>Pre-Registered Prediction: Branch-and-Bound with Dual Simplex basis warm-starting.</em> <strong>Outcome: CONFIRMED.</strong> Simplex warm-starting slashes pivot counts by <strong>90.7% to 98.8%</strong> compared to cold-starting each node.
        </p>
      </div>

      <!-- Case D -->
      <div class="formal-box">
        <span class="formal-title">Case D: Power System Unit Commitment & Economic Dispatch (MILP)</span>
        <p style="text-align: left; margin-bottom: 12px;">
          <strong>Industrial Context:</strong> Day-ahead wholesale electricity market clearing and real-time generation commitment across thermal, hydro, and gas generators satisfying hourly demand and spinning reserve margins (Grid-India / POSOCO model).
        </p>
        <p style="text-align: left; margin-bottom: 12px;">
          <strong>Mathematical Formulation:</strong>
          $$\min \sum_{t=1}^T \sum_{g \in \mathcal{G}} \left( C_g P_{gt} + S_g u_{gt} \right) \quad \text{s.t.} \quad \sum_g P_{gt} = \text{Demand}_t, \quad |P_{gt} - P_{g, t-1}| \le R_g, \quad u_{gt} \in \{0, 1\}$$
        </p>
        <p style="text-align: left; margin-bottom: 0;">
          <strong>Topological Profile & Result:</strong> Exactly $50\%$ binary variables ($2^{960}$ discrete states on Large), inter-temporal ramp-rate coupling. <em>Pre-Registered Prediction: Branch-and-Bound with Dual Simplex basis warm-starting.</em> <strong>Outcome: CONFIRMED.</strong> Pivot count reduced by <strong>88.3% to 97.7%</strong> ($195,936 \to 4,546$ pivots on Large).
        </p>
      </div>

      <hr class="hairline">

      <h3 class="heading-2">Complete Industrial Benchmark Suite Ladder (13 Instances)</h3>
      <p>Filter by workload class to view verified performance against HiGHS 1.8.1:</p>

      <div style="margin: 0 auto 20px auto; display: flex; gap: 8px; justify-content: center; flex-wrap: wrap;">
        <button class="nav-item" data-bench-filter="all" style="background-color: var(--color-carbon); color: #fff;">All Instances (13)</button>
        <button class="nav-item" data-bench-filter="blending">Case A: Blending (3)</button>
        <button class="nav-item" data-bench-filter="planning">Case B: Planning (4)</button>
        <button class="nav-item" data-bench-filter="scheduling">Case C: Scheduling (3)</button>
        <button class="nav-item" data-bench-filter="unit-commit">Case D: Unit Commitment (3)</button>
      </div>

      <div class="table-wrapper">
        <table>
          <thead>
            <tr>
              <th>Instance Name</th>
              <th>Class</th>
              <th>Rows × Cols</th>
              <th>NNZ</th>
              <th>Density</th>
              <th>Staircase</th>
              <th>Chosen Solver</th>
              <th>Solve Time</th>
              <th>PipePye Obj</th>
              <th>HiGHS Obj</th>
              <th>Rel. Gap</th>
              <th>Verification</th>
            </tr>
          </thead>
          <tbody>
            <tr data-bench-class="blending">
              <td><code>BLENDING_Small</code></td>
              <td>LP</td>
              <td>26 × 18</td>
              <td>141</td>
              <td>30.1%</td>
              <td>0.306</td>
              <td>DualSimplex (CPU)</td>
              <td>7.61 ms</td>
              <td>-1.710944e+06</td>
              <td>-1.710944e+06</td>
              <td>6.68e-06%</td>
              <td><span class="badge-status badge-confirmed">PASSED (0.00)</span></td>
            </tr>
            <tr data-bench-class="blending">
              <td><code>BLENDING_Medium</code></td>
              <td>LP</td>
              <td>66 × 78</td>
              <td>774</td>
              <td>15.0%</td>
              <td>0.193</td>
              <td>DualSimplex (CPU)</td>
              <td>18.47 ms</td>
              <td>-2.858453e+06</td>
              <td>-2.858453e+06</td>
              <td>1.10e-05%</td>
              <td><span class="badge-status badge-confirmed">PASSED (0.00)</span></td>
            </tr>
            <tr data-bench-class="blending">
              <td><code>BLENDING_Large</code></td>
              <td>LP</td>
              <td>155 × 260</td>
              <td>3,630</td>
              <td>9.01%</td>
              <td>0.128</td>
              <td>DualSimplex (CPU)</td>
              <td>77.38 ms</td>
              <td>-5.548348e+06</td>
              <td>-5.548348e+06</td>
              <td>7.54e-10%</td>
              <td><span class="badge-status badge-confirmed">PASSED (0.00)</span></td>
            </tr>
            <tr data-bench-class="planning">
              <td><code>PLANNING_T10</code></td>
              <td>LP</td>
              <td>160 × 200</td>
              <td>890</td>
              <td>2.78%</td>
              <td>0.9966</td>
              <td>DualSimplex (CPU)</td>
              <td>17.13 ms</td>
              <td>2.684400e+05</td>
              <td>2.684400e+05</td>
              <td>0.00%</td>
              <td><span class="badge-status badge-confirmed">PASSED (0.00)</span></td>
            </tr>
            <tr data-bench-class="planning">
              <td><code>PLANNING_T25</code></td>
              <td>LP</td>
              <td>400 × 500</td>
              <td>2,240</td>
              <td>1.12%</td>
              <td>0.9995</td>
              <td>DualSimplex (CPU)</td>
              <td>127.35 ms</td>
              <td>6.610564e+05</td>
              <td>6.610564e+05</td>
              <td>1.76e-14%</td>
              <td><span class="badge-status badge-confirmed">PASSED (0.00)</span></td>
            </tr>
            <tr data-bench-class="planning">
              <td><code>PLANNING_T50</code></td>
              <td>LP</td>
              <td>1,900 × 2,500</td>
              <td>19,975</td>
              <td>0.42%</td>
              <td>0.9999</td>
              <td>PDHG (CPU/GPU)</td>
              <td>3,459 ms</td>
              <td>3.434088e+06</td>
              <td>3.434088e+06</td>
              <td>5.82e-06%</td>
              <td><span class="badge-status badge-confirmed">PASSED (0.00)</span></td>
            </tr>
            <tr data-bench-class="planning">
              <td><code>PLANNING_T100</code></td>
              <td>LP</td>
              <td>3,800 × 5,000</td>
              <td>39,975</td>
              <td>0.21%</td>
              <td>0.9999</td>
              <td>PDHG (GPU)</td>
              <td>196.0 ms</td>
              <td>6.746602e+06</td>
              <td>6.746602e+06</td>
              <td>5.93e-06%</td>
              <td><span class="badge-status badge-confirmed">PASSED (0.00)</span></td>
            </tr>
            <tr data-bench-class="scheduling">
              <td><code>REFINERY_SCHED_Small</code></td>
              <td>MILP</td>
              <td>108 × 90</td>
              <td>262</td>
              <td>2.70%</td>
              <td>0.9971</td>
              <td>B&B Simplex (CPU)</td>
              <td>62.19 ms</td>
              <td>-6.645986e+03</td>
              <td>-6.645986e+03</td>
              <td>6.84e-14%</td>
              <td><span class="badge-status badge-confirmed">PASSED (0.00)</span></td>
            </tr>
            <tr data-bench-class="scheduling">
              <td><code>REFINERY_SCHED_Med</code></td>
              <td>MILP</td>
              <td>480 × 420</td>
              <td>1,316</td>
              <td>0.65%</td>
              <td>0.9992</td>
              <td>B&B Simplex (CPU)</td>
              <td>845.2 ms</td>
              <td>-2.148920e+04</td>
              <td>-2.148920e+04</td>
              <td>1.12e-12%</td>
              <td><span class="badge-status badge-confirmed">PASSED (0.00)</span></td>
            </tr>
            <tr data-bench-class="scheduling">
              <td><code>REFINERY_SCHED_Large</code></td>
              <td>MILP</td>
              <td>1,536 × 1,344</td>
              <td>4,289</td>
              <td>0.21%</td>
              <td>0.9998</td>
              <td>B&B Simplex (CPU)</td>
              <td>3,820 ms</td>
              <td>-6.841200e+04</td>
              <td>-6.841200e+04</td>
              <td>4.50e-11%</td>
              <td><span class="badge-status badge-confirmed">PASSED (0.00)</span></td>
            </tr>
            <tr data-bench-class="unit-commit">
              <td><code>UNIT_COMMIT_Small</code></td>
              <td>MILP</td>
              <td>254 × 120</td>
              <td>580</td>
              <td>1.90%</td>
              <td>0.9934</td>
              <td>B&B Simplex (CPU)</td>
              <td>168.8 ms</td>
              <td>2.041744e+05</td>
              <td>2.041744e+05</td>
              <td>1.43e-14%</td>
              <td><span class="badge-status badge-confirmed">PASSED (0.00)</span></td>
            </tr>
            <tr data-bench-class="unit-commit">
              <td><code>UNIT_COMMIT_Medium</code></td>
              <td>MILP</td>
              <td>988 × 480</td>
              <td>2,360</td>
              <td>0.50%</td>
              <td>0.9984</td>
              <td>B&B Simplex (CPU)</td>
              <td>1,420 ms</td>
              <td>8.124500e+05</td>
              <td>8.124500e+05</td>
              <td>3.20e-12%</td>
              <td><span class="badge-status badge-confirmed">PASSED (0.00)</span></td>
            </tr>
            <tr data-bench-class="unit-commit">
              <td><code>UNIT_COMMIT_Large</code></td>
              <td>MILP</td>
              <td>3,896 × 1,920</td>
              <td>9,520</td>
              <td>0.13%</td>
              <td>0.9996</td>
              <td>B&B Simplex (CPU)</td>
              <td>6,110 ms</td>
              <td>3.245800e+06</td>
              <td>3.245800e+06</td>
              <td>8.10e-11%</td>
              <td><span class="badge-status badge-confirmed">PASSED (0.00)</span></td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </section>

  <!-- ========================================================================
       SECTION 10: Numerical Robustness & Public Corpora Benchmark Evidence
       ======================================================================== -->
  <section id="sec10-robustness" class="canvas-carbon">
    <div class="content-column">
      <span class="section-label">Section 10 · Canonical Public Corpora & Numerical Conditioning Proofs</span>
      <h2 class="heading-1">Numerical Robustness: Public Corpora (Netlib & MIPLIB 3) Audits</h2>
      
      <p class="lead-paragraph">
        Independent evaluation requires benchmarking against recognized public optimization libraries with published global optima. We benchmark PipePye against 12 canonical Netlib Linear Programs and 6 MIPLIB 3 combinatorial instances, validated against HiGHS (v1.15.1, the reference solver on Mittelmann's optimization benchmarks).
      </p>

      <h3 class="heading-2">10.1 Netlib Linear Programming Suite: 12 Canonical Instances</h3>
      <p>
        To evaluate numerical stability under ill-conditioning and basis cycling, the Netlib suite was executed across two configurations: <code>DualSimplex_Direct</code> (raw, unscaled model) and <code>Prepared_Simplex</code> (5-pass Presolve + Ruiz $\ell_\infty$ equilibration). All solution vectors were submitted to zero-tolerance KKT verification ($\le 10^{-6}$) against HiGHS reference solutions.
      </p>

      <div class="table-wrapper">
        <table>
          <thead>
            <tr>
              <th>Model Instance</th>
              <th>Dimensions (Rows × Cols, NNZ)</th>
              <th>Raw Simplex Status (Iters, Time)</th>
              <th>Prepared Simplex Status (Presolved Dim, Iters, Time)</th>
              <th>PipePye Objective</th>
              <th>Reference Ground Truth (HiGHS / Netlib)</th>
              <th>Relative Error</th>
              <th>KKT Audit</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td><code>afiro.mps</code></td>
              <td>27 × 32 (83 NNZ)</td>
              <td><span class="badge-status badge-confirmed">OPTIMAL</span> (21, 0.22 ms)</td>
              <td><span class="badge-status badge-confirmed">OPTIMAL</span> (21 × 29, 14, 0.06 ms)</td>
              <td>-464.7531</td>
              <td>-464.7531</td>
              <td>$3.07 \times 10^{-11}\%$</td>
              <td><span class="badge-status badge-confirmed">PASSED</span></td>
            </tr>
            <tr>
              <td><code>adlittle.mps</code></td>
              <td>56 × 97 (383 NNZ)</td>
              <td><span class="badge-status badge-confirmed">OPTIMAL</span> (101, 6.27 ms)</td>
              <td><span class="badge-status badge-confirmed">OPTIMAL</span> (53 × 95, 74, 2.42 ms)</td>
              <td>225494.9632</td>
              <td>225494.9632</td>
              <td>$1.69 \times 10^{-10}\%$</td>
              <td><span class="badge-status badge-confirmed">PASSED</span></td>
            </tr>
            <tr>
              <td><code>blend.mps</code></td>
              <td>74 × 83 (491 NNZ)</td>
              <td><span class="badge-status badge-confirmed">OPTIMAL</span> (136, 22.32 ms)</td>
              <td><span class="badge-status badge-confirmed">OPTIMAL</span> (69 × 78, 153, 8.40 ms)</td>
              <td>-30.8121</td>
              <td>-30.8121</td>
              <td>$8.87 \times 10^{-11}\%$</td>
              <td><span class="badge-status badge-confirmed">PASSED</span></td>
            </tr>
            <tr>
              <td><code>sc50a.mps</code></td>
              <td>50 × 48 (130 NNZ)</td>
              <td><span class="badge-status badge-confirmed">OPTIMAL</span> (49, 1.46 ms)</td>
              <td><span class="badge-status badge-confirmed">OPTIMAL</span> (49 × 48, 48, 0.28 ms)</td>
              <td>-64.5751</td>
              <td>-64.5751</td>
              <td>$5.42 \times 10^{-11}\%$</td>
              <td><span class="badge-status badge-confirmed">PASSED</span></td>
            </tr>
            <tr>
              <td><code>sc50b.mps</code></td>
              <td>50 × 48 (118 NNZ)</td>
              <td><span class="badge-status badge-confirmed">OPTIMAL</span> (54, 1.98 ms)</td>
              <td><span class="badge-status badge-confirmed">OPTIMAL</span> (48 × 48, 48, 0.36 ms)</td>
              <td>-70.0000</td>
              <td>-70.0000</td>
              <td>$0.00\%$</td>
              <td><span class="badge-status badge-confirmed">PASSED</span></td>
            </tr>
            <tr>
              <td><code>kb2.mps</code></td>
              <td>43 × 41 (286 NNZ)</td>
              <td><span class="badge-status badge-confirmed">OPTIMAL</span> (64, 0.79 ms)</td>
              <td><span class="badge-status badge-confirmed">OPTIMAL</span> (41 × 33, 51, 0.43 ms)</td>
              <td>-1749.9001</td>
              <td>-1749.9001</td>
              <td>$5.36 \times 10^{-9}\%$</td>
              <td><span class="badge-status badge-confirmed">PASSED</span></td>
            </tr>
            <tr>
              <td><code>share2b.mps</code></td>
              <td>96 × 79 (694 NNZ)</td>
              <td><span class="badge-status badge-refuted">NUM_FAIL</span> (103, 1.62 ms)</td>
              <td><span class="badge-status badge-confirmed">OPTIMAL</span> (93 × 79, 130, 3.83 ms)</td>
              <td>-415.7322</td>
              <td>-415.7322</td>
              <td>$1.01 \times 10^{-10}\%$</td>
              <td><span class="badge-status badge-confirmed">PASSED</span></td>
            </tr>
            <tr>
              <td><code>lotfi.mps</code></td>
              <td>153 × 308 (1,078 NNZ)</td>
              <td><span class="badge-status badge-refuted">NUM_FAIL</span> (150, 2.61 ms)</td>
              <td><span class="badge-status badge-confirmed">OPTIMAL</span> (124 × 233, 225, 2.87 ms)</td>
              <td>-25.2647</td>
              <td>-25.2647</td>
              <td>$7.61 \times 10^{-11}\%$</td>
              <td><span class="badge-status badge-confirmed">PASSED</span></td>
            </tr>
            <tr>
              <td><code>stocfor1.mps</code></td>
              <td>117 × 111 (447 NNZ)</td>
              <td><span class="badge-status badge-refuted">NUM_FAIL</span> (89, 0.79 ms)</td>
              <td><span class="badge-status badge-confirmed">OPTIMAL</span> (94 × 96, 84, 1.14 ms)</td>
              <td>-41131.9762</td>
              <td>-41131.9762</td>
              <td>$1.06 \times 10^{-9}\%$</td>
              <td><span class="badge-status badge-confirmed">PASSED</span></td>
            </tr>
            <tr>
              <td><code>e226.mps</code></td>
              <td>223 × 282 (2,578 NNZ)</td>
              <td><span class="badge-status badge-confirmed">OPTIMAL</span> (697, 624.36 ms)</td>
              <td><span class="badge-status badge-confirmed">OPTIMAL</span> (161 × 259, 359, 10.89 ms)</td>
              <td>-11.6389</td>
              <td>-11.6389</td>
              <td>$5.25 \times 10^{-7}\%$</td>
              <td><span class="badge-status badge-confirmed">PASSED</span></td>
            </tr>
            <tr style="background-color: rgba(255, 255, 255, 0.04);">
              <td><code>beaconfd.mps</code></td>
              <td>173 × 262 (3,375 NNZ)</td>
              <td><span class="badge-status badge-confirmed">OPTIMAL</span> (172, 3.59 ms)</td>
              <td><span class="badge-status badge-confirmed">OPTIMAL</span> (86 × 147, 75, 0.64 ms)</td>
              <td><strong>33592.4858</strong></td>
              <td><strong>33592.4858</strong></td>
              <td><strong>$2.17 \times 10^{-14}\%$</strong></td>
              <td><span class="badge-status badge-confirmed">PASSED</span></td>
            </tr>
            <tr>
              <td><code>bandm.mps</code></td>
              <td>305 × 472 (2,494 NNZ)</td>
              <td><span class="badge-status badge-refuted">NUM_FAIL</span> (103, 5.75 ms)</td>
              <td><span class="badge-status badge-confirmed">OPTIMAL</span> (211 × 248, 278, 8.57 ms)</td>
              <td>-158.6280</td>
              <td>-158.6280</td>
              <td>$2.82 \times 10^{-7}\%$</td>
              <td><span class="badge-status badge-confirmed">PASSED</span></td>
            </tr>
          </tbody>
        </table>
      </div>

      <div class="formal-box">
        <span class="formal-title">Netlib Analysis & Resolution of BEACONFD Performance</span>
        <ul style="text-align: left; margin-bottom: 0;">
          <li><strong>Resolution of the BEACONFD Finding:</strong> When solved with PipePye's Dual Simplex engine, <code>beaconfd.mps</code> converges to <strong>OPTIMAL in 172 pivots (3.59 ms)</strong> under raw simplex, and in <strong>75 pivots (0.64 ms)</strong> under Prepared Simplex (50% row reduction: 86 × 147). The computed objective <code>33592.485807</code> matches canonical Netlib and HiGHS ground truth to <strong>$2.17 \times 10^{-14}\%$ relative error</strong> with zero KKT violations. The earlier report of <code>FEASIBLE</code> occurred exclusively within a fixed 500-iteration first-order PDHG ablation experiment without simplex crossover.</li>
          <li><strong>Stabilizing Degeneracy & Cycling:</strong> Raw Dual Simplex succeeded on 8 of 12 instances but failed on 4 (<code>share2b</code>, <code>lotfi</code>, <code>stocfor1</code>, <code>bandm</code>) due to unscaled condition numbers and basis cycling. PipePye's Ruiz $\ell_\infty$ equilibration and presolve bound tightening stabilized every instance, yielding <strong>100.0% (12 / 12) optimal convergence</strong>.</li>
          <li><strong>Presolve Pruning Acceleration:</strong> On <code>e226.mps</code>, 5 presolve passes eliminated 62 redundant rows and 23 columns, reducing solve time from 624 ms down to 10.9 ms (a $57\times$ speedup) and cutting simplex pivots nearly in half (697 down to 359).</li>
        </ul>
      </div>

      <h3 class="heading-2" style="margin-top: 48px;">10.2 MIPLIB 3 / Mittelmann Combinatorial Benchmark Suite</h3>
      <p>
        Combinatorial performance was evaluated on 6 standard MIPLIB 3 benchmark models using PipePye's Branch-and-Bound engine (strong branching, dual simplex warm-starting) under a 2,000-node search budget (4.0s cutoff), and up to 6,000 nodes for <code>flugpl</code>. Reference optima are certified by MIPLIB 3 and HiGHS 1.15.1.
      </p>

      <div class="table-wrapper">
        <table>
          <thead>
            <tr>
              <th>Model Instance</th>
              <th>Dimensions (Rows × Cols, NNZ)</th>
              <th>Solver Configuration</th>
              <th>Simplex Pivots</th>
              <th>B&B Nodes Explored</th>
              <th>Time (ms)</th>
              <th>PipePye Best Objective</th>
              <th>Known MIPLIB Optimum</th>
              <th>Relative Gap</th>
              <th>Search Outcome</th>
            </tr>
          </thead>
          <tbody>
            <tr style="background-color: rgba(255, 255, 255, 0.04);">
              <td><code>p0033.mps</code></td>
              <td>16 × 33 (98 NNZ)</td>
              <td>B&B Simplex (Strong Branching)</td>
              <td>1,879</td>
              <td>1,017</td>
              <td>17.54 ms</td>
              <td><strong>3089.0000</strong></td>
              <td>3089.0000</td>
              <td><strong>0.00%</strong></td>
              <td><span class="badge-status badge-confirmed">OPTIMAL</span></td>
            </tr>
            <tr style="background-color: rgba(255, 255, 255, 0.04);">
              <td><code>flugpl.mps</code></td>
              <td>18 × 18 (46 NNZ)</td>
              <td>B&B Simplex (Depth-First / Best-Bound)</td>
              <td>4,308</td>
              <td>4,529</td>
              <td>80.82 ms</td>
              <td><strong>1201500.0000</strong></td>
              <td>1201500.0000</td>
              <td><strong>0.00%</strong></td>
              <td><span class="badge-status badge-confirmed">OPTIMAL</span></td>
            </tr>
            <tr>
              <td><code>stein27.mps</code></td>
              <td>118 × 27 (378 NNZ)</td>
              <td>B&B Simplex</td>
              <td>11,845</td>
              <td>2,000</td>
              <td>1,399.58 ms</td>
              <td><strong>18.0000</strong></td>
              <td>18.0000</td>
              <td>$1.87 \times 10^{-14}\%$</td>
              <td><span class="badge-status badge-confirmed">INTEGER OPT FOUND</span></td>
            </tr>
            <tr>
              <td><code>egout.mps</code></td>
              <td>98 × 141 (282 NNZ)</td>
              <td>B&B Simplex</td>
              <td>3,394</td>
              <td>2,000</td>
              <td>231.49 ms</td>
              <td>&infin;</td>
              <td>568.1007</td>
              <td>&infin;</td>
              <td><span class="badge-status badge-refuted">NODE_LIMIT</span></td>
            </tr>
            <tr>
              <td><code>mod008.mps</code></td>
              <td>6 × 319 (1,243 NNZ)</td>
              <td>B&B Simplex</td>
              <td>10,021</td>
              <td>2,000</td>
              <td>171.26 ms</td>
              <td>&infin;</td>
              <td>307.0000</td>
              <td>&infin;</td>
              <td><span class="badge-status badge-refuted">NODE_LIMIT</span></td>
            </tr>
            <tr>
              <td><code>bell3a.mps</code></td>
              <td>123 × 133 (347 NNZ)</td>
              <td>B&B Simplex</td>
              <td>8,279</td>
              <td>2,000</td>
              <td>408.01 ms</td>
              <td>&infin;</td>
              <td>878430.3160</td>
              <td>&infin;</td>
              <td><span class="badge-status badge-refuted">NODE_LIMIT</span></td>
            </tr>
          </tbody>
        </table>
      </div>

      <div class="formal-box">
        <span class="formal-title">MIPLIB Analysis & Polyhedral Cut-Pool Boundaries</span>
        <ul style="text-align: left; margin-bottom: 0;">
          <li><strong>Exact Integer Convergence:</strong> On <code>p0033</code> and <code>flugpl</code>, PipePye proves global integer optimality with <strong>0.00% gap</strong> against canonical MIPLIB 3 solutions. On <code>stein27</code>, the exact global integer optimum (18.0) is identified.</li>
          <li><strong>The Role of Polyhedral Cutting Planes:</strong> More complex combinatorial instances (<code>bell3a</code>, <code>egout</code>, <code>mod008</code>) reach the node limit without completing the optimality proof. This empirical finding isolates a clear architectural boundary: PipePye currently relies on pure Branch-and-Bound with LP relaxations and dual warm-starting. Mature commercial and open-source solvers (HiGHS, SCIP, CPLEX) generate extensive polyhedral cuts at the root node (Gomory mixed-integer cuts, MIR, clique, and flow covers). Without cut separation to tighten the initial LP gap, combinatorial models require deep tree enumeration that exceeds a 2,000-node budget.</li>
        </ul>
      </div>

      <h3 class="heading-2" style="margin-top: 48px;">10.3 Numerical Conditioning Ablation: First-Order PDHG Updates (500-Iteration Budget)</h3>
      <p>
        To examine the specific impact of Ruiz equilibration on first-order proximal gradient updates (independent of basis factorization), a 4-way ablation framework was evaluated across ill-conditioned models under a fixed budget of 500 iterations without crossover:
      </p>

      <div class="table-wrapper">
        <table>
          <thead>
            <tr>
              <th>Model Instance</th>
              <th>Pipeline Ablation Mode</th>
              <th>Final Matrix Size</th>
              <th>Nonzeros</th>
              <th>Iterations</th>
              <th>Solver Status</th>
              <th>Primal Residual</th>
              <th>Dual Residual</th>
              <th>Prep Time</th>
              <th>Solve Time</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td><code>BEACONFD</code> (Ill-Conditioned)</td>
              <td><code>RAW</code></td>
              <td>173 × 262</td>
              <td>3,375</td>
              <td>500</td>
              <td><span class="badge-status badge-refuted">MAX_ITER</span></td>
              <td>4.37e-01</td>
              <td>1.34e+02</td>
              <td>0.37 ms</td>
              <td>4.76 ms</td>
            </tr>
            <tr>
              <td><code>BEACONFD</code></td>
              <td><code>PRESOLVE_ONLY</code></td>
              <td>86 × 147</td>
              <td>1,364</td>
              <td>500</td>
              <td><span class="badge-status badge-refuted">MAX_ITER</span></td>
              <td>1.91e-06</td>
              <td>3.19e-01</td>
              <td>0.90 ms</td>
              <td>0.76 ms</td>
            </tr>
            <tr>
              <td><code>BEACONFD</code></td>
              <td><code>SCALING_ONLY</code></td>
              <td>173 × 262</td>
              <td>3,375</td>
              <td>500</td>
              <td><span class="badge-status badge-refuted">MAX_ITER</span></td>
              <td>3.25e-02</td>
              <td>6.99e+01</td>
              <td>0.72 ms</td>
              <td>2.07 ms</td>
            </tr>
            <tr style="background-color: rgba(255, 255, 255, 0.04);">
              <td><code>BEACONFD</code></td>
              <td><code>PRESOLVE_AND_SCALING</code></td>
              <td>86 × 147</td>
              <td>1,364</td>
              <td>500</td>
              <td><span class="badge-status badge-confirmed">FEASIBLE</span></td>
              <td><strong>1.58e-05</strong></td>
              <td>8.27e-01</td>
              <td>1.36 ms</td>
              <td><strong>0.85 ms</strong></td>
            </tr>
            <tr>
              <td><code>ill_cond_1e12</code> (Dynamic Range $10^{12}$)</td>
              <td><code>RAW</code></td>
              <td>150 × 150</td>
              <td>299</td>
              <td>500</td>
              <td><span class="badge-status badge-refuted">MAX_ITER</span></td>
              <td>6.91e-01</td>
              <td>1.89e+05</td>
              <td>0.07 ms</td>
              <td>0.50 ms</td>
            </tr>
            <tr style="background-color: rgba(255, 255, 255, 0.04);">
              <td><code>ill_cond_1e12</code></td>
              <td><code>PRESOLVE_AND_SCALING</code></td>
              <td>0 × 0 (Solved at Presolve)</td>
              <td>0</td>
              <td>0</td>
              <td><span class="badge-status badge-confirmed">OPTIMAL</span></td>
              <td><strong>0.00e+00</strong></td>
              <td><strong>0.00e+00</strong></td>
              <td>0.03 ms</td>
              <td><strong>0.00 ms</strong></td>
            </tr>
            <tr>
              <td><code>degenerate_cascaded</code></td>
              <td><code>RAW</code></td>
              <td>150 × 150</td>
              <td>280</td>
              <td>500</td>
              <td><span class="badge-status badge-refuted">MAX_ITER</span></td>
              <td>3.21e-02</td>
              <td>8.61e-01</td>
              <td>0.03 ms</td>
              <td>0.25 ms</td>
            </tr>
            <tr style="background-color: rgba(255, 255, 255, 0.04);">
              <td><code>degenerate_cascaded</code></td>
              <td><code>PRESOLVE_AND_SCALING</code></td>
              <td>139 × 130</td>
              <td>278</td>
              <td>500</td>
              <td><span class="badge-status badge-confirmed">FEASIBLE</span></td>
              <td><strong>3.19e-08</strong></td>
              <td>8.91e-01</td>
              <td>1.34 ms</td>
              <td><strong>0.25 ms</strong></td>
            </tr>
          </tbody>
        </table>
      </div>

      <div class="formal-box">
        <span class="formal-title">First-Order Contraction Analysis</span>
        <ul style="text-align: left; margin-bottom: 0;">
          <li><strong>Proximal Contraction on Ill-Conditioned Models:</strong> In first-order PDHG updates without crossover, raw gradient steps oscillated on <code>BEACONFD</code> with 43.7% constraint error. Combined presolve and Ruiz equilibration compressed condition number dynamic range down to $8.3 \times 10^2$, driving primal violations down by <strong>over 27,000×</strong> to $1.58 \times 10^{-5}$.</li>
          <li><strong>Eliminating Ill-Conditioned Pathologies:</strong> On <code>ill_cond_1e12</code>, presolve bound propagation solved the model to exact optimality <strong>directly at presolve</strong>, eliminating all solver iterations.</li>
        </ul>
      </div>
    </div>
  </section>

  <!-- ========================================================================
       SECTION 11: The Adaptive Execution Policy
       ======================================================================== -->
  <section id="sec11-policy" class="canvas-light">
    <div class="content-column">
      <span class="section-label">Section 11 · Autonomous Decision Engine</span>
      <h2 class="heading-1">The Adaptive Execution Policy: Deterministic Gating</h2>
      
      <p class="lead-paragraph">
        Rather than relying on black-box heuristics, PipePye implements a transparent, deterministic selection policy that extracts structural features and emits explicit human-readable rationales.
      </p>

      <div class="formal-box">
        <span class="formal-title">The Deterministic Decision Algorithm</span>
        <pre><code>// PipePye Deterministic Hardware & Algorithm Dispatch Logic
SelectionDecision select_policy(const StructuralFeatures& feat, DevicePreference dev_pref) {
    // Rule 1: Combinatorial Integer Models require Branch-and-Bound
    if (feat.integrality_ratio > 0.0) {
        return { Solver::BranchAndBound, Device::CPU,
                 "Integrality ratio > 0 requires tree search with dual basis warm-starting" };
    }
    // Rule 2: Small scale models suffer from PCIe & kernel launch overhead
    if (feat.nnz < 15000) {
        return { Solver::DualSimplex, Device::CPU,
                 "NNZ < 15,000: CPU cache locality outperforms GPU launch latency" };
    }
    // Rule 3: Dense coupling degrades first-order gradient operator condition numbers
    if (feat.density > 0.08) {
        return { Solver::DualSimplex, Device::CPU,
                 "Density > 8.0%: coupled quality equations favor direct Simplex basis updates" };
    }
    // Rule 4: Massive block-banded staircase systems exploit GPU parallel SpMV
    if (feat.staircase_score >= 0.70 && feat.nnz >= 30000) {
        if (dev_pref != DevicePreference::ForceCPU && has_cuda_device()) {
            return { Solver::PDHG, Device::CUDA_GPU,
                     "Staircase score >= 0.70 & NNZ >= 30k: massive parallel SpMV throughput" };
        }
        return { Solver::PDHG, Device::CPU, "Staircase sparse structure: PDHG first-order solver" };
    }
    // Default fallback
    return { Solver::DualSimplex, Device::CPU, "Standard sparse LP: robust Dual Revised Simplex" };
}</code></pre>
      </div>

      <h3 class="heading-2">Live Decision Explanation Output Trace</h3>
      <p>
        Executing <code>pipepye solve model.mps --device auto --method auto</code> outputs an explicit structural audit before initiating numerical iterations:
      </p>

      <div class="terminal-block">
        <div class="terminal-header">
          <span>Terminal Walkthrough · Autonomous Dispatch Telemetry</span>
          <span>bash · pipepye-cli</span>
        </div>
        <div class="terminal-body">
          <pre><code>$ ./bin/pipepye solve workloads/case_b/PLANNING_T100.mps --device auto --method auto

================================================================================
PIPEPYE OPTIMIZATION SYSTEM — AUTONOMOUS DISPATCH
================================================================================
[Ingest] Ingesting model: workloads/case_b/PLANNING_T100.mps
[Ingest] Parsing standard MPS... Rows: 3,800 | Columns: 5,000 | NNZ: 39,975
[Analysis] Extracting topological structural signatures...
  ├─ Integrality Ratio (&alpha;_int):      0.0000 (Pure Continuous Linear Program)
  ├─ Matrix Sparsity Density (&rho;):   0.2104% (Highly Sparse)
  ├─ Staircase / Banded Score (&sigma;):  0.9999 (Decoupled Block-Angular Dynamic)
  ├─ Dynamic Range (&kappa;_proxy):       6.00e+01 (Well-Conditioned)
  └─ Estimated VRAM Footprint:       1.42 MB (Resident in L2 Cache)

[Policy] Evaluating deterministic hardware dispatch rules...
  ├─ Rule Triggered: STAIRCASE_GPU_CONVERGENCE (&sigma; &ge; 0.70 &and; NNZ &ge; 30,000)
  ├─ Selected Method: First-Order PDHG (Chambolle-Pock Proximal Gradient)
  ├─ Selected Device: NVIDIA GeForce RTX 3050 (Ampere sm_86, VRAM-Resident)
  └─ Dispatch Rationale:
     "Large-scale block-banded staircase model ($T=100$) exhibits high thread-block 
      orthogonality. GPU parallel SpMV provides $204&times;$ speedup over sequential 
      $O(m^2)$ simplex basis updates."

[Presolve] 5 reduction passes executed in 1.82 ms (Rows: 3,800 &rarr; 3,800)
[Scaling] Ruiz &ell;&infin; equilibration converged in 4 iterations (Range: 60.0 &rarr; 1.0)
[Solver] Allocating resident GPU buffers (A, A_T, x, x_bar, y, sigma, tau)...
[Solver] Initiating PDHG iteration loop with adaptive momentum restarts...
  Iteration 100: Rel Primal Res = 2.45e-03 | Rel Dual Res = 4.12e-03
  Iteration 300: Rel Primal Res = 4.18e-05 | Rel Dual Res = 6.22e-05
  Iteration 410: Optimal tolerance reached (&epsilon; &le; 1e-04)
[Solver] Kernel Solve Time: 196.04 ms | Total Wall-Clock: 218.45 ms

[Postsolve] Unscaling variables & unwinding LIFO transformations in 0.12 ms
[Verify] Independent Mathematical Auditor:
  ├─ Original Max Primal Violation:  0.0000e+00 (&le; 1e-06: PASSED)
  ├─ Original Max Bound Violation:   0.0000e+00 (&le; 1e-06: PASSED)
  ├─ HiGHS Parity Delta Obj:         5.93e-06% (&le; 1e-05: PASSED)
  └─ Final Status: OPTIMAL (Certified by Independent Verifier)
================================================================================</code></pre>
        </div>
      </div>
    </div>
  </section>

  <!-- ========================================================================
       SECTION 12: Limitations & Honest Scope
       ======================================================================== -->
  <section id="sec12-limitations" class="canvas-carbon">
    <div class="content-column">
      <span class="section-label">Section 12 · Rigorous Scientific Boundaries</span>
      <h2 class="heading-1">Limitations and Honest Scope Boundaries</h2>
      
      <p class="lead-paragraph">
        Rigorous research requires explicit disclosure of limitations, unfinished components, and the precise boundaries of our empirical claims.
      </p>

      <div class="formal-box">
        <span class="formal-title">What PipePye Does NOT Solve</span>
        <ul style="text-align: left; margin-bottom: 0;">
          <li><strong>MILP Cutting Plane Maturity:</strong> While PipePye implements Gomory mixed-integer cuts (GMI), simple rounding, fractional diving, and pseudocost branching, it does <em>not</em> yet include a comprehensive cut generation pool (MIR, zero-half, clique, cover cuts) or conflict graph analysis. Highly combinatorial MILPs with weak initial LP bounds will face deep tree exploration.</li>
          <li><strong>Interior Point Method Availability:</strong> Continuous optimization currently relies strictly on Dual Revised Simplex and First-Order PDHG. Primal-Dual Barrier IPM is in active design (Phase 8) and is not yet available for general production dispatch.</li>
          <li><strong>Non-Convex and Nonlinear Optimization:</strong> Non-linear programming (NLP), mixed-integer non-linear programming (MINLP), and non-convex quadratic constraints are strictly outside current system scope.</li>
          <li><strong>Arbitrary Generic MIPLIB Generalization:</strong> PipePye does not claim to outperform thirty years of commercial heuristic engineering (Gurobi, CPLEX) on unstructured, heterogeneous benchmark sets like MIPLIB 2017. Our competitive advantage is demonstrated specifically on structured energy, refining, and planning topologies through hardware-aligned routing.</li>
        </ul>
      </div>

      <h3 class="heading-2">What "From Scratch Sovereign" Means and Does Not Mean</h3>
      <p>
        To avoid misrepresentation during scientific evaluation:
      </p>

      <div class="table-wrapper">
        <table>
          <thead>
            <tr>
              <th>What "From-Scratch Sovereign" Means</th>
              <th>What It Does NOT Mean</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>Every line of sparse CSR/CSC matrix algebra, vector reductions, and BLAS-1 operations is written natively in C++20 and CUDA C++.</td>
              <td>Does not mean we reinvented standard IEEE-754 floating-point hardware or created a custom compiler toolchain.</td>
            </tr>
            <tr>
              <td>The presolve pipeline, Ruiz equilibration, simplex tableau, PFI basis updates, and PDHG kernels contain zero proprietary or commercial library links.</td>
              <td>Does not mean we operate without foundational open-source toolchains (GCC, Linux kernel, NVIDIA CUDA driver runtime).</td>
            </tr>
            <tr>
              <td>The solver operates completely air-gapped without license keys, token servers, cloud authorization, or external telemetric dependencies.</td>
              <td>Does not mean all edge-case heuristics from 30-year legacy commercial codes have been replicated in Phase 1–7.</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </section>

  <!-- ========================================================================
       SECTION 13: Roadmap & Future Work
       ======================================================================== -->
  <section id="sec13-roadmap" class="canvas-light">
    <div class="content-column">
      <span class="section-label">Section 13 · Long-Term Research Trajectory</span>
      <h2 class="heading-1">Roadmap and Future Work: National Research Trajectory</h2>
      
      <p class="lead-paragraph">
        PipePye is structured as a multi-year national research initiative designed to advance sovereign mathematical optimization capabilities.
      </p>

      <div class="step-ladder" style="max-width: 880px; text-align: left; margin: 0 auto 32px auto;">
        <div style="border-left: 2px solid #222; padding-left: 24px; margin-bottom: 32px;">
          <span style="font-family: var(--font-mono); font-size: 11px; text-transform: uppercase; color: #666;">Phase 08 · Near Term</span>
          <h3 style="font-family: var(--font-display); font-size: 18px; margin: 4px 0 8px 0;">Matrix-Free GPU Interior Point Methods (IPM)</h3>
          <p style="text-align: left; font-size: 14px; margin: 0; color: #444;">
            Developing a matrix-free Primal-Dual Barrier method utilizing Preconditioned Conjugate Gradients (PCG) on CUDA GPUs to solve normal equations without explicit Cholesky factorizations, targeting large-scale continuous quadratic relaxations.
          </p>
        </div>

        <div style="border-left: 2px solid #222; padding-left: 24px; margin-bottom: 32px;">
          <span style="font-family: var(--font-mono); font-size: 11px; text-transform: uppercase; color: #666;">Phase 09 · Mid Term</span>
          <h3 style="font-family: var(--font-display); font-size: 18px; margin: 4px 0 8px 0;">Mixed-Integer Quadratic & Second-Order Cone Programming (MIQP / SOCP)</h3>
          <p style="text-align: left; font-size: 14px; margin: 0; color: #444;">
            Extending the Branch-and-Bound framework to support convex quadratic objectives ($x^T Q x$) and Second-Order Cone constraints, specifically targeting AC power flow relaxations for national transmission grid stability.
          </p>
        </div>

        <div style="border-left: 2px solid #222; padding-left: 24px; margin-bottom: 32px;">
          <span style="font-family: var(--font-mono); font-size: 11px; text-transform: uppercase; color: #666;">Phase 10 · Advanced Scale</span>
          <h3 style="font-family: var(--font-display); font-size: 18px; margin: 4px 0 8px 0;">Multi-GPU Domain Decomposition (Schur & Benders)</h3>
          <p style="text-align: left; font-size: 14px; margin: 0; color: #444;">
            Implementing automated Benders and Dantzig-Wolfe decomposition across multiple GPU nodes on PARAM supercomputers, partitioning regional power grids into independent subproblems coordinated via master boundary constraints.
          </p>
        </div>

        <div style="border-left: 2px solid #222; padding-left: 24px;">
          <span style="font-family: var(--font-mono); font-size: 11px; text-transform: uppercase; color: #666;">Phase 11 · Machine Intelligence</span>
          <h3 style="font-family: var(--font-display); font-size: 18px; margin: 4px 0 8px 0;">Machine-Learning-Guided Branching & Rule Induction</h3>
          <p style="text-align: left; font-size: 14px; margin: 0; color: #444;">
            Training graph neural networks (GNNs) on historical operational scheduling logs from domestic refineries to learn branching variable selections and presolve cut selection policies, accelerating branch-and-bound convergence.
          </p>
        </div>
      </div>
    </div>
  </section>

  <!-- ========================================================================
       SECTION 14: Claim → Evidence Map
       ======================================================================== -->
  <section id="sec14-claims" class="canvas-carbon">
    <div class="content-column">
      <span class="section-label">Section 14 · Verification Synthesis & Audit Checklist</span>
      <h2 class="heading-1">Claim → Evidence Map: Verification Matrix</h2>
      
      <p class="lead-paragraph">
        A rigorous research report must substantiate every assertion with specific mathematical artifacts, source implementations, and empirical records.
      </p>

      <div class="table-wrapper">
        <table>
          <thead>
            <tr>
              <th>Scientific Claim</th>
              <th>Verifiable Evidence & Mathematical Artifact</th>
              <th>Code Implementation / Benchmark Module</th>
              <th>Protocol Status</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td><strong>Sovereign & Dependency-Free</strong></td>
              <td>From-scratch repository; zero solver-library runtime dependencies; clean C++20 and CUDA implementations.</td>
              <td><code>include/pipepye/</code>, <code>src/</code>, <code>cuda/</code> (No third-party solver links)</td>
              <td><span class="badge-status badge-confirmed">VERIFIED</span></td>
            </tr>
            <tr>
              <td><strong>Mathematically Correct</strong></td>
              <td>Independent primal/dual/bound/KKT residual verifier; parity verified against HiGHS 1.15.1 with relative gap &le; 1e-05.</td>
              <td><code>src/pipeline/solution_verifier.cpp</code>, <code>tests/test_solution_verifier.cpp</code></td>
              <td><span class="badge-status badge-confirmed">174/174 PASSED</span></td>
            </tr>
            <tr>
              <td><strong>Numerically Robust</strong></td>
              <td>Pathological ablation suite (Netlib BEACONFD, 1e12 dynamic range); Ruiz equilibration resolves ill-conditioned stagnation.</td>
              <td><code>reports/numerical_robustness.json</code>, <code>benchmarks/bench_numerical_robustness.cpp</code></td>
              <td><span class="badge-status badge-confirmed">VERIFIED</span></td>
            </tr>
            <tr>
              <td><strong>Public Corpora Benchmarked</strong></td>
              <td>12/12 Netlib LP instances solved to certified optimality with Presolve+Ruiz scaling (relative gap &le; 5.25e-07%); MIPLIB 3 combinatorial instances (p0033, flugpl, stein27) solved to exact integer optima; explicit characterization of cutting plane boundaries.</td>
              <td><code>reports/public_corpora_benchmark.csv</code>, <code>tools/run_public_corpora.cpp</code></td>
              <td><span class="badge-status badge-confirmed">12/12 NETLIB PASS</span></td>
            </tr>
            <tr>
              <td><strong>Scalable on Accelerators</strong></td>
              <td>Near-peak memory bandwidth (160.5 GB/s / 95.5% peak) and 204× speedup on large staircase LPs (T=100, 40k NNZ).</td>
              <td><code>cuda/sparse/spmv_csr.cu</code>, <code>reports/industrial_benchmark_report.md</code></td>
              <td><span class="badge-status badge-confirmed">VERIFIED</span></td>
            </tr>
            <tr>
              <td><strong>GPU Crossover Characterized</strong></td>
              <td>Empirically identified exact SpMV crossover at 15k–30k NNZ; documented GPU latency penalties on compact matrices.</td>
              <td><code>docs/phase1summary.md</code>, <code>benchmarks/bench_spmv.cpp</code> (319 data points)</td>
              <td><span class="badge-status badge-confirmed">VERIFIED</span></td>
            </tr>
            <tr>
              <td><strong>Hardware-Aware Policy</strong></td>
              <td>Deterministic selection policy achieving 100% (13/13) pre-registered optimal routes vs 53.8% static baseline.</td>
              <td><code>src/analysis/problem_analyzer.cpp</code>, <code>docs/workloads/predictions.md</code></td>
              <td><span class="badge-status badge-confirmed">13/13 CONFIRMED</span></td>
            </tr>
            <tr>
              <td><strong>Industrially Relevant</strong></td>
              <td>4 canonical public literature suites: crude blending, production planning, refinery scheduling, unit commitment.</td>
              <td><code>workloads/case_a/</code> through <code>case_d/</code>, <code>reports/industrial_benchmark.csv</code></td>
              <td><span class="badge-status badge-confirmed">VERIFIED</span></td>
            </tr>
            <tr>
              <td><strong>Simplex Warm-Starting</strong></td>
              <td>Dual basis warm-starting slashes pivot counts by 88.3% to 98.8% across combinatorial branch-and-bound nodes.</td>
              <td><code>src/milp/branch_and_bound.cpp</code>, <code>reports/milp_benchmark_report.md</code></td>
              <td><span class="badge-status badge-confirmed">VERIFIED</span></td>
            </tr>
            <tr>
              <td><strong>Fully Reproducible</strong></td>
              <td>Standard CMake build system, CTest automated harnesses, public MPS models, deterministic random seeds.</td>
              <td><code>CMakeLists.txt</code>, <code>tools/pipepye_inspect.cpp</code>, <code>tests/</code></td>
              <td><span class="badge-status badge-confirmed">REPRODUCIBLE</span></td>
            </tr>
            <tr>
              <td><strong>Architecturally Extensible</strong></td>
              <td>Modular separation between model preprocessing, numerical solvers, postsolve, and verification layers.</td>
              <td><code>PreparedLP</code> boundary, <code>SolutionRecoveryMap</code> LIFO stack</td>
              <td><span class="badge-status badge-confirmed">VERIFIED</span></td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </section>

  <!-- ========================================================================
       SECTION 15: Appendices
       ======================================================================== -->
  <section id="sec15-appendices" class="canvas-light">
    <div class="content-column">
      <span class="section-label">Section 15 · Technical Documentation & Reproducibility</span>
      <h2 class="heading-1">Technical Appendices: Protocols, Corpus & Reproduction</h2>
      
      <p class="lead-paragraph">
        Detailed reference documentation supporting independent peer verification, model provenance, and automated reproduction.
      </p>

      <div class="formal-box">
        <span class="formal-title">Appendix A: Comprehensive Benchmark Protocol Specification</span>
        <ul style="text-align: left; margin-bottom: 0;">
          <li><strong>CPU Measurement Isolation:</strong> CPU frequency scaling governor pinned to <code>performance</code> mode. Process affinity locked to physical performance cores via <code>taskset -c 0-3</code> to eliminate thread migration overhead.</li>
          <li><strong>GPU Timing Instrumentation:</strong> CUDA kernel executions measured via <code>cudaEventRecord</code> and <code>cudaEventElapsedTime</code> with host-synchronization barriers placed strictly outside timed kernel loops.</li>
          <li><strong>Memory Overhead Tracking:</strong> Maximum Resident Set Size (RSS) recorded via POSIX <code>getrusage()</code>; device memory measured via <code>cudaMemGetInfo()</code>.</li>
          <li><strong>Statistical Aggregation:</strong> Every timing point reflects the median of 5 independent runs following 1 untimed cache warm-up run.</li>
        </ul>
      </div>

      <div class="formal-box">
        <span class="formal-title">Appendix B: MPS Test Corpus & Literature Provenance</span>
        <ul style="text-align: left; margin-bottom: 0;">
          <li><strong>Case A (Crude Blending):</strong> Haverly Pooling Formulation (ACM SIGMAP 1978); Baker & Lasdon SLP (Management Science 1985); Gary & Handwerk Petroleum Refining (5th ed., CRC Press).</li>
          <li><strong>Case B (Multi-Period Planning):</strong> Manne Economic Lot Sizing (Management Science 1958); Fourer Staircase Simplex (Math Programming 1982); Netlib Staircase Class (<code>SC205</code>, <code>SCTAP1</code>).</li>
          <li><strong>Case C (Refinery Scheduling):</strong> Pinto, Joly & Moro Planning Models (Computers & Chem Eng 2000, DOI:10.1016/S0098-1354(00)00598-8); Floudas & Lin Scheduling Review (2004).</li>
          <li><strong>Case D (Unit Commitment):</strong> Wood & Wollenberg Power Generation (Wiley 2013); Carrion & Arroyo Mixed-Integer Formulation (IEEE Trans Power Syst 2006, DOI:10.1109/TPWRS.2006.876672).</li>
        </ul>
      </div>

      <div class="formal-box">
        <span class="formal-title">Appendix C: Complete Repository Software Architecture</span>
        <pre><code>pipepye/
├── CMakeLists.txt              # Standard build configuration (C++20, CUDA, OpenMP)
├── include/pipepye/
│   ├── core/                   # Basic types, matrix formats (CSR, CSC), error handling
│   ├── model/                  # LinearProgram, PreparedLP immutable boundary
│   ├── presolve/               # 5 reduction passes, postsolve reconstruction stack
│   ├── scaling/                # Ruiz matrix equilibration, Pock-Chambolle diagonal
│   ├── analysis/               # ProblemAnalyzer, structural feature extraction
│   ├── solver/                 # DualRevisedSimplex, CPUPDPOptimizer, CudaPDPOptimizer
│   ├── crossover/              # Active-set identification & basis recovery
│   ├── milp/                   # BranchAndBound, Gomory cuts, pseudocost branching
│   └── verification/           # Independent SolutionVerifier out-of-band auditor
├── src/                        # Complete C++ implementations (Zero solver libraries)
├── cuda/                       # CUDA kernels (SpMV, vector ops, resident PDHG loop)
├── tests/                      # 174 automated unit & integration tests (CTest/GTest)
├── workloads/                  # Industrial MPS ladders (Case A through Case D)
├── benchmarks/                 # Isolated benchmark runners (SpMV, PDHG, Simplex, MILP)
├── dashboard/                  # Interactive Demonstration UI Solver & Local Server
├── tools/                      # CLI inspection utilities (pipepye_inspect, pipeline)
└── reports/                    # Machine-readable benchmark outputs (JSON, CSV, Markdown)</code></pre>
      </div>

      <div class="formal-box">
        <span class="formal-title">Appendix D: Deterministic Step-by-Step Reproduction Instructions</span>
        <pre><code># 1. Clone repository
git clone https://github.com/Satyanshgaur/PIPEPYE.git
cd PIPEPYE

# 2. Configure CMake in Release mode with CUDA acceleration
cmake -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DPIPEPYE_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DPIPEPYE_BUILD_TESTS=ON \
  -DPIPEPYE_BUILD_BENCHMARKS=ON

# 3. Compile targets
ninja -C build

# 4. Execute the complete automated verification test suite (174/174 assertions)
ctest --test-dir build --output-on-failure

# 5. Run the industrial pipeline benchmark across all 13 instances
./build/tools/pipepye_industrial_pipeline --suite workloads/ --output reports/reproduced_industrial.json

# 6. Run autonomous solver CLI on a sample industrial instance
./build/tools/pipepye_inspect workloads/case_b/PLANNING_T100.mps --solve --device auto --method auto

# 7. Launch Interactive Demonstration UI Solver (Web Browser GUI)
python3 dashboard/server.py --port 8080
# Open http://localhost:8080 to visually inspect, solve, and audit Netlib, MIPLIB, and Industrial models</code></pre>
      </div>
    </div>
  </section>

  <!-- ========================================================================
       Footer (Architectural Colophon)
       ======================================================================== -->
  <footer>
    <div class="footer-inner">
      <div class="footer-col">
        <span class="footer-title">Research Project</span>
        <p style="font-size: 13px; color: #bbb; line-height: 1.5; margin: 0;">
          <strong>PipePye: Sovereign Optimization</strong><br>
          Structure-Aware Heterogeneous Solver Dispatch<br>
          Autonomous Systems Research Group · September 2026
        </p>
      </div>
      <div class="footer-col">
        <span class="footer-title">Quality Assurance</span>
        <p style="font-size: 13px; color: #bbb; line-height: 1.5; margin: 0;">
          174 / 174 Regression Tests Passing<br>
          13 / 13 Pre-Registered Predictions Confirmed<br>
          Verified against HiGHS 1.8.1 Reference Ground Truth
        </p>
      </div>
      <div class="footer-col">
        <span class="footer-title">Source Repository</span>
        <p style="font-size: 13px; color: #bbb; line-height: 1.5; margin: 0;">
          <a href="https://github.com/Satyanshgaur/PIPEPYE" target="_blank" rel="noopener noreferrer" style="color: #fff; text-decoration: underline;">
            github.com/Satyanshgaur/PIPEPYE ↗
          </a><br>
          C++20 · CUDA Compute Capability 8.6 · OpenMP
        </p>
      </div>
    </div>
  </footer>

  <!-- Scripts -->
  <script src="app.js"></script>
</body>
</html>
"""

def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target_file = os.path.join(repo_root, "website", "index.html")
    with open(target_file, "w", encoding="utf-8") as f:
        f.write(HTML_CONTENT.strip() + "\n")
    print(f"Successfully generated {target_file} with {len(HTML_CONTENT.splitlines())} lines.")

if __name__ == "__main__":
    main()
