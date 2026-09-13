#!/usr/bin/env python3
"""
PipePye Automated Benchmark Plotting Engine
Generates publication-quality figures from benchmark JSON/CSV datasets:
1. runtime_vs_nnz.png - Crossover plot (runtime vs NNZ, log-log)
2. speedup_vs_problem_size.png - Speedup vs CPU 1T across problem size
3. performance_vs_sparsity.png - Effective memory bandwidth vs matrix density
4. structure_comparison.png - Topologies & irregular structures comparison
5. reductions_bandwidth.png - Vector reductions bandwidth scaling vs hardware peak
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Style configuration for clean, modern scientific figures
plt.rcParams.update({
    'font.sans-serif': 'DejaVu Sans',
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'lines.linewidth': 2.0,
    'lines.markersize': 7,
    'grid.alpha': 0.4,
    'grid.linestyle': '--',
    'axes.grid': True,
    'savefig.dpi': 300
})

COLORS = {
    'CPU_1T': '#4a5568',        # Slate Gray
    'CPU_12T': '#3182ce',       # Blue
    'GPU_Scalar': '#e53e3e',     # Red
    'GPU_Vector': '#dd6b20',     # Orange
    'GPU_Adaptive': '#38a169',   # Green
    'GPU_Balanced': '#805ad5',   # Purple
    'CUDA': '#2b6cb0',          # Dark Blue for reductions
}

LABELS = {
    'CPU_1T': 'CPU (1 Thread)',
    'CPU_12T': 'CPU (12 Threads OpenMP)',
    'GPU_Scalar': 'GPU Scalar (1 thr/row)',
    'GPU_Vector': 'GPU Vector (32 thr/row)',
    'GPU_Adaptive': 'GPU Adaptive (sub-warp 8)',
    'GPU_Balanced': 'GPU Balanced (partitioned)',
    'CUDA': 'GPU CUDA (Warp/Block Tree)'
}

MARKERS = {
    'CPU_1T': 'o',
    'CPU_12T': 's',
    'GPU_Scalar': '^',
    'GPU_Vector': 'D',
    'GPU_Adaptive': 'v',
    'GPU_Balanced': 'P',
    'CUDA': 'o'
}

def load_data(results_dir):
    json_path = os.path.join(results_dir, 'benchmark_results.json')
    red_json_path = os.path.join(results_dir, 'reductions_results.json')

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Matrix benchmark file not found: {json_path}")
    if not os.path.exists(red_json_path):
        raise FileNotFoundError(f"Reductions benchmark file not found: {red_json_path}")

    with open(json_path, 'r') as f:
        matrix_data = json.load(f)
    with open(red_json_path, 'r') as f:
        red_data = json.load(f)

    return matrix_data, red_data


def plot_runtime_vs_nnz(matrix_records, output_path):
    """Plot 1: Crossover Analysis - Execution Runtime vs NNZ (log-log)."""
    fig, ax = plt.subplots(figsize=(9.5, 6))

    records = [r for r in matrix_records if r.get('experiment_type') == 'nnz_scaling']
    variants = ['CPU_1T', 'CPU_12T', 'GPU_Scalar', 'GPU_Vector', 'GPU_Adaptive', 'GPU_Balanced']
    
    for v in variants:
        v_recs = [r for r in records if r.get('variant') == v]
        if not v_recs:
            continue
        v_recs.sort(key=lambda x: x['nnz'])
        nnz = [r['nnz'] for r in v_recs]
        runtime = [r['runtime_ms'] for r in v_recs]
        ax.plot(nnz, runtime, label=LABELS.get(v, v), color=COLORS.get(v, '#333333'),
                marker=MARKERS.get(v, 'o'), linewidth=2.2)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Matrix Nonzeros (NNZ)')
    ax.set_ylabel('Execution Time (ms)')
    ax.set_title('SpMV Execution Time vs Problem Size (NNZ) — Crossover Analysis')
    
    # Highlight crossover zone
    ax.axvspan(10000, 30000, color='orange', alpha=0.15, label='GPU Crossover Zone (~15k-30k NNZ)')
    
    ax.grid(True, which="both", ls="--", alpha=0.3)
    ax.legend(loc='upper left', framealpha=0.92)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"  [Saved] {output_path}")


def plot_speedup_vs_problem_size(matrix_records, output_path):
    """Plot 2: Speedup vs CPU 1T across problem size."""
    fig, ax = plt.subplots(figsize=(9.5, 6))

    records = [r for r in matrix_records if r.get('experiment_type') == 'nnz_scaling']
    variants = ['CPU_12T', 'GPU_Scalar', 'GPU_Vector', 'GPU_Adaptive', 'GPU_Balanced']

    for v in variants:
        v_recs = [r for r in records if r.get('variant') == v]
        if not v_recs:
            continue
        v_recs.sort(key=lambda x: x['nnz'])
        nnz = [r['nnz'] for r in v_recs]
        speedup = [r['speedup_vs_cpu_1t'] for r in v_recs]
        ax.plot(nnz, speedup, label=LABELS.get(v, v), color=COLORS.get(v, '#333333'),
                marker=MARKERS.get(v, 'o'), linewidth=2.2)

    ax.axhline(1.0, color='#4a5568', linestyle='--', linewidth=1.8, label='CPU 1-Thread Baseline (1.0x)')
    ax.set_xscale('log')
    ax.set_xlabel('Matrix Nonzeros (NNZ)')
    ax.set_ylabel('Speedup over CPU 1-Thread (x)')
    ax.set_title('SpMV Acceleration vs CPU Single-Thread by Problem Size')
    ax.grid(True, which="both", ls="--", alpha=0.3)
    ax.legend(loc='upper left', framealpha=0.92)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"  [Saved] {output_path}")


def plot_performance_vs_sparsity(matrix_records, output_path):
    """Plot 3: Effective Memory Bandwidth vs Matrix Density for 10k x 10k matrix."""
    fig, ax = plt.subplots(figsize=(9.5, 6))

    records = [r for r in matrix_records if r.get('experiment_type') == 'density_scaling']
    variants = ['CPU_1T', 'CPU_12T', 'GPU_Scalar', 'GPU_Vector', 'GPU_Adaptive', 'GPU_Balanced']

    for v in variants:
        v_recs = [r for r in records if r.get('variant') == v]
        if not v_recs:
            continue
        v_recs.sort(key=lambda x: x['density_pct'])
        density = [r['density_pct'] for r in v_recs]
        bw = [r['bandwidth_gbs'] for r in v_recs]
        ax.plot(density, bw, label=LABELS.get(v, v), color=COLORS.get(v, '#333333'),
                marker=MARKERS.get(v, 'o'), linewidth=2.2)

    # Theoretical peak bandwidth line for RTX 3050 (~168 GB/s)
    ax.axhline(168.0, color='red', linestyle=':', linewidth=1.5, alpha=0.7, label='GPU Theoretical Bus (168 GB/s)')

    ax.set_xlabel('Matrix Density (%) [Fixed 10,000 x 10,000 Matrix]')
    ax.set_ylabel('Effective Memory Bandwidth (GB/s)')
    ax.set_title('SpMV Memory Throughput vs Sparsity / Density')
    ax.grid(True, ls="--", alpha=0.3)
    ax.legend(loc='lower right', framealpha=0.92)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"  [Saved] {output_path}")


def plot_structure_comparison(matrix_records, output_path):
    """Plot 4: Performance comparison across structural topologies and real LP instances."""
    fig, ax = plt.subplots(figsize=(13, 6.5))

    recs = [r for r in matrix_records if r.get('experiment_type') in ('structural_topology', 'netlib')]
    
    matrix_order = []
    for r in recs:
        name = r['matrix_name']
        if name not in matrix_order:
            matrix_order.append(name)

    variants = ['CPU_1T', 'CPU_12T', 'GPU_Scalar', 'GPU_Vector', 'GPU_Adaptive', 'GPU_Balanced']
    
    x = np.arange(len(matrix_order))
    bar_width = 0.13
    
    for idx, v in enumerate(variants):
        bws = []
        for m in matrix_order:
            matching = [r for r in recs if r['matrix_name'] == m and r['variant'] == v]
            if matching:
                bws.append(matching[0]['bandwidth_gbs'])
            else:
                bws.append(0.0)
        
        offset = (idx - len(variants)/2.0 + 0.5) * bar_width
        ax.bar(x + offset, bws, width=bar_width, label=LABELS.get(v, v), color=COLORS.get(v, '#333333'))

    ax.set_ylabel('Effective Memory Bandwidth (GB/s)')
    ax.set_title('SpMV Performance Across Sparse Topologies and Real-World Netlib LP Instances')
    ax.set_xticks(x)
    
    labels = []
    for m in matrix_order:
        matching = [r for r in recs if r['matrix_name'] == m]
        topo = matching[0]['topology'] if matching else ""
        nnz = matching[0]['nnz'] if matching else 0
        labels.append(f"{m}\n({topo})\n{nnz:,} NNZ")
    
    ax.set_xticklabels(labels, rotation=0, ha='center', fontsize=8.5)
    ax.legend(loc='upper right', framealpha=0.92)
    ax.grid(True, axis='y', ls="--", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"  [Saved] {output_path}")


def plot_reductions_bandwidth(reduction_records, output_path):
    """Plot 5: Vector reductions bandwidth scaling vs hardware peak."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Dot Product Scaling
    dot_recs = [r for r in reduction_records if r.get('operation') == 'dot']
    sizes = sorted(list(set(r['size'] for r in dot_recs)))
    
    for v in ['CPU_1T', 'CPU_12T', 'CUDA']:
        v_recs = [r for r in dot_recs if r.get('variant') == v]
        v_recs.sort(key=lambda x: x['size'])
        sz = [r['size'] for r in v_recs]
        bw = [r['bandwidth_gbs'] for r in v_recs]
        col = COLORS.get(v, '#2b6cb0')
        ax1.plot(sz, bw, label=LABELS.get(v, v), color=col, marker='o', linewidth=2.2)

    ax1.axhline(168.0, color='red', linestyle=':', linewidth=1.5, alpha=0.7, label='GPU Theoretical Bus (168 GB/s)')
    ax1.set_xscale('log')
    ax1.set_xlabel('Vector Length N (doubles)')
    ax1.set_ylabel('Effective Memory Bandwidth (GB/s)')
    ax1.set_title('Dot Product (x^T y): Bandwidth Scaling vs N')
    ax1.grid(True, which="both", ls="--", alpha=0.3)
    ax1.legend(loc='upper left', framealpha=0.92)

    # Right: Operation Comparison at Maximum Size (10,000,000 doubles)
    max_size = max(sizes) if sizes else 10000000
    ops = ['dot', 'norm_2', 'norm_inf']
    op_labels = ['Dot Product\n(x^T y)', 'Euclidean Norm\n(||x||_2)', 'Infinity Norm\n(||x||_inf)']
    
    x_pos = np.arange(len(ops))
    bar_width = 0.25

    for idx, v in enumerate(['CPU_1T', 'CPU_12T', 'CUDA']):
        bws = []
        for op in ops:
            matching = [r for r in reduction_records if r['operation'] == op and r['size'] == max_size and r['variant'] == v]
            bws.append(matching[0]['bandwidth_gbs'] if matching else 0.0)
        
        offset = (idx - 1) * bar_width
        ax2.bar(x_pos + offset, bws, width=bar_width, label=LABELS.get(v, v), color=COLORS.get(v, '#2b6cb0'))

    ax2.axhline(168.0, color='red', linestyle=':', linewidth=1.5, alpha=0.7, label='GPU Theoretical Bus (168 GB/s)')
    ax2.set_ylabel('Effective Memory Bandwidth (GB/s)')
    ax2.set_title(f'Vector Reductions at N = {max_size:,} (80 MB Working Set)')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(op_labels)
    ax2.grid(True, axis='y', ls="--", alpha=0.3)
    ax2.legend(loc='upper right', framealpha=0.92)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"  [Saved] {output_path}")


def main():
    results_dir = "benchmarks/results"
    plots_dir = "benchmarks/plots"
    docs_plots_dir = "docs/plots"

    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
    if len(sys.argv) > 2:
        plots_dir = sys.argv[2]

    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(docs_plots_dir, exist_ok=True)

    print(f"================================================================")
    print(f" Generating Benchmark Research Figures")
    print(f" Input Results: {results_dir}")
    print(f" Output Plots:  {plots_dir} & {docs_plots_dir}")
    print(f"================================================================")

    matrix_data, red_data = load_data(results_dir)
    matrix_records = matrix_data.get('experiments', [])

    print(f"Loaded {len(matrix_records)} matrix records and {len(red_data)} reduction records.")

    plots = [
        ("runtime_vs_nnz.png", lambda p: plot_runtime_vs_nnz(matrix_records, p)),
        ("speedup_vs_problem_size.png", lambda p: plot_speedup_vs_problem_size(matrix_records, p)),
        ("performance_vs_sparsity.png", lambda p: plot_performance_vs_sparsity(matrix_records, p)),
        ("structure_comparison.png", lambda p: plot_structure_comparison(matrix_records, p)),
        ("reductions_bandwidth.png", lambda p: plot_reductions_bandwidth(red_data, p))
    ]

    for filename, plot_fn in plots:
        out1 = os.path.join(plots_dir, filename)
        plot_fn(out1)
        import shutil
        out2 = os.path.join(docs_plots_dir, filename)
        shutil.copyfile(out1, out2)

    print("All research figures successfully generated!\n")

if __name__ == '__main__':
    main()
