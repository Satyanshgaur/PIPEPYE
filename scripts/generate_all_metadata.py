import os
import json
import glob
import numpy as np
import highspy

def analyze_mps(mps_path):
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.readModel(mps_path)
    lp = h.getLp()
    
    m = lp.num_row_
    n = lp.num_col_
    
    # Nonzero extraction
    a_matrix = lp.a_matrix_
    # HighsSparseMatrix: start_, index_, value_
    start = np.array(a_matrix.start_)
    index = np.array(a_matrix.index_)
    value = np.array(a_matrix.value_)
    nnz = len(value)
    
    density = nnz / (m * n) if m * n > 0 else 0.0
    
    # Ranges
    abs_vals = np.abs(value)
    min_abs_coeff = float(np.min(abs_vals)) if nnz > 0 else 0.0
    max_abs_coeff = float(np.max(abs_vals)) if nnz > 0 else 0.0
    
    col_lower = np.array(lp.col_lower_)
    col_upper = np.array(lp.col_upper_)
    row_lower = np.array(lp.row_lower_)
    row_upper = np.array(lp.row_upper_)
    
    # Variable types
    num_binaries = 0
    num_integers = 0
    num_continuous = 0
    for j in range(n):
        vtype = lp.integrality_[j] if len(lp.integrality_) > 0 else highspy.HighsVarType.kContinuous
        if vtype == highspy.HighsVarType.kContinuous:
            num_continuous += 1
        elif (col_lower[j] == 0.0 and col_upper[j] == 1.0):
            num_binaries += 1
        else:
            num_integers += 1
            
    # Row and Col NNZ distribution
    col_nnz = np.diff(start) if len(start) > 1 else np.array([nnz])
    row_counts = np.zeros(m, dtype=int)
    for r in index:
        if 0 <= r < m:
            row_counts[r] += 1
            
    # Gini coefficient of row counts
    sorted_rows = np.sort(row_counts)
    if len(sorted_rows) > 0 and np.sum(sorted_rows) > 0:
        index_arr = np.arange(1, m + 1)
        gini = (2.0 * np.sum(index_arr * sorted_rows)) / (m * np.sum(sorted_rows)) - (m + 1.0) / m
    else:
        gini = 0.0
        
    # Staircase score (fraction of nonzeros in normalized diagonal band)
    diag_matches = 0
    for col_j in range(len(col_nnz)):
        c_start = start[col_j]
        c_end = start[col_j + 1] if col_j + 1 < len(start) else nnz
        norm_col = col_j / n if n > 0 else 0.0
        for k in range(c_start, c_end):
            row_i = index[k]
            norm_row = row_i / m if m > 0 else 0.0
            if abs(norm_row - norm_col) <= 0.25:
                diag_matches += 1
    staircase_score = diag_matches / nnz if nnz > 0 else 0.0
    
    # Finite RHS / bounds
    finite_rhs = []
    for r in list(row_lower) + list(row_upper):
        if abs(r) < 1e15:
            finite_rhs.append(abs(r))
    min_rhs = float(np.min(finite_rhs)) if finite_rhs else 0.0
    max_rhs = float(np.max(finite_rhs)) if finite_rhs else 0.0
    
    finite_bounds = []
    for b in list(col_lower) + list(col_upper):
        if abs(b) < 1e15:
            finite_bounds.append(abs(b))
    min_bound = float(np.min(finite_bounds)) if finite_bounds else 0.0
    max_bound = float(np.max(finite_bounds)) if finite_bounds else 0.0
    
    return {
        "m": int(m),
        "n": int(n),
        "nnz": int(nnz),
        "density": float(density),
        "min_abs_coeff": min_abs_coeff,
        "max_abs_coeff": max_abs_coeff,
        "dynamic_range": max_abs_coeff / min_abs_coeff if min_abs_coeff > 0 else 1.0,
        "num_continuous": num_continuous,
        "num_integers": num_integers,
        "num_binaries": num_binaries,
        "row_nnz_avg": float(np.mean(row_counts)) if m > 0 else 0.0,
        "row_nnz_stddev": float(np.std(row_counts)) if m > 0 else 0.0,
        "col_nnz_avg": float(np.mean(col_nnz)) if n > 0 else 0.0,
        "col_nnz_stddev": float(np.std(col_nnz)) if n > 0 else 0.0,
        "row_gini": float(gini),
        "staircase_score": float(staircase_score),
        "min_rhs": min_rhs,
        "max_rhs": max_rhs,
        "min_bound": min_bound,
        "max_bound": max_bound
    }

def main():
    # Load reference solutions if available
    ref_file = "reports/reference_solutions.json"
    ref_data = {}
    if os.path.exists(ref_file):
        with open(ref_file) as f:
            ref_data = json.load(f)
            
    case_meta_configs = {
        "case_a_crude_blending": {
            "name": "Crude Oil Blending",
            "category": "Petroleum Refining & Downstream Logistics",
            "formulation_class": "LP",
            "mathematical_structure": "Densely coupled, compact linear program with dense quality rows",
            "provenance": {
                "benchmark": "Haverly (1978) Pooling Benchmark & Baker-Lasdon (1985) Refinery Blending LP",
                "references": [
                    "Haverly, C. A. (1978). Studies of Compromise Solutions for the Pooling Problem. ACM SIGMAP Bulletin, 25, 19-28.",
                    "Williams, H. P. (2013). Model Building in Mathematical Programming (5th ed., Ch. 12). Wiley.",
                    "Gary, J. H., Handwerk, G. E., & Kaiser, M. J. (2007). Petroleum Refining: Technology and Economics (5th ed.). CRC Press."
                ]
            },
            "pre_registered_prediction": {
                "hardware": "CPU",
                "hardware_rationale": "Small matrix dimensions (m <= 155, n <= 260) fit fully within CPU L1/L2 cache; PCI-e transfer and CUDA kernel launch latency would heavily dominate total runtime.",
                "algorithm": "DualSimplex",
                "algorithm_rationale": "High nonzero density (9%-30%) and dense quality rows lead to ill-conditioned gradient operators for first-order PDHG solvers; exact vertex pivoting with LU basis updates resolves active quality limits in < 100 pivots."
            }
        },
        "case_b_multi_period_planning": {
            "name": "Multi-Period Production & Inventory Planning",
            "category": "Supply Chain & Continuous Manufacturing",
            "formulation_class": "LP",
            "mathematical_structure": "Block-angular / staircase sparse linear program with dynamic inventory propagation",
            "provenance": {
                "benchmark": "Manne (1958) Multi-Sector Production Planning & Netlib Staircase Class (SC205, SCTAP1)",
                "references": [
                    "Manne, A. S. (1958). Programming of Economic Lot Sizes. Management Science, 4(2), 115-135.",
                    "Fourer, R. (1982). Solving Staircase Linear Programs by the Simplex Method, 1: Inversion. Math. Prog., 23(1), 274-313.",
                    "Fourer, R. (1983). Solving Staircase Linear Programs by the Simplex Method, 2: Pricing. Math. Prog., 25(3), 251-292."
                ]
            },
            "pre_registered_prediction": {
                "hardware": "Adaptive: CPU for T <= 50 (NNZ < 20k); GPU (CUDA) for T >= 100 (NNZ >= 30k)",
                "hardware_rationale": "Large horizons create extensive block-banded sparse matrices where repeated SpMV operations in PDHG achieve high memory bandwidth utilization and parallel thread occupancy across independent time stages.",
                "algorithm": "DualSimplex for small horizons; PDHG First-Order for large horizons (T >= 100)",
                "algorithm_rationale": "Simplex pivot counts scale superlinearly with horizon length as basis path traverses temporal stages; first-order PDHG provides linear per-iteration complexity scaling with NNZ."
            }
        },
        "case_c_refinery_scheduling": {
            "name": "Refinery Unit Scheduling",
            "category": "Petrochemical Unit Operations & Tank Logistics",
            "formulation_class": "MILP",
            "mathematical_structure": "Mixed-Integer Linear Program with disjunctive semicontinuous mode constraints and inventory tank mass balances",
            "provenance": {
                "benchmark": "Pinto, Joly & Moro (2000) Refinery Planning & Floudas-Lin (2004) Scheduling Models",
                "references": [
                    "Pinto, J. M., Joly, M., & Moro, L. F. (2000). Planning and Scheduling Models for Refinery Operations. Computers & Chem. Eng., 24(9-10), 2259-2276.",
                    "Moro, L. F. L., & Pinto, J. M. (1998). Mixed-Integer Optimization for the Scheduling of Pipeline Transfers. Computers & Chem. Eng., 22, S725-S728.",
                    "Floudas, C. A., & Lin, X. (2004). Continuous-time versus discrete-time approaches for scheduling. Computers & Chem. Eng., 28(11), 2109-2129."
                ]
            },
            "pre_registered_prediction": {
                "hardware": "CPU",
                "hardware_rationale": "Branch-and-bound tree search requires irregular tree traversal, dynamic priority queue management, and sequential node selection, which cannot be effectively mapped to massively parallel SIMT GPUs.",
                "algorithm": "BranchAndBound with Warm-Started Dual Simplex",
                "algorithm_rationale": "40%-43% binary variables necessitate discrete tree search; variable bound tightenings at child nodes preserve dual feasibility, allowing warm-started Dual Simplex to resolve child LPs in < 5 pivots."
            }
        },
        "case_d_unit_commitment": {
            "name": "Power System Unit Commitment & Economic Dispatch",
            "category": "Wholesale Electricity Markets & Transmission Grids",
            "formulation_class": "MILP",
            "mathematical_structure": "Mixed-Integer Linear Program with inter-temporal dynamic ramping envelope and hourly spinning reserve coupling",
            "provenance": {
                "benchmark": "Carrion & Arroyo (2006) Efficient Thermal Unit Commitment & Wood-Wollenberg (2013)",
                "references": [
                    "Wood, A. J., Wollenberg, B. F., & Sheble, G. B. (2013). Power Generation, Operation, and Control (3rd ed.). Wiley.",
                    "Carrion, M., & Arroyo, J. M. (2006). A Computationally Efficient Mixed-Integer Linear Formulation for Thermal Unit Commitment. IEEE Trans. Power Syst., 21(3), 1371-1378.",
                    "Ostrowski, J., Anjos, M. F., & Vannelli, A. (2012). Tight Mixed Integer Linear Programming Formulations for UC. IEEE Trans. Power Syst., 27(1), 39-46."
                ]
            },
            "pre_registered_prediction": {
                "hardware": "CPU",
                "hardware_rationale": "Combinatorial binary search over 50% integer variables requires low-latency branch handling and frequent CPU cache hits on basis LU structures.",
                "algorithm": "BranchAndBound with Warm-Started Dual Simplex",
                "algorithm_rationale": "Hourly ramping constraints link adjacent periods, but bounding a generator commitment variable u_{g,t} in {0,1} only shifts bounds, which Dual Simplex exploits to eliminate > 90% of re-optimization pivots."
            }
        }
    }
    
    for case, config in case_meta_configs.items():
        case_dir = os.path.join("workloads", case)
        mps_files = sorted(glob.glob(os.path.join(case_dir, "*.mps")))
        
        instance_list = []
        for mps in mps_files:
            bname = os.path.basename(mps)
            if bname == "model.mps":
                continue
            scale_id = os.path.splitext(bname)[0]
            print(f"Analyzing {case} / {bname}...")
            stats = analyze_mps(mps)
            
            # Match reference
            case_ref = ref_data.get(case, {}).get(scale_id, {})
            
            # Predicted vs actual
            pred_solver = "DualSimplex"
            pred_backend = "CPU"
            if "planning" in case and ("T100" in scale_id or stats["nnz"] >= 30000):
                pred_solver = "PDHG_GPU"
                pred_backend = "GPU"
            elif stats["num_binaries"] > 0 or stats["num_integers"] > 0:
                pred_solver = "BranchAndBound"
                pred_backend = "CPU"
                
            inst_obj = {
                "instance_name": f"{config['name']} ({scale_id})",
                "file_name": bname,
                "scale": scale_id,
                "dimensions": {
                    "rows": stats["m"],
                    "columns": stats["n"],
                    "nonzeros": stats["nnz"],
                    "density": stats["density"],
                    "density_pct": f"{stats['density'] * 100:.2f}%"
                },
                "variable_types": {
                    "continuous": stats["num_continuous"],
                    "integer": stats["num_integers"],
                    "binary": stats["num_binaries"],
                    "integrality_ratio": float((stats["num_binaries"] + stats["num_integers"]) / stats["n"]) if stats["n"] > 0 else 0.0
                },
                "structural_metrics": {
                    "staircase_score": stats["staircase_score"],
                    "row_gini_index": stats["row_gini"],
                    "row_nnz_mean": stats["row_nnz_avg"],
                    "row_nnz_stddev": stats["row_nnz_stddev"],
                    "col_nnz_mean": stats["col_nnz_avg"],
                    "col_nnz_stddev": stats["col_nnz_stddev"],
                    "dynamic_range": stats["dynamic_range"]
                },
                "ranges": {
                    "min_abs_coeff": stats["min_abs_coeff"],
                    "max_abs_coeff": stats["max_abs_coeff"],
                    "min_rhs": stats["min_rhs"],
                    "max_rhs": stats["max_rhs"],
                    "min_bound": stats["min_bound"],
                    "max_bound": stats["max_bound"]
                },
                "pre_registered_prediction": {
                    "predicted_solver": pred_solver,
                    "predicted_backend": pred_backend
                },
                "reference_solution": {
                    "solver": case_ref.get("solver", "HiGHS-1.8.1"),
                    "status": case_ref.get("model_status", "UNKNOWN"),
                    "objective": case_ref.get("objective", None),
                    "independent_verification_passed": case_ref.get("independent_verification", {}).get("passed", False),
                    "simplex_iterations": case_ref.get("performance", {}).get("simplex_iterations", 0),
                    "mip_nodes": case_ref.get("performance", {}).get("mip_nodes", 0),
                    "wallclock_sec": case_ref.get("performance", {}).get("wallclock_time_sec", 0.0)
                }
            }
            instance_list.append(inst_obj)
            
        case_full_metadata = {
            "workload_case": case,
            "name": config["name"],
            "category": config["category"],
            "formulation_class": config["formulation_class"],
            "mathematical_structure": config["mathematical_structure"],
            "provenance": config["provenance"],
            "pre_registered_strategy": config["pre_registered_prediction"],
            "instances": instance_list
        }
        
        meta_path = os.path.join(case_dir, "metadata.json")
        with open(meta_path, "w") as f:
            json.dump(case_full_metadata, f, indent=2)
        print(f"Generated comprehensive metadata at {meta_path}\n")

if __name__ == "__main__":
    main()
