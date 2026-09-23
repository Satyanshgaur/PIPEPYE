import os
import json
import glob
import time
import numpy as np
import highspy

def solve_and_verify(mps_path):
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    status = h.readModel(mps_path)
    if status != highspy.HighsStatus.kOk:
        raise RuntimeError(f"Failed to read {mps_path}")
    
    lp = h.getLp()
    num_cols = lp.num_col_
    num_rows = lp.num_row_
    
    t0 = time.perf_counter()
    h.run()
    elapsed = time.perf_counter() - t0
    
    model_status = h.getModelStatus()
    info = h.getInfo()
    sol = h.getSolution()
    
    obj_val = info.objective_function_value
    simplex_iters = info.simplex_iteration_count
    ipm_iters = info.ipm_iteration_count
    mip_nodes = info.mip_node_count
    
    # Primal solution
    col_value = np.array(sol.col_value)
    row_value = np.array(sol.row_value)
    
    # Independent verification
    col_lower = np.array(lp.col_lower_)
    col_upper = np.array(lp.col_upper_)
    row_lower = np.array(lp.row_lower_)
    row_upper = np.array(lp.row_upper_)
    
    # Check variable bound violations
    lower_viol = np.maximum(0.0, col_lower - col_value)
    upper_viol = np.maximum(0.0, col_value - col_upper)
    max_var_viol = float(np.max(np.maximum(lower_viol, upper_viol))) if len(col_value) > 0 else 0.0
    
    # Check row bound violations
    row_lower_viol = np.maximum(0.0, row_lower - row_value)
    row_upper_viol = np.maximum(0.0, row_value - row_upper)
    max_row_viol = float(np.max(np.maximum(row_lower_viol, row_upper_viol))) if len(row_value) > 0 else 0.0
    
    # Check integrality
    integrality_viol = 0.0
    num_integrals = 0
    if len(lp.integrality_) > 0:
        for j, int_type in enumerate(lp.integrality_):
            if int_type != highspy.HighsVarType.kContinuous:
                num_integrals += 1
                val = col_value[j]
                integrality_viol = max(integrality_viol, abs(val - round(val)))
                
    # Recompute objective directly: c^T x + offset
    col_cost = np.array(lp.col_cost_)
    recomputed_obj = float(np.dot(col_cost, col_value) + lp.offset_)
    obj_diff = abs(recomputed_obj - obj_val)
    
    status_str = h.modelStatusToString(model_status)
    
    verification_passed = (
        max_var_viol < 1e-4 and
        max_row_viol < 1e-4 and
        integrality_viol < 1e-4 and
        obj_diff < 1e-3 and
        status_str in ["Optimal", "Model status not set"]
    )
    
    return {
        "mps_file": os.path.basename(mps_path),
        "solver": "HiGHS-1.8.1",
        "model_status": status_str,
        "objective": obj_val,
        "recomputed_objective": recomputed_obj,
        "objective_discrepancy": obj_diff,
        "independent_verification": {
            "passed": bool(verification_passed),
            "max_variable_bound_violation": max_var_viol,
            "max_row_bound_violation": max_row_viol,
            "max_integrality_violation": float(integrality_viol),
            "num_integer_variables": num_integrals
        },
        "performance": {
            "simplex_iterations": simplex_iters,
            "ipm_iterations": ipm_iters,
            "mip_nodes": mip_nodes,
            "wallclock_time_sec": elapsed,
            "highs_run_time_sec": h.getRunTime()
        }
    }

def main():
    cases = [
        "case_a_crude_blending",
        "case_b_multi_period_planning",
        "case_c_refinery_scheduling",
        "case_d_unit_commitment"
    ]
    
    all_reference = {}
    
    for case in cases:
        case_dir = os.path.join("workloads", case)
        ref_dir = os.path.join(case_dir, "reference")
        os.makedirs(ref_dir, exist_ok=True)
        
        mps_files = sorted(glob.glob(os.path.join(case_dir, "*.mps")))
        case_refs = {}
        for mps in mps_files:
            bname = os.path.basename(mps)
            if bname == "model.mps":
                continue
            inst_name = os.path.splitext(bname)[0]
            print(f"Solving & verifying {case} / {bname} with HiGHS...")
            ref_data = solve_and_verify(mps)
            case_refs[inst_name] = ref_data
            print(f"  -> Status: {ref_data['model_status']}, Obj: {ref_data['objective']:.6e}, VarViol: {ref_data['independent_verification']['max_variable_bound_violation']:.2e}, RowViol: {ref_data['independent_verification']['max_row_bound_violation']:.2e}, Passed: {ref_data['independent_verification']['passed']}")
            
        all_reference[case] = case_refs
        
        # Write case-specific reference solution JSON
        ref_file = os.path.join(ref_dir, "solution.json")
        with open(ref_file, "w") as f:
            json.dump(case_refs, f, indent=2)
        print(f"Saved {ref_file}\n")
        
    # Write global consolidated reference solutions
    os.makedirs("reports", exist_ok=True)
    with open("reports/reference_solutions.json", "w") as f:
        json.dump(all_reference, f, indent=2)
    print("Saved reports/reference_solutions.json successfully!")

if __name__ == "__main__":
    main()
