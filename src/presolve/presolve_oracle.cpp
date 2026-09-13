#include <pipepye/presolve/presolve_oracle.hpp>
#include <cmath>
#include <random>
#include <sstream>
#include <algorithm>
#include <iostream>

namespace pipepye::presolve {

std::string OracleVerificationResult::format_report() const {
    std::ostringstream ss;
    ss << "=== PRESOLVE ORACLE VERIFICATION REPORT ===\n";
    ss << "  Passed: " << (passed ? "YES" : "NO") << "\n";
    ss << "  Samples Checked: " << samples_checked << "\n";
    ss << "  Feasible Samples Found: " << feasible_samples_found << "\n";
    ss << "  Bound Violations: " << bound_violations << " (max violation: " << max_bound_violation << ")\n";
    ss << "  Constraint Violations: " << constraint_violations << " (max violation: " << max_constraint_violation << ")\n";
    ss << "  Objective Mismatches: " << objective_mismatches << " (max error: " << max_objective_error << ")\n";
    if (!error_messages.empty()) {
        ss << "  Errors encountered:\n";
        for (const auto& err : error_messages) {
            ss << "    - " << err << "\n";
        }
    }
    ss << "===========================================\n";
    return ss.str();
}

static bool is_feasible_in_lp(const model::LinearProgram& lp, const std::vector<scalar_t>& x,
                             scalar_t tol, scalar_t& out_max_bound_viol, scalar_t& out_max_con_viol) {
    out_max_bound_viol = 0.0;
    out_max_con_viol = 0.0;
    const scalar_t inf_thresh = 1e19;

    // 1. Variable bounds
    for (size_t j = 0; j < x.size(); ++j) {
        scalar_t val = x[j];
        scalar_t lb = lp.col_lower[j];
        scalar_t ub = lp.col_upper[j];
        if (lb > -inf_thresh && val < lb - tol) {
            out_max_bound_viol = std::max(out_max_bound_viol, lb - val);
            return false;
        }
        if (ub < inf_thresh && val > ub + tol) {
            out_max_bound_viol = std::max(out_max_bound_viol, val - ub);
            return false;
        }
    }

    // 2. Constraints: Ax
    index_t m = lp.num_rows();
    if (!lp.csr_row_ptr.empty()) {
        for (index_t i = 0; i < m; ++i) {
            scalar_t ax = 0.0;
            index_t start = lp.csr_row_ptr[i];
            index_t end = lp.csr_row_ptr[i + 1];
            for (index_t p = start; p < end; ++p) {
                ax += lp.csr_values[p] * x[lp.csr_col_ind[p]];
            }
            scalar_t lb = lp.row_lower[i];
            scalar_t ub = lp.row_upper[i];
            if (lb > -inf_thresh && ax < lb - tol) {
                out_max_con_viol = std::max(out_max_con_viol, lb - ax);
                return false;
            }
            if (ub < inf_thresh && ax > ub + tol) {
                out_max_con_viol = std::max(out_max_con_viol, ax - ub);
                return false;
            }
        }
    }
    return true;
}

OracleVerificationResult PresolveOracle::verify_feasibility_and_objective(
    const model::LinearProgram& original_lp,
    const PresolvedModel& presolved) {

    OracleVerificationResult res;

    // Case 1: OptimalSolved (all variables fixed/eliminated)
    if (presolved.status == PresolveStatus::OptimalSolved) {
        PrimalDualSolution pre_sol;
        auto post_res = presolved.postsolve_mgr.postsolve(original_lp, pre_sol);
        if (!post_res.is_ok()) {
            res.passed = false;
            res.error_messages.push_back("Postsolve failed on OptimalSolved model: " + post_res.status().message());
            return res;
        }

        res.samples_checked = 1;
        const auto& sol = post_res.value();
        scalar_t b_viol = 0.0, c_viol = 0.0;
        if (!is_feasible_in_lp(original_lp, sol.x, options_.tolerance, b_viol, c_viol)) {
            res.passed = false;
            res.bound_violations += (b_viol > options_.tolerance ? 1 : 0);
            res.constraint_violations += (c_viol > options_.tolerance ? 1 : 0);
            res.max_bound_violation = b_viol;
            res.max_constraint_violation = c_viol;
            res.error_messages.push_back("OptimalSolved reconstructed point violates original bounds/constraints.");
            return res;
        }

        res.feasible_samples_found = 1;

        // Check objective
        scalar_t manual_obj = original_lp.obj_offset;
        for (index_t j = 0; j < original_lp.num_cols(); ++j) {
            manual_obj += original_lp.c[j] * sol.x[j];
        }

        scalar_t diff = std::abs(manual_obj - presolved.lp.obj_offset);
        if (diff > options_.tolerance) {
            res.passed = false;
            ++res.objective_mismatches;
            res.max_objective_error = diff;
            res.error_messages.push_back("Objective mismatch in OptimalSolved: original " +
                                         std::to_string(manual_obj) + " vs presolved offset " +
                                         std::to_string(presolved.lp.obj_offset));
        }
        return res;
    }

    if (presolved.status != PresolveStatus::Reduced && presolved.status != PresolveStatus::Unchanged) {
        // Infeasible or Unbounded: verify separately
        return verify_infeasibility_consistency(original_lp, presolved);
    }

    const auto& pre_lp = presolved.lp;
    index_t pre_n = pre_lp.num_cols();
    if (pre_n == 0) {
        return res;
    }

    std::mt19937 rng(options_.seed);
    std::uniform_real_distribution<scalar_t> uniform_dist(0.0, 1.0);

    std::vector<std::vector<scalar_t>> candidate_samples;

    // 1. Midpoint candidate
    std::vector<scalar_t> mid_point(pre_n, 0.0);
    for (index_t j = 0; j < pre_n; ++j) {
        scalar_t lb = (pre_lp.col_lower[j] > -1e10) ? pre_lp.col_lower[j] : -10.0;
        scalar_t ub = (pre_lp.col_upper[j] < 1e10) ? pre_lp.col_upper[j] : 10.0;
        mid_point[j] = 0.5 * (lb + ub);
    }
    candidate_samples.push_back(mid_point);

    // 2. Bound boundary candidates
    std::vector<scalar_t> lower_bound_pt(pre_n, 0.0);
    for (index_t j = 0; j < pre_n; ++j) {
        lower_bound_pt[j] = (pre_lp.col_lower[j] > -1e10) ? pre_lp.col_lower[j] : 0.0;
    }
    candidate_samples.push_back(lower_bound_pt);

    // 3. Random bounded samples
    for (int s = 0; s < options_.random_samples; ++s) {
        std::vector<scalar_t> rand_pt(pre_n, 0.0);
        for (index_t j = 0; j < pre_n; ++j) {
            scalar_t lb = (pre_lp.col_lower[j] > -1e10) ? pre_lp.col_lower[j] : -10.0;
            scalar_t ub = (pre_lp.col_upper[j] < 1e10) ? pre_lp.col_upper[j] : 10.0;
            if (lb > ub) std::swap(lb, ub);
            rand_pt[j] = lb + uniform_dist(rng) * (ub - lb);
        }
        candidate_samples.push_back(rand_pt);
    }

    for (const auto& sample : candidate_samples) {
        ++res.samples_checked;
        scalar_t b_viol = 0.0, c_viol = 0.0;

        // Check feasibility in presolved LP
        if (!is_feasible_in_lp(pre_lp, sample, options_.tolerance, b_viol, c_viol)) {
            continue; // Not feasible in presolved LP, skip mapping
        }

        ++res.feasible_samples_found;

        // Construct presolved solution
        PrimalDualSolution pre_sol;
        pre_sol.x = sample;
        pre_sol.y.assign(pre_lp.num_rows(), 0.0);
        pre_sol.s.assign(pre_lp.num_cols(), 0.0);

        auto post_res = presolved.postsolve_mgr.postsolve(original_lp, pre_sol);
        if (!post_res.is_ok()) {
            res.passed = false;
            res.error_messages.push_back("Postsolve failed: " + post_res.status().message());
            continue;
        }

        const auto& orig_sol = post_res.value();
        scalar_t orig_b_viol = 0.0, orig_c_viol = 0.0;
        if (!is_feasible_in_lp(original_lp, orig_sol.x, options_.tolerance, orig_b_viol, orig_c_viol)) {
            res.passed = false;
            if (orig_b_viol > options_.tolerance) ++res.bound_violations;
            if (orig_c_viol > options_.tolerance) ++res.constraint_violations;
            res.max_bound_violation = std::max(res.max_bound_violation, orig_b_viol);
            res.max_constraint_violation = std::max(res.max_constraint_violation, orig_c_viol);
            res.error_messages.push_back("Presolved feasible point became infeasible after postsolve.");
        }

        // Compare objective values
        scalar_t orig_obj = original_lp.obj_offset;
        for (index_t j = 0; j < original_lp.num_cols(); ++j) {
            orig_obj += original_lp.c[j] * orig_sol.x[j];
        }

        scalar_t pre_obj = pre_lp.obj_offset;
        for (index_t j = 0; j < pre_lp.num_cols(); ++j) {
            pre_obj += pre_lp.c[j] * sample[j];
        }

        scalar_t obj_diff = std::abs(orig_obj - pre_obj);
        res.max_objective_error = std::max(res.max_objective_error, obj_diff);
        if (obj_diff > options_.tolerance) {
            res.passed = false;
            ++res.objective_mismatches;
            res.error_messages.push_back("Objective discrepancy: orig=" + std::to_string(orig_obj) +
                                         ", presolved=" + std::to_string(pre_obj));
        }
    }

    return res;
}

OracleVerificationResult PresolveOracle::verify_small_lp_optimality(
    const model::LinearProgram& original_lp,
    const PresolvedModel& presolved) {

    OracleVerificationResult res;
    if (original_lp.num_cols() > 5 || presolved.lp.num_cols() > 5) {
        // Model dimension too large for grid oracle
        return verify_feasibility_and_objective(original_lp, presolved);
    }

    // Grid search on original LP
    index_t orig_n = original_lp.num_cols();
    std::vector<std::vector<scalar_t>> dim_values(orig_n);
    for (index_t j = 0; j < orig_n; ++j) {
        scalar_t lb = (original_lp.col_lower[j] > -1e10) ? original_lp.col_lower[j] : -10.0;
        scalar_t ub = (original_lp.col_upper[j] < 1e10) ? original_lp.col_upper[j] : 10.0;
        int pts = options_.grid_points_per_dim;
        for (int p = 0; p < pts; ++p) {
            scalar_t t = static_cast<scalar_t>(p) / (pts - 1);
            dim_values[j].push_back(lb + t * (ub - lb));
        }
    }

    scalar_t best_orig_obj = 1e30;
    bool found_feasible_orig = false;

    // Recursive Cartesian product
    std::vector<scalar_t> current_pt(orig_n, 0.0);
    auto search_original = [&](auto self, index_t dim) -> void {
        if (dim == orig_n) {
            scalar_t b_v = 0.0, c_v = 0.0;
            if (is_feasible_in_lp(original_lp, current_pt, options_.tolerance, b_v, c_v)) {
                found_feasible_orig = true;
                scalar_t obj = original_lp.obj_offset;
                for (index_t j = 0; j < orig_n; ++j) obj += original_lp.c[j] * current_pt[j];
                best_orig_obj = std::min(best_orig_obj, obj);
            }
            return;
        }
        for (scalar_t val : dim_values[dim]) {
            current_pt[dim] = val;
            self(self, dim + 1);
        }
    };

    search_original(search_original, 0);

    if (presolved.status == PresolveStatus::Infeasible) {
        if (found_feasible_orig) {
            res.passed = false;
            res.error_messages.push_back("Oracle found feasible point but presolve declared model Infeasible.");
        }
        return res;
    }

    return verify_feasibility_and_objective(original_lp, presolved);
}

OracleVerificationResult PresolveOracle::verify_infeasibility_consistency(
    const model::LinearProgram& original_lp,
    const PresolvedModel& presolved) {

    OracleVerificationResult res;
    if (presolved.status != PresolveStatus::Infeasible) {
        return res;
    }

    // Infeasible: check if any sampled point in original bounds is feasible
    std::mt19937 rng(options_.seed);
    std::uniform_real_distribution<scalar_t> uniform_dist(0.0, 1.0);
    index_t n = original_lp.num_cols();

    for (int s = 0; s < options_.random_samples; ++s) {
        ++res.samples_checked;
        std::vector<scalar_t> pt(n, 0.0);
        for (index_t j = 0; j < n; ++j) {
            scalar_t lb = (original_lp.col_lower[j] > -1e10) ? original_lp.col_lower[j] : -10.0;
            scalar_t ub = (original_lp.col_upper[j] < 1e10) ? original_lp.col_upper[j] : 10.0;
            pt[j] = lb + uniform_dist(rng) * (ub - lb);
        }
        scalar_t b_v = 0.0, c_v = 0.0;
        if (is_feasible_in_lp(original_lp, pt, options_.tolerance, b_v, c_v)) {
            res.passed = false;
            res.error_messages.push_back("Oracle found a feasible point in a model flagged as Infeasible by presolve!");
            return res;
        }
    }

    res.passed = true;
    return res;
}

} // namespace pipepye::presolve
