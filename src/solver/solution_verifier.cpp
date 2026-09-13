#include <pipepye/solver/solution_verifier.hpp>
#include <cmath>
#include <algorithm>

namespace pipepye::solver {

VerificationResult SolutionVerifier::verify(
    const model::LinearProgram& lp,
    const std::vector<scalar_t>& x,
    const std::vector<scalar_t>& y,
    scalar_t claimed_objective,
    scalar_t tolerance) {

    VerificationResult res;
    index_t m = lp.num_rows();
    index_t n = lp.num_cols();

    if (static_cast<index_t>(x.size()) != n) {
        res.is_feasible = false;
        res.is_objective_consistent = false;
        res.details = "Dimension mismatch: x size (" + std::to_string(x.size()) +
                      ") does not match LP cols (" + std::to_string(n) + ")";
        return res;
    }

    const scalar_t kInf = 1e15;

    // 1. Check variable bounds: l_c <= x <= u_c
    scalar_t max_b_viol = 0.0;
    index_t worst_b_idx = -1;
    for (index_t j = 0; j < n; ++j) {
        scalar_t xj = x[j];
        scalar_t lb = lp.col_lower[j];
        scalar_t ub = lp.col_upper[j];
        scalar_t v = 0.0;
        if (lb > -kInf && xj < lb) {
            v = std::max(v, lb - xj);
        }
        if (ub < kInf && xj > ub) {
            v = std::max(v, xj - ub);
        }
        if (v > max_b_viol) {
            max_b_viol = v;
            worst_b_idx = j;
        }
    }
    res.max_bound_violation = max_b_viol;
    res.worst_bound_var = worst_b_idx;

    // 2. Check row constraints: l_r <= A x <= u_r
    scalar_t max_c_viol = 0.0;
    index_t worst_c_idx = -1;

    for (index_t i = 0; i < m; ++i) {
        scalar_t act = 0.0;
        index_t start = lp.csr_row_ptr[i];
        index_t end = lp.csr_row_ptr[i + 1];
        for (index_t p = start; p < end; ++p) {
            act += lp.csr_values[p] * x[lp.csr_col_ind[p]];
        }

        scalar_t lb = lp.row_lower[i];
        scalar_t ub = lp.row_upper[i];
        scalar_t v = 0.0;
        if (lb > -kInf && act < lb) {
            v = std::max(v, lb - act);
        }
        if (ub < kInf && act > ub) {
            v = std::max(v, act - ub);
        }
        if (v > max_c_viol) {
            max_c_viol = v;
            worst_c_idx = i;
        }
    }
    res.max_constraint_violation = max_c_viol;
    res.worst_constraint_row = worst_c_idx;

    // 3. Recompute objective: c^T x + c_0
    scalar_t obj = lp.obj_offset;
    for (index_t j = 0; j < n; ++j) {
        obj += lp.c[j] * x[j];
    }
    res.recomputed_objective = obj;
    res.objective_mismatch = std::abs(obj - claimed_objective);

    // 4. Check dual stationarity if y is provided
    scalar_t max_d_viol = 0.0;
    if (!y.empty() && static_cast<index_t>(y.size()) == m) {
        std::vector<scalar_t> Aty(n, 0.0);
        // Transposed SpMV: A^T y using CSC or CSR
        if (!lp.csc_col_ptr.empty() && static_cast<index_t>(lp.csc_col_ptr.size()) == n + 1) {
            for (index_t j = 0; j < n; ++j) {
                index_t start = lp.csc_col_ptr[j];
                index_t end = lp.csc_col_ptr[j + 1];
                scalar_t sum = 0.0;
                for (index_t p = start; p < end; ++p) {
                    sum += lp.csc_values[p] * y[lp.csc_row_ind[p]];
                }
                Aty[j] = sum;
            }
        } else {
            // Fallback via CSR
            for (index_t i = 0; i < m; ++i) {
                scalar_t yi = y[i];
                if (std::abs(yi) < 1e-15) continue;
                index_t start = lp.csr_row_ptr[i];
                index_t end = lp.csr_row_ptr[i + 1];
                for (index_t p = start; p < end; ++p) {
                    Aty[lp.csr_col_ind[p]] += lp.csr_values[p] * yi;
                }
            }
        }

        for (index_t j = 0; j < n; ++j) {
            scalar_t r = lp.c[j] + Aty[j];
            scalar_t xj = x[j];
            scalar_t lb = lp.col_lower[j];
            scalar_t ub = lp.col_upper[j];
            scalar_t v = 0.0;

            // If not at lower bound, r must be <= 0
            if ((lb <= -kInf || xj > lb + tolerance) && r < 0.0) {
                v = std::max(v, -r);
            }
            // If not at upper bound, r must be >= 0
            if ((ub >= kInf || xj < ub - tolerance) && r > 0.0) {
                v = std::max(v, r);
            }
            max_d_viol = std::max(max_d_viol, v);
        }
        res.max_dual_stationarity_violation = max_d_viol;
    }

    // Determine aggregate validity
    res.is_feasible = (max_b_viol <= tolerance) && (max_c_viol <= tolerance);
    scalar_t obj_tol = std::max(tolerance, static_cast<scalar_t>(1e-4) * (1.0 + std::abs(obj)));
    res.is_objective_consistent = (res.objective_mismatch <= obj_tol);
    res.is_dual_consistent = (y.empty() || max_d_viol <= tolerance * 10.0);

    return res;
}

} // namespace pipepye::solver
