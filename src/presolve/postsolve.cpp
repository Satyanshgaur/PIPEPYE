#include <pipepye/presolve/postsolve.hpp>
#include <cmath>
#include <algorithm>

namespace pipepye::presolve {

void FixedVariableAction::postsolve(const model::LinearProgram& original_lp,
                                    PrimalDualSolution& sol) const {
    sol.x[col_idx_] = fixed_val_;

    // Compute dual reduced cost s_j = c_j - A_{*, j}^T y
    scalar_t a_dot_y = 0.0;
    if (!original_lp.csc_col_ptr.empty()) {
        index_t start = original_lp.csc_col_ptr[col_idx_];
        index_t end = original_lp.csc_col_ptr[col_idx_ + 1];
        for (index_t p = start; p < end; ++p) {
            index_t row = original_lp.csc_row_ind[p];
            scalar_t val = original_lp.csc_values[p];
            a_dot_y += val * sol.y[row];
        }
    }
    sol.s[col_idx_] = cost_ - a_dot_y;
}

void EmptyColumnAction::postsolve(const model::LinearProgram& /*original_lp*/,
                                  PrimalDualSolution& sol) const {
    sol.x[col_idx_] = val_;
    sol.s[col_idx_] = cost_;
}

void SingletonRowAction::postsolve(const model::LinearProgram& /*original_lp*/,
                                   PrimalDualSolution& sol) const {
    // If the singleton row was binding on the variable, assign consistent dual multiplier
    scalar_t xj = sol.x[col_idx_];
    scalar_t ax = coeff_ * xj;
    const scalar_t eps = 1e-7;

    if (std::abs(ax - row_lb_) <= eps && std::abs(ax - row_ub_) <= eps) {
        // Equality
        sol.y[row_idx_] = (coeff_ != 0.0) ? (sol.s[col_idx_] / coeff_) : 0.0;
    } else if (std::abs(ax - row_lb_) <= eps) {
        // Active lower bound => y_i >= 0
        scalar_t implied_y = (coeff_ != 0.0) ? (sol.s[col_idx_] / coeff_) : 0.0;
        sol.y[row_idx_] = std::max(0.0, implied_y);
    } else if (std::abs(ax - row_ub_) <= eps) {
        // Active upper bound => y_i <= 0
        scalar_t implied_y = (coeff_ != 0.0) ? (sol.s[col_idx_] / coeff_) : 0.0;
        sol.y[row_idx_] = std::min(0.0, implied_y);
    } else {
        // Non-binding row
        sol.y[row_idx_] = 0.0;
    }
}

void SingletonColumnAction::postsolve(const model::LinearProgram& /*original_lp*/,
                                     PrimalDualSolution& sol) const {
    scalar_t other_sum = 0.0;
    for (const auto& [col, val] : other_entries_) {
        other_sum += val * sol.x[col];
    }
    if (std::abs(coeff_) > 1e-15) {
        sol.x[col_idx_] = (rhs_ - other_sum) / coeff_;
        sol.y[row_idx_] = cost_ / coeff_;
        sol.s[col_idx_] = 0.0;
    }
}

void RedundantRowAction::postsolve(const model::LinearProgram& /*original_lp*/,
                                   PrimalDualSolution& sol) const {
    sol.y[row_idx_] = 0.0;
}

StatusOr<PrimalDualSolution> PostsolveManager::postsolve(
    const model::LinearProgram& original_lp,
    const PrimalDualSolution& presolved_sol) const {

    PrimalDualSolution sol;
    sol.x.assign(orig_cols_, 0.0);
    sol.y.assign(orig_rows_, 0.0);
    sol.s.assign(orig_cols_, 0.0);

    // 1. Scatter presolved retained values back to original indices
    for (index_t j = 0; j < orig_cols_; ++j) {
        index_t pre_j = orig_to_presolved_col_[j];
        if (pre_j >= 0 && pre_j < static_cast<index_t>(presolved_sol.x.size())) {
            sol.x[j] = presolved_sol.x[pre_j];
            if (pre_j < static_cast<index_t>(presolved_sol.s.size())) {
                sol.s[j] = presolved_sol.s[pre_j];
            }
        }
    }

    for (index_t i = 0; i < orig_rows_; ++i) {
        index_t pre_i = orig_to_presolved_row_[i];
        if (pre_i >= 0 && pre_i < static_cast<index_t>(presolved_sol.y.size())) {
            sol.y[i] = presolved_sol.y[pre_i];
        }
    }

    // 2. Unwind recorded postsolve transformations in reverse order (LIFO)
    for (auto it = actions_.rbegin(); it != actions_.rend(); ++it) {
        (*it)->postsolve(original_lp, sol);
    }

    // 3. Compute reconstructed objective value
    scalar_t obj = original_lp.obj_offset;
    for (index_t j = 0; j < orig_cols_; ++j) {
        obj += original_lp.c[j] * sol.x[j];
    }
    sol.objective_value = obj;

    // 4. Feasibility check against original model bounds
    const scalar_t eps = 1e-6;
    bool feasible = true;

    for (index_t j = 0; j < orig_cols_; ++j) {
        scalar_t xj = sol.x[j];
        if (xj < original_lp.col_lower[j] - eps || xj > original_lp.col_upper[j] + eps) {
            feasible = false;
            break;
        }
    }

    if (feasible && !original_lp.csr_row_ptr.empty()) {
        for (index_t i = 0; i < orig_rows_; ++i) {
            scalar_t ax = 0.0;
            index_t start = original_lp.csr_row_ptr[i];
            index_t end = original_lp.csr_row_ptr[i + 1];
            for (index_t p = start; p < end; ++p) {
                ax += original_lp.csr_values[p] * sol.x[original_lp.csr_col_ind[p]];
            }
            if (ax < original_lp.row_lower[i] - eps || ax > original_lp.row_upper[i] + eps) {
                feasible = false;
                break;
            }
        }
    }

    sol.is_feasible = feasible;

    // 5. Update transformation log with reconstructed primal and dual variables
    for (index_t j = 0; j < orig_cols_; ++j) {
        log_.record_reconstructed_variable(j, sol.x[j], sol.s[j]);
    }
    for (index_t i = 0; i < orig_rows_; ++i) {
        log_.record_reconstructed_constraint(i, sol.y[i]);
    }

    return sol;
}

} // namespace pipepye::presolve
