#include <pipepye/presolve/presolve_pass.hpp>
#include <cmath>
#include <algorithm>

namespace pipepye::presolve {

// =============================================================================
// 1. EmptyRowColPass
// =============================================================================
StatusOr<PassStats> EmptyRowColPass::run(PresolveContext& ctx) {
    PassStats stats;
    stats.pass_name = name();
    const scalar_t eps = ctx.options().tolerance;
    const scalar_t inf = ctx.options().infinity_threshold;

    // A. Detect Empty Rows
    for (index_t i = 0; i < ctx.num_rows(); ++i) {
        if (!ctx.is_row_active(i) || ctx.row_degree(i) > 0) continue;

        scalar_t lb = ctx.row_lower(i);
        scalar_t ub = ctx.row_upper(i);

        // An empty constraint is 0.0 in [lb, ub]
        if (lb <= eps && ub >= -eps) {
            // Satisfied vacuously -> redundant
            ctx.remove_row(i);
            ctx.postsolve_mgr().add_action(std::make_unique<RedundantRowAction>(i));
            ++stats.rows_removed;
            ++stats.redundant_rows;
        } else {
            // 0.0 is outside [lb, ub] -> mathematically infeasible
            ctx.set_status(PresolveStatus::Infeasible);
            return stats;
        }
    }

    // B. Detect Empty Columns (Variables not in any constraint)
    for (index_t j = 0; j < ctx.num_cols(); ++j) {
        if (!ctx.is_col_active(j) || ctx.col_degree(j) > 0) continue;

        scalar_t c = ctx.cost(j);
        scalar_t lb = ctx.col_lower(j);
        scalar_t ub = ctx.col_upper(j);

        if (c > eps) {
            // Minimizing positive cost c * x -> set x to smallest possible value
            if (lb > -inf) {
                ctx.fix_variable(j, lb);
                ++stats.cols_removed;
                ++stats.variables_fixed;
            } else {
                // x can go to -infinity with cost c > 0 -> unbounded below!
                ctx.set_status(PresolveStatus::Unbounded);
                return stats;
            }
        } else if (c < -eps) {
            // Minimizing negative cost c * x -> set x to largest possible value
            if (ub < inf) {
                ctx.fix_variable(j, ub);
                ++stats.cols_removed;
                ++stats.variables_fixed;
            } else {
                // x can go to +infinity with negative cost -> unbounded!
                ctx.set_status(PresolveStatus::Unbounded);
                return stats;
            }
        } else {
            // c == 0: variable has zero objective impact and no constraints
            if (lb > ub + eps) {
                ctx.set_status(PresolveStatus::Infeasible);
                return stats;
            }
            scalar_t val = 0.0;
            if (val < lb) val = lb;
            if (val > ub) val = ub;
            ctx.fix_variable(j, val);
            ++stats.cols_removed;
            ++stats.variables_fixed;
        }
    }

    return stats;
}

// =============================================================================
// 2. FixedVariablePass
// =============================================================================
StatusOr<PassStats> FixedVariablePass::run(PresolveContext& ctx) {
    PassStats stats;
    stats.pass_name = name();
    const scalar_t eps = ctx.options().tolerance;

    for (index_t j = 0; j < ctx.num_cols(); ++j) {
        if (!ctx.is_col_active(j)) continue;

        scalar_t lb = ctx.col_lower(j);
        scalar_t ub = ctx.col_upper(j);

        if (lb > ub + eps) {
            ctx.set_status(PresolveStatus::Infeasible);
            return stats;
        }

        if (std::abs(ub - lb) <= eps) {
            scalar_t fixed_val = lb;
            ctx.fix_variable(j, fixed_val);
            ++stats.cols_removed;
            ++stats.variables_fixed;
        }
    }

    return stats;
}

// =============================================================================
// 3. SingletonPass
// =============================================================================
StatusOr<PassStats> SingletonPass::run(PresolveContext& ctx) {
    PassStats stats;
    stats.pass_name = name();
    const scalar_t eps = ctx.options().tolerance;
    const scalar_t inf = ctx.options().infinity_threshold;

    // A. Singleton Rows (exactly 1 active column in row)
    for (index_t i = 0; i < ctx.num_rows(); ++i) {
        if (!ctx.is_row_active(i) || ctx.row_degree(i) != 1) continue;

        // Find the single active column
        index_t col = -1;
        scalar_t a = 0.0;
        for (const auto& entry : ctx.row_entries(i)) {
            if (ctx.is_col_active(entry.index)) {
                col = entry.index;
                a = entry.value;
                break;
            }
        }

        if (col == -1 || std::abs(a) <= eps) continue;

        scalar_t row_l = ctx.row_lower(i);
        scalar_t row_u = ctx.row_upper(i);

        scalar_t new_lb, new_ub;
        if (a > 0.0) {
            new_lb = (row_l > -inf) ? (row_l / a) : -inf;
            new_ub = (row_u < inf) ? (row_u / a) : inf;
        } else {
            new_lb = (row_u < inf) ? (row_u / a) : -inf;
            new_ub = (row_l > -inf) ? (row_l / a) : inf;
        }

        scalar_t cur_lb = ctx.col_lower(col);
        scalar_t cur_ub = ctx.col_upper(col);

        scalar_t tight_lb = std::max(cur_lb, new_lb);
        scalar_t tight_ub = std::min(cur_ub, new_ub);

        if (tight_lb > tight_ub + eps) {
            ctx.set_status(PresolveStatus::Infeasible);
            return stats;
        }

        if (tight_lb > cur_lb + eps || tight_ub < cur_ub - eps) {
            ctx.set_col_lower(col, tight_lb);
            ctx.set_col_upper(col, tight_ub);
            ++stats.bounds_tightened;
        }

        ctx.postsolve_mgr().add_action(
            std::make_unique<SingletonRowAction>(i, col, a, row_l, row_u));
        ctx.remove_row(i);
        ++stats.rows_removed;
        ++stats.redundant_rows;
    }

    // B. Singleton Columns (column appears in exactly 1 equality constraint)
    for (index_t j = 0; j < ctx.num_cols(); ++j) {
        if (!ctx.is_col_active(j) || ctx.col_degree(j) != 1) continue;

        index_t row = -1;
        scalar_t a = 0.0;
        for (const auto& entry : ctx.col_entries(j)) {
            if (ctx.is_row_active(entry.index)) {
                row = entry.index;
                a = entry.value;
                break;
            }
        }

        if (row == -1 || std::abs(a) <= eps) continue;

        scalar_t row_l = ctx.row_lower(row);
        scalar_t row_u = ctx.row_upper(row);

        // Only substitute if the row is an equality constraint
        if (std::abs(row_u - row_l) <= eps && ctx.row_degree(row) >= 2) {
            scalar_t rhs = row_l;
            scalar_t cost_j = ctx.cost(j);

            // Collect other active entries in row
            std::vector<std::pair<index_t, scalar_t>> others;
            for (const auto& entry : ctx.row_entries(row)) {
                if (entry.index != j && ctx.is_col_active(entry.index)) {
                    others.push_back({entry.index, entry.value});
                }
            }

            // Substitute x_j = (rhs - sum_{k != j} a_{row, k} x_k) / a into objective:
            // c_k <- c_k - c_j * (a_{row, k} / a)
            // obj_offset <- obj_offset + c_j * (rhs / a)
            for (const auto& [k, val] : others) {
                scalar_t delta_c = cost_j * (val / a);
                ctx.set_cost(k, ctx.cost(k) - delta_c);
            }
            ctx.add_obj_offset(cost_j * (rhs / a));

            // Transfer bounds on x_j to row activity
            scalar_t col_l = ctx.col_lower(j);
            scalar_t col_u = ctx.col_upper(j);

            scalar_t new_row_l, new_row_u;
            if (a > 0.0) {
                new_row_l = (col_u < inf) ? (rhs - a * col_u) : -inf;
                new_row_u = (col_l > -inf) ? (rhs - a * col_l) : inf;
            } else {
                new_row_l = (col_l > -inf) ? (rhs - a * col_l) : -inf;
                new_row_u = (col_u < inf) ? (rhs - a * col_u) : inf;
            }

            ctx.set_row_lower(row, new_row_l);
            ctx.set_row_upper(row, new_row_u);

            ctx.postsolve_mgr().add_action(
                std::make_unique<SingletonColumnAction>(j, row, a, cost_j, rhs, others));

            ctx.remove_col(j);
            ++stats.cols_removed;
        }
    }

    return stats;
}

// =============================================================================
// 4. ForcingRedundancyPass
// =============================================================================
StatusOr<PassStats> ForcingRedundancyPass::run(PresolveContext& ctx) {
    PassStats stats;
    stats.pass_name = name();
    const scalar_t eps = ctx.options().tolerance;
    const scalar_t inf = ctx.options().infinity_threshold;

    for (index_t i = 0; i < ctx.num_rows(); ++i) {
        if (!ctx.is_row_active(i) || ctx.row_degree(i) == 0) continue;

        scalar_t min_act = 0.0;
        scalar_t max_act = 0.0;
        int min_inf_count = 0;
        int max_inf_count = 0;

        for (const auto& entry : ctx.row_entries(i)) {
            index_t j = entry.index;
            if (!ctx.is_col_active(j)) continue;

            scalar_t a = entry.value;
            scalar_t lb = ctx.col_lower(j);
            scalar_t ub = ctx.col_upper(j);

            if (a > 0.0) {
                if (lb > -inf) min_act += a * lb; else ++min_inf_count;
                if (ub < inf) max_act += a * ub; else ++max_inf_count;
            } else {
                if (ub < inf) min_act += a * ub; else ++min_inf_count;
                if (lb > -inf) max_act += a * lb; else ++max_inf_count;
            }
        }

        scalar_t row_l = ctx.row_lower(i);
        scalar_t row_u = ctx.row_upper(i);

        // A. Infeasibility Check
        if (min_inf_count == 0 && row_u < inf && min_act > row_u + eps) {
            ctx.set_status(PresolveStatus::Infeasible);
            return stats;
        }
        if (max_inf_count == 0 && row_l > -inf && max_act < row_l - eps) {
            ctx.set_status(PresolveStatus::Infeasible);
            return stats;
        }

        // B. Redundancy Check: row is guaranteed satisfied for all valid x
        if (min_inf_count == 0 && max_inf_count == 0) {
            if (min_act >= row_l - eps && max_act <= row_u + eps) {
                ctx.remove_row(i);
                ctx.postsolve_mgr().add_action(std::make_unique<RedundantRowAction>(i));
                ++stats.rows_removed;
                ++stats.redundant_rows;
                continue;
            }
        }

        // C. Forcing Constraints: min_act == row_u forces all variables to their lower contribution
        if (min_inf_count == 0 && row_u < inf && std::abs(min_act - row_u) <= eps) {
            std::vector<SparseEntry> row_vars = ctx.row_entries(i);
            for (const auto& entry : row_vars) {
                index_t j = entry.index;
                if (!ctx.is_col_active(j)) continue;
                scalar_t a = entry.value;
                scalar_t forced_val = (a > 0.0) ? ctx.col_lower(j) : ctx.col_upper(j);
                ctx.fix_variable(j, forced_val);
                ++stats.variables_fixed;
                ++stats.cols_removed;
            }
            if (ctx.is_row_active(i)) {
                ctx.remove_row(i);
                ctx.postsolve_mgr().add_action(std::make_unique<RedundantRowAction>(i));
                ++stats.rows_removed;
            }
            continue;
        }

        // Max_act == row_l forces all variables to their upper contribution
        if (max_inf_count == 0 && row_l > -inf && std::abs(max_act - row_l) <= eps) {
            std::vector<SparseEntry> row_vars = ctx.row_entries(i);
            for (const auto& entry : row_vars) {
                index_t j = entry.index;
                if (!ctx.is_col_active(j)) continue;
                scalar_t a = entry.value;
                scalar_t forced_val = (a > 0.0) ? ctx.col_upper(j) : ctx.col_lower(j);
                ctx.fix_variable(j, forced_val);
                ++stats.variables_fixed;
                ++stats.cols_removed;
            }
            if (ctx.is_row_active(i)) {
                ctx.remove_row(i);
                ctx.postsolve_mgr().add_action(std::make_unique<RedundantRowAction>(i));
                ++stats.rows_removed;
            }
        }
    }

    return stats;
}

// =============================================================================
// 5. BoundTighteningPass
// =============================================================================
StatusOr<PassStats> BoundTighteningPass::run(PresolveContext& ctx) {
    PassStats stats;
    stats.pass_name = name();
    const scalar_t eps = ctx.options().tolerance;
    const scalar_t inf = ctx.options().infinity_threshold;

    for (index_t i = 0; i < ctx.num_rows(); ++i) {
        if (!ctx.is_row_active(i) || ctx.row_degree(i) <= 1) continue;

        scalar_t min_act = 0.0;
        scalar_t max_act = 0.0;
        int min_inf_count = 0;
        int max_inf_count = 0;

        for (const auto& entry : ctx.row_entries(i)) {
            index_t j = entry.index;
            if (!ctx.is_col_active(j)) continue;

            scalar_t a = entry.value;
            scalar_t lb = ctx.col_lower(j);
            scalar_t ub = ctx.col_upper(j);

            if (a > 0.0) {
                if (lb > -inf) min_act += a * lb; else ++min_inf_count;
                if (ub < inf) max_act += a * ub; else ++max_inf_count;
            } else {
                if (ub < inf) min_act += a * ub; else ++min_inf_count;
                if (lb > -inf) max_act += a * lb; else ++max_inf_count;
            }
        }

        scalar_t row_l = ctx.row_lower(i);
        scalar_t row_u = ctx.row_upper(i);

        for (const auto& entry : ctx.row_entries(i)) {
            index_t k = entry.index;
            if (!ctx.is_col_active(k)) continue;

            scalar_t a = entry.value;
            scalar_t lb = ctx.col_lower(k);
            scalar_t ub = ctx.col_upper(k);

            if (a > 0.0) {
                // Upper bound from row upper bound: a * x_k <= row_u - min_act_{-k}
                bool k_min_inf = (lb <= -inf);
                if (row_u < inf && (min_inf_count - (k_min_inf ? 1 : 0)) == 0) {
                    scalar_t min_act_other = min_act - (k_min_inf ? 0.0 : a * lb);
                    scalar_t implied_ub = (row_u - min_act_other) / a;
                    if (implied_ub < ub - eps) {
                        ctx.set_col_upper(k, implied_ub);
                        ++stats.bounds_tightened;
                        ub = implied_ub;
                    }
                }

                // Lower bound from row lower bound: a * x_k >= row_l - max_act_{-k}
                bool k_max_inf = (ub >= inf);
                if (row_l > -inf && (max_inf_count - (k_max_inf ? 1 : 0)) == 0) {
                    scalar_t max_act_other = max_act - (k_max_inf ? 0.0 : a * ub);
                    scalar_t implied_lb = (row_l - max_act_other) / a;
                    if (implied_lb > lb + eps) {
                        ctx.set_col_lower(k, implied_lb);
                        ++stats.bounds_tightened;
                        lb = implied_lb;
                    }
                }
            } else { // a < 0.0
                // a * x_k <= row_u - min_act_{-k} => x_k >= (row_u - min_act_{-k}) / a
                bool k_min_inf = (ub >= inf);
                if (row_u < inf && (min_inf_count - (k_min_inf ? 1 : 0)) == 0) {
                    scalar_t min_act_other = min_act - (k_min_inf ? 0.0 : a * ub);
                    scalar_t implied_lb = (row_u - min_act_other) / a;
                    if (implied_lb > lb + eps) {
                        ctx.set_col_lower(k, implied_lb);
                        ++stats.bounds_tightened;
                        lb = implied_lb;
                    }
                }

                // a * x_k >= row_l - max_act_{-k} => x_k <= (row_l - max_act_{-k}) / a
                bool k_max_inf = (lb <= -inf);
                if (row_l > -inf && (max_inf_count - (k_max_inf ? 1 : 0)) == 0) {
                    scalar_t max_act_other = max_act - (k_max_inf ? 0.0 : a * lb);
                    scalar_t implied_ub = (row_l - max_act_other) / a;
                    if (implied_ub < ub - eps) {
                        ctx.set_col_upper(k, implied_ub);
                        ++stats.bounds_tightened;
                        ub = implied_ub;
                    }
                }
            }

            if (lb > ub + eps) {
                ctx.set_status(PresolveStatus::Infeasible);
                return stats;
            }
        }
    }

    return stats;
}

} // namespace pipepye::presolve
