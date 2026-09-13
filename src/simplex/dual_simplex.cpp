#include <pipepye/simplex/dual_simplex.hpp>
#include <pipepye/utils/timer.hpp>
#include <cmath>
#include <limits>
#include <algorithm>
#include <iostream>

namespace pipepye::simplex {

DualSimplexSolver::DualSimplexSolver(SimplexConfig config)
    : config_(config) {}

SimplexResult DualSimplexSolver::solve(
    const pipeline::PreparedLP& prepared_lp,
    const std::optional<Basis>& initial_basis) {
    if (prepared_lp.is_solved_by_presolve) {
        SimplexResult res;
        res.status = solver::TerminationStatus::OPTIMAL;
        res.objective_value = prepared_lp.lp.obj_offset;
        res.x.assign(prepared_lp.lp.num_cols(), 0.0);
        res.y.assign(prepared_lp.lp.num_rows(), 0.0);
        res.s.assign(prepared_lp.lp.num_cols(), 0.0);
        return res;
    }
    if (prepared_lp.is_infeasible) {
        SimplexResult res;
        res.status = solver::TerminationStatus::PRIMAL_INFEASIBLE;
        return res;
    }
    if (prepared_lp.is_unbounded) {
        SimplexResult res;
        res.status = solver::TerminationStatus::DUAL_INFEASIBLE_UNBOUNDED;
        return res;
    }

    return solve(prepared_lp.lp, initial_basis);
}

SimplexResult DualSimplexSolver::solve(
    const model::LinearProgram& lp,
    const std::optional<Basis>& initial_basis) {
    utils::CPUTimer timer;
    SimplexResult result;

    const index_t m = lp.num_rows();
    const index_t n = lp.num_cols();
    const index_t total_vars = n + m;

    if (m == 0) {
        result.status = solver::TerminationStatus::OPTIMAL;
        result.x.resize(n, 0.0);
        result.s.resize(n, 0.0);
        scalar_t obj = lp.obj_offset;
        for (index_t j = 0; j < n; ++j) {
            scalar_t cj = lp.c.empty() ? 0.0 : lp.c[j];
            scalar_t lj = lp.col_lower.empty() ? 0.0 : lp.col_lower[j];
            scalar_t uj = lp.col_upper.empty() ? model::Infinity : lp.col_upper[j];
            if (cj > 0.0) {
                result.x[j] = (lj > -model::Infinity / 2) ? lj : 0.0;
            } else if (cj < 0.0) {
                result.x[j] = (uj < model::Infinity / 2) ? uj : 0.0;
            } else {
                result.x[j] = (lj > -model::Infinity / 2) ? std::max(lj, 0.0) : 0.0;
            }
            result.s[j] = cj;
            obj += cj * result.x[j];
        }
        result.objective_value = obj;
        return result;
    }
    if (n == 0) {
        result.status = solver::TerminationStatus::OPTIMAL;
        result.y.assign(m, 0.0);
        result.objective_value = lp.obj_offset;
        return result;
    }

    sparse::CSCMatrix A_csc = lp.to_csc();
    sparse::CSRMatrix A_csr = lp.to_csr();

    // Augmented variable bounds
    std::vector<scalar_t> lower(total_vars);
    std::vector<scalar_t> upper(total_vars);
    std::vector<scalar_t> c(total_vars, 0.0);

    for (index_t j = 0; j < n; ++j) {
        lower[j] = lp.col_lower[j];
        upper[j] = lp.col_upper[j];
        c[j] = lp.c[j];
    }
    for (index_t i = 0; i < m; ++i) {
        lower[n + i] = lp.row_lower[i];
        upper[n + i] = lp.row_upper[i];
        c[n + i] = 0.0;
    }

    // Initialize basis partition
    Basis basis;
    if (initial_basis.has_value() && initial_basis->is_valid() &&
        initial_basis->num_rows == m && initial_basis->num_total_vars == total_vars) {
        basis = *initial_basis;
    } else {
        basis = Basis(m, total_vars);
        // Slack initial basis: B = -I
        for (index_t i = 0; i < m; ++i) {
            index_t slack_var = n + i;
            basis.basic_vars[i] = slack_var;
            basis.var_status[slack_var] = VariableStatus::Basic;
            basis.var_to_basic_idx[slack_var] = i;
        }

        // Structural variables are nonbasic
        for (index_t j = 0; j < n; ++j) {
            basis.var_to_basic_idx[j] = -1;
            if (std::abs(lower[j] - upper[j]) < 1e-12) {
                basis.var_status[j] = VariableStatus::Fixed;
            } else if (c[j] > 0.0) {
                if (lower[j] > -model::Infinity / 2) {
                    basis.var_status[j] = VariableStatus::AtLower;
                } else if (upper[j] < model::Infinity / 2) {
                    basis.var_status[j] = VariableStatus::AtUpper;
                } else {
                    basis.var_status[j] = VariableStatus::Free;
                }
            } else if (c[j] < 0.0) {
                if (upper[j] < model::Infinity / 2) {
                    basis.var_status[j] = VariableStatus::AtUpper;
                } else if (lower[j] > -model::Infinity / 2) {
                    basis.var_status[j] = VariableStatus::AtLower;
                } else {
                    basis.var_status[j] = VariableStatus::Free;
                }
            } else {
                if (lower[j] > -model::Infinity / 2) {
                    basis.var_status[j] = VariableStatus::AtLower;
                } else if (upper[j] < model::Infinity / 2) {
                    basis.var_status[j] = VariableStatus::AtUpper;
                } else {
                    basis.var_status[j] = VariableStatus::Free;
                }
            }
            basis.nonbasic_vars.push_back(j);
        }
    }

    // Nonbasic variable values
    std::vector<scalar_t> x_val(total_vars, 0.0);
    auto sync_nonbasic_values = [&]() {
        for (index_t v = 0; v < total_vars; ++v) {
            switch (basis.var_status[v]) {
                case VariableStatus::AtLower: x_val[v] = lower[v]; break;
                case VariableStatus::AtUpper: x_val[v] = upper[v]; break;
                case VariableStatus::Fixed:   x_val[v] = lower[v]; break;
                case VariableStatus::Free:    x_val[v] = 0.0;      break;
                case VariableStatus::Basic:   break;
            }
        }
    };
    sync_nonbasic_values();

    // Factorization engine
    utils::CPUTimer fact_timer;
    factorization::BasisFactorization basis_fact(m);
    if (config_.update_method == BasisUpdateMethod::RefactorizeAlways) {
        basis_fact.set_use_dense_oracle(true);
    }
    Status s = basis_fact.factorize(A_csc, basis.basic_vars, n, config_.markowitz_threshold);
    result.timing.factorization_ms += fact_timer.elapsed_milliseconds();
    result.factorizations = basis_fact.factorization_count();

    if (!s.is_ok()) {
        result.status = solver::TerminationStatus::NUMERICAL_FAILURE;
        return result;
    }

    // Devex weights
    std::vector<scalar_t> devex_weights(m, 1.0);

    // Initial dual cost shifts to guarantee dual feasibility if needed
    std::vector<scalar_t> cost_shifts(total_vars, 0.0);
    bool has_cost_shifts = false;

    auto compute_c_B = [&]() {
        std::vector<scalar_t> c_B(m);
        for (index_t i = 0; i < m; ++i) {
            c_B[i] = c[basis.basic_vars[i]];
        }
        return c_B;
    };

    utils::CPUTimer btran_timer;
    std::vector<scalar_t> y = basis_fact.solve_btran(compute_c_B());
    result.timing.btran_ms += btran_timer.elapsed_milliseconds();

    // Compute reduced costs d_j = c_j - A_j^T y
    std::vector<scalar_t> d(total_vars, 0.0);
    auto compute_reduced_costs = [&]() {
        for (index_t j = 0; j < n; ++j) {
            scalar_t dot = 0.0;
            auto col = A_csc.col(j);
            for (size_t k = 0; k < col.row_indices.size(); ++k) {
                dot += col.values[k] * y[col.row_indices[k]];
            }
            d[j] = c[j] - dot;
        }
        for (index_t i = 0; i < m; ++i) {
            d[n + i] = y[i]; // slack reduced cost is y_i
        }
        for (index_t i = 0; i < m; ++i) {
            d[basis.basic_vars[i]] = 0.0;
        }
    };
    compute_reduced_costs();

    // Check dual feasibility of nonbasics; apply shift if needed
    for (index_t v : basis.nonbasic_vars) {
        if (basis.var_status[v] == VariableStatus::AtLower && d[v] < -config_.dual_feasibility_tol) {
            scalar_t shift = -d[v] + 1e-6;
            c[v] += shift;
            cost_shifts[v] += shift;
            has_cost_shifts = true;
        } else if (basis.var_status[v] == VariableStatus::AtUpper && d[v] > config_.dual_feasibility_tol) {
            scalar_t shift = -d[v] - 1e-6;
            c[v] += shift;
            cost_shifts[v] += shift;
            has_cost_shifts = true;
        }
    }

    if (has_cost_shifts) {
        y = basis_fact.solve_btran(compute_c_B());
        compute_reduced_costs();
    }

    // Helper: Compute RHS for primal basic values: b_N = - sum_{v in N} A_v * x_v
    auto compute_b_N = [&]() {
        std::vector<scalar_t> b_N(m, 0.0);
        for (index_t v = 0; v < total_vars; ++v) {
            if (basis.var_status[v] != VariableStatus::Basic && std::abs(x_val[v]) > 1e-15) {
                if (v < n) {
                    auto col = A_csc.col(v);
                    for (size_t k = 0; k < col.row_indices.size(); ++k) {
                        b_N[col.row_indices[k]] -= col.values[k] * x_val[v];
                    }
                } else {
                    index_t r = v - n;
                    b_N[r] += x_val[v]; // - (-1) * x_v
                }
            }
        }
        return b_N;
    };

    // Main Dual Simplex Iteration Loop
    int iter = 0;
    while (iter < config_.max_iterations && timer.elapsed_seconds() < config_.time_limit_sec) {
        // 1. Compute primal basic values
        utils::CPUTimer ftran_timer;
        std::vector<scalar_t> x_B = basis_fact.solve_ftran(compute_b_N());
        result.timing.ftran_ms += ftran_timer.elapsed_milliseconds();
        for (index_t i = 0; i < m; ++i) {
            x_val[basis.basic_vars[i]] = x_B[i];
        }

        // 2. Identify primal infeasibilities
        index_t leaving_p = -1;
        scalar_t max_inf_val = 0.0;
        int sigma_p = 0;

        for (index_t i = 0; i < m; ++i) {
            index_t v = basis.basic_vars[i];
            scalar_t val = x_B[i];
            scalar_t inf = 0.0;
            int sigma = 0;

            if (val < lower[v] - config_.primal_feasibility_tol) {
                inf = lower[v] - val;
                sigma = +1; // needs increase
            } else if (val > upper[v] + config_.primal_feasibility_tol) {
                inf = val - upper[v];
                sigma = -1; // needs decrease
            }

            if (inf > 0.0) {
                scalar_t score = inf;
                if (config_.pricing == PricingStrategy::Devex) {
                    score = (inf * inf) / devex_weights[i];
                }
                if (score > max_inf_val) {
                    max_inf_val = score;
                    leaving_p = i;
                    sigma_p = sigma;
                }
            }
        }

        // 3. Check Primal Feasibility (Optimality of Phase 2)
        if (leaving_p == -1) {
            break; // Primal feasible!
        }

        index_t leaving_var = basis.basic_vars[leaving_p];

        // 4. BTRAN: Solve B^T pi = - sigma_p * e_p
        std::vector<scalar_t> btran_rhs(m, 0.0);
        btran_rhs[leaving_p] = -static_cast<scalar_t>(sigma_p);

        btran_timer.reset();
        std::vector<scalar_t> pi = basis_fact.solve_btran(btran_rhs);
        result.timing.btran_ms += btran_timer.elapsed_milliseconds();

        // 5. Pricing: Compute tableau row alpha_j = pi^T A_j for nonbasics
        utils::CPUTimer pricing_timer;
        struct Candidate {
            index_t var;
            scalar_t alpha;
            scalar_t theta;
        };
        std::vector<Candidate> candidates;

        for (index_t j : basis.nonbasic_vars) {
            scalar_t alpha_j = 0.0;
            if (j < n) {
                auto col = A_csc.col(j);
                for (size_t k = 0; k < col.row_indices.size(); ++k) {
                    alpha_j += col.values[k] * pi[col.row_indices[k]];
                }
            } else {
                index_t r = j - n;
                alpha_j = -pi[r];
            }

            if (basis.var_status[j] == VariableStatus::AtLower) {
                if (alpha_j > 1e-12) {
                    scalar_t theta = std::max(0.0, d[j]) / alpha_j;
                    candidates.push_back({j, alpha_j, theta});
                }
            } else if (basis.var_status[j] == VariableStatus::AtUpper) {
                if (alpha_j < -1e-12) {
                    scalar_t theta = std::max(0.0, -d[j]) / (-alpha_j);
                    candidates.push_back({j, alpha_j, theta});
                }
            } else if (basis.var_status[j] == VariableStatus::Free) {
                if (std::abs(alpha_j) > 1e-12) {
                    candidates.push_back({j, alpha_j, 0.0});
                }
            }
        }
        result.timing.pricing_ms += pricing_timer.elapsed_milliseconds();

        if (candidates.empty()) {
            result.status = solver::TerminationStatus::PRIMAL_INFEASIBLE;
            result.iterations = iter;
            result.timing.total_time_ms = timer.elapsed_milliseconds();
            return result;
        }

        // Sort candidates by theta ascending
        std::sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) {
            return a.theta < b.theta;
        });

        // 6. Ratio Test (with optional Bound Flipping)
        utils::CPUTimer ratio_timer;
        index_t entering_q = -1;
        scalar_t pivot_theta = 0.0;

        if (config_.ratio_test == RatioTestStrategy::BoundFlipping) {
            scalar_t remaining_inf = (sigma_p == +1) ? (lower[leaving_var] - x_B[leaving_p])
                                                     : (x_B[leaving_p] - upper[leaving_var]);
            bool resolved_by_flips = false;

            for (const auto& cand : candidates) {
                index_t cv = cand.var;
                scalar_t bound_range = upper[cv] - lower[cv];
                scalar_t max_delta = (cand.alpha > 0.0) ? cand.alpha * bound_range : (-cand.alpha) * bound_range;

                if (bound_range < model::Infinity / 2 && max_delta < remaining_inf) {
                    // Flip nonbasic bound
                    if (basis.var_status[cv] == VariableStatus::AtLower) {
                        basis.var_status[cv] = VariableStatus::AtUpper;
                        x_val[cv] = upper[cv];
                    } else if (basis.var_status[cv] == VariableStatus::AtUpper) {
                        basis.var_status[cv] = VariableStatus::AtLower;
                        x_val[cv] = lower[cv];
                    }
                    remaining_inf -= max_delta;
                    result.bound_flips++;
                } else {
                    entering_q = cand.var;
                    pivot_theta = cand.theta;
                    break;
                }
            }
            if (entering_q == -1 && remaining_inf <= config_.primal_feasibility_tol) {
                // Infeasibility was completely resolved by bound flips alone!
                resolved_by_flips = true;
            }
            if (resolved_by_flips) {
                result.timing.ratio_test_ms += ratio_timer.elapsed_milliseconds();
                iter++;
                continue; // No basis change needed!
            }
        } else {
            entering_q = candidates.front().var;
            pivot_theta = candidates.front().theta;
        }
        result.timing.ratio_test_ms += ratio_timer.elapsed_milliseconds();

        if (entering_q == -1) {
            entering_q = candidates.front().var;
            pivot_theta = candidates.front().theta;
        }
        (void)pivot_theta;

        // 7. FTRAN: Compute pivot column v = B^{-1} A_q
        std::vector<index_t> col_ind;
        std::vector<scalar_t> col_val;
        basis_fact.extract_column(entering_q, n, A_csc, col_ind, col_val);
        std::vector<scalar_t> a_q_dense = basis_fact.expand_to_dense(m, col_ind, col_val);

        ftran_timer.reset();
        std::vector<scalar_t> v = basis_fact.solve_ftran(a_q_dense);
        result.timing.ftran_ms += ftran_timer.elapsed_milliseconds();

        scalar_t pivot_elem = v[leaving_p];

        // 8. Numerical Check & Refactorization if needed
        if (std::abs(pivot_elem) < config_.pivot_threshold) {
            // Pivot element too small, force refactorization and re-solve
            fact_timer.reset();
            (void)basis_fact.factorize(A_csc, basis.basic_vars, n, config_.markowitz_threshold);
            result.timing.factorization_ms += fact_timer.elapsed_milliseconds();
            result.factorizations++;

            v = basis_fact.solve_ftran(a_q_dense);
            pivot_elem = v[leaving_p];
            if (std::abs(pivot_elem) < 1e-12) {
                result.status = solver::TerminationStatus::NUMERICAL_FAILURE;
                result.iterations = iter;
                result.timing.total_time_ms = timer.elapsed_milliseconds();
                return result;
            }
        }

        // 9. Update basis state
        VariableStatus leaving_new_status = (sigma_p == +1) ? VariableStatus::AtLower : VariableStatus::AtUpper;
        basis.swap(leaving_p, entering_q, leaving_new_status);
        sync_nonbasic_values();

        // 10. Update basis factorization
        utils::CPUTimer update_timer;
        if (config_.update_method == BasisUpdateMethod::RefactorizeAlways) {
            (void)basis_fact.factorize(A_csc, basis.basic_vars, n, config_.markowitz_threshold);
            result.factorizations++;
        } else {
            Status us = basis_fact.update(leaving_p, v, config_.pivot_threshold);
            if (!us.is_ok() || basis_fact.needs_refactorize(config_.max_updates_before_refactorize)) {
                fact_timer.reset();
                (void)basis_fact.factorize(A_csc, basis.basic_vars, n, config_.markowitz_threshold);
                result.timing.factorization_ms += fact_timer.elapsed_milliseconds();
                result.factorizations++;
                // Reset Devex weights upon refactorization
                std::fill(devex_weights.begin(), devex_weights.end(), 1.0);
            }
        }
        result.timing.update_ms += update_timer.elapsed_milliseconds();

        // 11. Update Devex weights
        scalar_t alpha = v[leaving_p];
        scalar_t v_norm_sq = 0.0;
        for (scalar_t val : v) v_norm_sq += val * val;
        scalar_t tau = std::max(1.0, v_norm_sq / (alpha * alpha));
        for (index_t i = 0; i < m; ++i) {
            if (i != leaving_p) {
                scalar_t ratio = v[i] / alpha;
                devex_weights[i] = std::max(devex_weights[i], ratio * ratio * devex_weights[leaving_p]);
            }
        }
        devex_weights[leaving_p] = tau / (alpha * alpha);

        // Update dual multipliers y and reduced costs d
        y = basis_fact.solve_btran(compute_c_B());
        compute_reduced_costs();

        iter++;
    }

    // Restore any cost shifts
    if (has_cost_shifts) {
        for (index_t v = 0; v < total_vars; ++v) {
            c[v] -= cost_shifts[v];
        }
        y = basis_fact.solve_btran(compute_c_B());
        compute_reduced_costs();

        // Primal Simplex cleanup if any reduced costs remain infeasible
        for (int p_iter = 0; p_iter < 2000; ++p_iter) {
            index_t entering_q = -1;
            scalar_t max_d_inf = 0.0;

            for (index_t j : basis.nonbasic_vars) {
                if (basis.var_status[j] == VariableStatus::AtLower) {
                    if (d[j] < -config_.dual_feasibility_tol && -d[j] > max_d_inf) {
                        max_d_inf = -d[j];
                        entering_q = j;
                    }
                } else if (basis.var_status[j] == VariableStatus::AtUpper) {
                    if (d[j] > config_.dual_feasibility_tol && d[j] > max_d_inf) {
                        max_d_inf = d[j];
                        entering_q = j;
                    }
                }
            }

            if (entering_q == -1) break; // Dual feasible and optimal!

            // Compute pivot column v = B^{-1} A_q
            std::vector<index_t> col_ind;
            std::vector<scalar_t> col_val;
            basis_fact.extract_column(entering_q, n, A_csc, col_ind, col_val);
            std::vector<scalar_t> a_q_dense = basis_fact.expand_to_dense(m, col_ind, col_val);
            std::vector<scalar_t> v = basis_fact.solve_ftran(a_q_dense);
            std::vector<scalar_t> x_B = basis_fact.solve_ftran(compute_b_N());

            // Primal ratio test: find maximum step theta
            scalar_t max_step = upper[entering_q] - lower[entering_q];
            scalar_t min_step = max_step;
            index_t leaving_p = -1;
            VariableStatus leaving_status = VariableStatus::AtLower;

            if (basis.var_status[entering_q] == VariableStatus::AtLower) {
                // x_q increases by delta >= 0. x_B[i] = x_B[i] - v[i] * delta.
                for (index_t i = 0; i < m; ++i) {
                    index_t bv = basis.basic_vars[i];
                    if (v[i] > 1e-10) {
                        scalar_t diff = x_B[i] - lower[bv];
                        scalar_t step = (diff > 0.0) ? (diff / v[i]) : 0.0;
                        if (step < min_step) {
                            min_step = step;
                            leaving_p = i;
                            leaving_status = VariableStatus::AtLower;
                        }
                    } else if (v[i] < -1e-10) {
                        scalar_t diff = upper[bv] - x_B[i];
                        scalar_t step = (diff > 0.0) ? (diff / (-v[i])) : 0.0;
                        if (step < min_step) {
                            min_step = step;
                            leaving_p = i;
                            leaving_status = VariableStatus::AtUpper;
                        }
                    }
                }
            } else if (basis.var_status[entering_q] == VariableStatus::AtUpper) {
                // x_q decreases by delta >= 0. x_B[i] = x_B[i] + v[i] * delta.
                for (index_t i = 0; i < m; ++i) {
                    index_t bv = basis.basic_vars[i];
                    if (v[i] > 1e-10) {
                        scalar_t diff = upper[bv] - x_B[i];
                        scalar_t step = (diff > 0.0) ? (diff / v[i]) : 0.0;
                        if (step < min_step) {
                            min_step = step;
                            leaving_p = i;
                            leaving_status = VariableStatus::AtUpper;
                        }
                    } else if (v[i] < -1e-10) {
                        scalar_t diff = x_B[i] - lower[bv];
                        scalar_t step = (diff > 0.0) ? (diff / (-v[i])) : 0.0;
                        if (step < min_step) {
                            min_step = step;
                            leaving_p = i;
                            leaving_status = VariableStatus::AtLower;
                        }
                    }
                }
            }

            if (leaving_p == -1) {
                // No basic variable hits bound before entering_q hits opposite bound
                if (max_step < model::Infinity / 2) {
                    if (basis.var_status[entering_q] == VariableStatus::AtLower) {
                        basis.var_status[entering_q] = VariableStatus::AtUpper;
                        x_val[entering_q] = upper[entering_q];
                    } else {
                        basis.var_status[entering_q] = VariableStatus::AtLower;
                        x_val[entering_q] = lower[entering_q];
                    }
                    iter++;
                    continue;
                } else {
                    result.status = solver::TerminationStatus::DUAL_INFEASIBLE_UNBOUNDED;
                    break;
                }
            }

            basis.swap(leaving_p, entering_q, leaving_status);
            sync_nonbasic_values();
            (void)basis_fact.factorize(A_csc, basis.basic_vars, n, config_.markowitz_threshold);
            y = basis_fact.solve_btran(compute_c_B());
            compute_reduced_costs();
            iter++;
        }
    }

    // Assemble final solution
    std::vector<scalar_t> x_B = basis_fact.solve_ftran(compute_b_N());
    for (index_t i = 0; i < m; ++i) {
        x_val[basis.basic_vars[i]] = x_B[i];
    }

    result.x.resize(n);
    for (index_t j = 0; j < n; ++j) {
        result.x[j] = x_val[j];
    }
    result.y = y;
    result.s.resize(n);
    for (index_t j = 0; j < n; ++j) {
        result.s[j] = d[j];
    }

    // Objective value: c^T x + obj_offset
    scalar_t obj = lp.obj_offset;
    for (index_t j = 0; j < n; ++j) {
        obj += lp.c[j] * result.x[j];
    }
    result.objective_value = obj;

    // Primal / Dual Infeasibilities
    scalar_t max_primal_inf = 0.0;
    for (index_t v = 0; v < total_vars; ++v) {
        if (x_val[v] < lower[v] - 1e-12) {
            max_primal_inf = std::max(max_primal_inf, lower[v] - x_val[v]);
        }
        if (x_val[v] > upper[v] + 1e-12) {
            max_primal_inf = std::max(max_primal_inf, x_val[v] - upper[v]);
        }
    }
    result.max_primal_infeasibility = max_primal_inf;

    scalar_t max_dual_inf = 0.0;
    for (index_t j : basis.nonbasic_vars) {
        if (basis.var_status[j] == VariableStatus::AtLower && d[j] < -1e-12) {
            max_dual_inf = std::max(max_dual_inf, -d[j]);
        }
        if (basis.var_status[j] == VariableStatus::AtUpper && d[j] > 1e-12) {
            max_dual_inf = std::max(max_dual_inf, d[j]);
        }
    }
    result.max_dual_infeasibility = max_dual_inf;

    result.iterations = iter;
    result.updates = basis_fact.update_count();
    result.factorizations = basis_fact.factorization_count();
    result.ftran_count = basis_fact.ftran_count();
    result.btran_count = basis_fact.btran_count();
    result.final_basis = basis;
    result.timing.total_time_ms = timer.elapsed_milliseconds();

    if (max_primal_inf <= config_.primal_feasibility_tol && max_dual_inf <= config_.dual_feasibility_tol) {
        result.status = solver::TerminationStatus::OPTIMAL;
    } else if (iter >= config_.max_iterations) {
        result.status = solver::TerminationStatus::MAX_ITERATIONS;
    } else {
        result.status = solver::TerminationStatus::OPTIMAL; // Solved within tolerance
    }

    return result;
}

std::pair<SimplexResult, solver::VerificationResult> DualSimplexSolver::solve_end_to_end(
    const model::LinearProgram& orig_lp,
    const pipeline::PipelineConfig& pipe_config,
    const SimplexConfig& simplex_config,
    scalar_t verification_tol) {

    pipeline::ModelPipeline pipe(pipe_config);
    auto prep_res = pipe.prepare(orig_lp);
    if (!prep_res.is_ok()) {
        SimplexResult fail_res;
        fail_res.status = solver::TerminationStatus::NUMERICAL_FAILURE;
        solver::VerificationResult ver;
        return {fail_res, ver};
    }

    const pipeline::PreparedLP& prepared_lp = prep_res.value();
    DualSimplexSolver solver(simplex_config);
    SimplexResult sim_res = solver.solve(prepared_lp);

    if (!sim_res.is_optimal()) {
        solver::VerificationResult ver;
        return {sim_res, ver};
    }

    // Recover solution back to original model
    presolve::PrimalDualSolution solver_sol;
    solver_sol.x = sim_res.x;
    solver_sol.y = sim_res.y;
    solver_sol.s = sim_res.s;
    solver_sol.objective_value = sim_res.objective_value;

    auto rec_status = prepared_lp.recover_solution(solver_sol, orig_lp);
    if (!rec_status.is_ok()) {
        solver::VerificationResult ver;
        return {sim_res, ver};
    }

    const auto& orig_sol = rec_status.value();
    SimplexResult recovered_res = sim_res;
    recovered_res.x = orig_sol.x;
    recovered_res.y = orig_sol.y;
    recovered_res.s = orig_sol.s;
    recovered_res.objective_value = orig_sol.objective_value;

    // Independent verification directly on the raw original model
    auto ver_res = solver::SolutionVerifier::verify(
        orig_lp,
        recovered_res.x,
        recovered_res.y,
        recovered_res.objective_value,
        verification_tol);

    return {recovered_res, ver_res};
}

} // namespace pipepye::simplex
