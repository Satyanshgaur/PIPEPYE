#include <pipepye/solver/pdhg_solver_cpu.hpp>
#include <pipepye/utils/timer.hpp>
#include <cmath>
#include <algorithm>
#include <iostream>

namespace pipepye::solver {

namespace {

const scalar_t kInf = 1e15;

inline scalar_t clip(scalar_t val, scalar_t lb, scalar_t ub) noexcept {
    if (lb > -kInf && val < lb) return lb;
    if (ub < kInf && val > ub) return ub;
    return val;
}

} // namespace

SolverResult CPUPDPOptimizer::solve(const model::LinearProgram& lp) const {
    SolverResult result;
    index_t m = lp.num_rows();
    index_t n = lp.num_cols();

    utils::CPUTimer total_timer;
    total_timer.start();

    // 0. Trivial Edge Cases: 0 variables or 0 constraints
    if (n == 0) {
        result.status = TerminationStatus::OPTIMAL;
        result.primal_objective = lp.obj_offset;
        result.dual_objective = lp.obj_offset;
        total_timer.stop();
        result.timing.pure_solve_ms = total_timer.elapsed_milliseconds();
        result.timing.total_time_ms = result.timing.pure_solve_ms;
        return result;
    }

    if (m == 0) {
        // Pure bound-constrained optimization
        result.x.resize(n, 0.0);
        result.s.resize(n, 0.0);
        bool unbounded = false;
        scalar_t obj = lp.obj_offset;

        for (index_t j = 0; j < n; ++j) {
            scalar_t cj = lp.c[j];
            scalar_t lb = lp.col_lower[j];
            scalar_t ub = lp.col_upper[j];
            if (cj > 0.0) {
                if (lb <= -kInf) { unbounded = true; break; }
                result.x[j] = lb;
            } else if (cj < 0.0) {
                if (ub >= kInf) { unbounded = true; break; }
                result.x[j] = ub;
            } else {
                result.x[j] = (lb > -kInf && ub < kInf) ? 0.5 * (lb + ub) : 0.0;
            }
            obj += cj * result.x[j];
            result.s[j] = cj;
        }

        if (unbounded) {
            result.status = TerminationStatus::DUAL_INFEASIBLE_UNBOUNDED;
        } else {
            result.status = TerminationStatus::OPTIMAL;
            result.primal_objective = obj;
            result.dual_objective = obj;
        }
        total_timer.stop();
        result.timing.pure_solve_ms = total_timer.elapsed_milliseconds();
        result.timing.total_time_ms = result.timing.pure_solve_ms;
        return result;
    }

    // 1. Memory Allocation for Explicit State Vectors
    std::vector<scalar_t> x(n, 0.0);
    std::vector<scalar_t> x_prev(n, 0.0);
    std::vector<scalar_t> x_bar(n, 0.0);
    std::vector<scalar_t> y(m, 0.0);
    std::vector<scalar_t> Ax(m, 0.0);
    std::vector<scalar_t> Aty(n, 0.0);
    std::vector<scalar_t> rp(m, 0.0);
    std::vector<scalar_t> rd(n, 0.0);

    // Initial primal state: projection of 0 onto [l_c, u_c]
    for (index_t j = 0; j < n; ++j) {
        x[j] = clip(0.0, lp.col_lower[j], lp.col_upper[j]);
        x_prev[j] = x[j];
        x_bar[j] = x[j];
    }

    // 2. Normalization Factors for Relative Residuals
    scalar_t b_norm = 0.0;
    for (index_t i = 0; i < m; ++i) {
        scalar_t b_val = 0.0;
        if (lp.row_lower[i] > -kInf && lp.row_upper[i] < kInf) {
            b_val = 0.5 * (lp.row_lower[i] + lp.row_upper[i]);
        } else if (lp.row_upper[i] < kInf) {
            b_val = lp.row_upper[i];
        } else if (lp.row_lower[i] > -kInf) {
            b_val = lp.row_lower[i];
        }
        b_norm += b_val * b_val;
    }
    b_norm = std::sqrt(b_norm) + 1.0;

    scalar_t c_norm = 0.0;
    for (scalar_t val : lp.c) c_norm += val * val;
    c_norm = std::sqrt(c_norm) + 1.0;

    // 3. Step-Size Initialization
    std::vector<scalar_t> tau(n, 1.0);
    std::vector<scalar_t> sigma(m, 1.0);

    if (config_.step_size_strategy == StepSizeStrategy::CONSTANT) {
        // Approximate ||A||_2 via a 5-iteration power iteration
        std::vector<scalar_t> v(n, 1.0 / std::sqrt(static_cast<double>(n)));
        std::vector<scalar_t> Av(m, 0.0);
        scalar_t s_norm = 1.0;
        for (int iter = 0; iter < 5; ++iter) {
            // Av = A * v
            for (index_t i = 0; i < m; ++i) {
                scalar_t sum = 0.0;
                index_t start = lp.csr_row_ptr[i];
                index_t end = lp.csr_row_ptr[i + 1];
                for (index_t p = start; p < end; ++p) sum += lp.csr_values[p] * v[lp.csr_col_ind[p]];
                Av[i] = sum;
            }
            // v = A^T * Av
            std::fill(v.begin(), v.end(), 0.0);
            for (index_t i = 0; i < m; ++i) {
                scalar_t avi = Av[i];
                if (std::abs(avi) < 1e-15) continue;
                index_t start = lp.csr_row_ptr[i];
                index_t end = lp.csr_row_ptr[i + 1];
                for (index_t p = start; p < end; ++p) v[lp.csr_col_ind[p]] += lp.csr_values[p] * avi;
            }
            scalar_t v_len = 0.0;
            for (scalar_t val : v) v_len += val * val;
            s_norm = std::sqrt(v_len);
            if (s_norm > 1e-12) {
                for (scalar_t& val : v) val /= s_norm;
            }
        }
        scalar_t spectral_norm = std::max(s_norm, static_cast<scalar_t>(1e-6));
        scalar_t base_step = config_.omega / spectral_norm;
        std::fill(tau.begin(), tau.end(), base_step * config_.step_size_ratio);
        std::fill(sigma.begin(), sigma.end(), base_step / config_.step_size_ratio);
    } else {
        // Pock-Chambolle diagonal coordinate preconditioning:
        // d_r(i) = sum_j |A_ij|, d_c(j) = sum_i |A_ij|
        std::vector<scalar_t> col_abs_sum(n, 0.0);
        std::vector<scalar_t> row_abs_sum(m, 0.0);

        for (index_t i = 0; i < m; ++i) {
            index_t start = lp.csr_row_ptr[i];
            index_t end = lp.csr_row_ptr[i + 1];
            for (index_t p = start; p < end; ++p) {
                scalar_t val = std::abs(lp.csr_values[p]);
                row_abs_sum[i] += val;
                col_abs_sum[lp.csr_col_ind[p]] += val;
            }
        }

        for (index_t j = 0; j < n; ++j) {
            if (col_abs_sum[j] > 1e-12) {
                tau[j] = (config_.omega / col_abs_sum[j]) * config_.step_size_ratio;
            } else {
                tau[j] = 1.0 * config_.step_size_ratio;
            }
        }
        for (index_t i = 0; i < m; ++i) {
            if (row_abs_sum[i] > 1e-12) {
                sigma[i] = (config_.omega / row_abs_sum[i]) / config_.step_size_ratio;
            } else {
                sigma[i] = 1.0 / config_.step_size_ratio;
            }
        }
    }

    // 4. Convergence & Tracking Variables
    scalar_t best_comb_res = 1e30;
    int last_restart_iter = 0;
    int restart_count = 0;
    scalar_t theta = config_.theta;

    utils::CPUTimer solve_timer;
    solve_timer.start();

    // 5. Main PDHG Iteration Loop
    for (int iter = 1; iter <= config_.max_iterations; ++iter) {
        result.iterations = iter;

        // --- Step 5a: Dual update via SpMV Ax = A * x_bar ---
        // Ax_i = sum_j A_ij * x_bar_j
        std::fill(Ax.begin(), Ax.end(), 0.0);
        for (index_t i = 0; i < m; ++i) {
            index_t start = lp.csr_row_ptr[i];
            index_t end = lp.csr_row_ptr[i + 1];
            scalar_t sum = 0.0;
            for (index_t p = start; p < end; ++p) {
                sum += lp.csr_values[p] * x_bar[lp.csr_col_ind[p]];
            }
            Ax[i] = sum;

            // Moreau proximal dual projection for range l_r <= Ax <= u_r:
            // v_i = y_i / sigma_i + Ax_i
            // y_i = sigma_i * (v_i - proj_[l_r, u_r](v_i))
            scalar_t v = (y[i] / sigma[i]) + sum;
            scalar_t proj_v = clip(v, lp.row_lower[i], lp.row_upper[i]);
            y[i] = sigma[i] * (v - proj_v);
        }

        // --- Step 5b: Primal update via SpMV Transpose A^T y ---
        std::fill(Aty.begin(), Aty.end(), 0.0);
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

        // --- Step 5c: Primal bound projection & Extrapolation ---
        for (index_t j = 0; j < n; ++j) {
            x_prev[j] = x[j];
            scalar_t step_val = x[j] - tau[j] * (Aty[j] + lp.c[j]);
            x[j] = clip(step_val, lp.col_lower[j], lp.col_upper[j]);
            x_bar[j] = x[j] + theta * (x[j] - x_prev[j]);
        }

        // --- Step 5d: Residual, Termination & Adaptive Evaluation ---
        if (iter % config_.check_interval == 0 || iter == config_.max_iterations) {
            // Primal constraint residual: r_p = A x - proj_[l_r, u_r](A x)
            scalar_t p_res_sq = 0.0;
            for (index_t i = 0; i < m; ++i) {
                scalar_t sum = 0.0;
                index_t start = lp.csr_row_ptr[i];
                index_t end = lp.csr_row_ptr[i + 1];
                for (index_t p = start; p < end; ++p) {
                    sum += lp.csr_values[p] * x[lp.csr_col_ind[p]];
                }
                scalar_t proj_act = clip(sum, lp.row_lower[i], lp.row_upper[i]);
                scalar_t diff = sum - proj_act;
                p_res_sq += diff * diff;
            }
            scalar_t p_res = std::sqrt(p_res_sq) / b_norm;

            // Dual stationarity residual: r_d = x - proj_[l_c, u_c](x - (A^T y + c))
            scalar_t d_res_sq = 0.0;
            for (index_t j = 0; j < n; ++j) {
                scalar_t grad = Aty[j] + lp.c[j];
                scalar_t proj_val = clip(x[j] - grad, lp.col_lower[j], lp.col_upper[j]);
                scalar_t diff = x[j] - proj_val;
                d_res_sq += diff * diff;
            }
            scalar_t d_res = std::sqrt(d_res_sq) / c_norm;

            // Primal objective: c^T x + c_0
            scalar_t p_obj = lp.obj_offset;
            for (index_t j = 0; j < n; ++j) p_obj += lp.c[j] * x[j];

            // Dual objective approximation:
            // D(y) = c_0 + sum_i (y_i > 0 ? y_i * l_r[i] : y_i * u_r[i]) + sum_j (s_j > 0 ? s_j * l_c[j] : s_j * u_c[j])
            scalar_t d_obj = lp.obj_offset;
            for (index_t i = 0; i < m; ++i) {
                if (y[i] > 0.0 && lp.row_lower[i] > -kInf) d_obj += y[i] * lp.row_lower[i];
                else if (y[i] < 0.0 && lp.row_upper[i] < kInf) d_obj += y[i] * lp.row_upper[i];
            }
            for (index_t j = 0; j < n; ++j) {
                scalar_t sj = lp.c[j] + Aty[j];
                if (sj > 0.0 && lp.col_lower[j] > -kInf) d_obj += sj * lp.col_lower[j];
                else if (sj < 0.0 && lp.col_upper[j] < kInf) d_obj += sj * lp.col_upper[j];
            }
            scalar_t gap = std::abs(p_obj - d_obj) / (1.0 + std::abs(p_obj) + std::abs(d_obj));

            result.primal_residual = p_res;
            result.dual_residual = d_res;
            result.primal_objective = p_obj;
            result.dual_objective = d_obj;
            result.duality_gap = gap;

            scalar_t comb_res = std::sqrt(p_res * p_res + d_res * d_res);

            // Optional Iteration Logging
            if (config_.record_history) {
                IterationLog entry;
                entry.iteration = iter;
                entry.elapsed_ms = solve_timer.elapsed_milliseconds();
                entry.objective = p_obj;
                entry.primal_residual = p_res;
                entry.dual_residual = d_res;
                entry.duality_gap = gap;
                entry.step_primal = tau[0];
                entry.step_dual = sigma[0];
                entry.restarted = false;
                result.history.push_back(entry);
            }

            // Check Numerical Failure (NaN / Inf / Divergence)
            if (std::isnan(p_res) || std::isinf(p_res) || p_res > 1e10 ||
                std::isnan(d_res) || std::isinf(d_res) || d_res > 1e10) {
                result.status = TerminationStatus::NUMERICAL_FAILURE;
                break;
            }

            // Check Convergence
            if (p_res < config_.primal_tol && d_res < config_.dual_tol) {
                result.status = TerminationStatus::OPTIMAL;
                break;
            }

            // Check Wall-Clock Time Limit
            if (solve_timer.elapsed_seconds() > config_.time_limit_sec) {
                result.status = TerminationStatus::TIME_LIMIT;
                break;
            }

            // Adaptive Step-Size Adjustment
            if (config_.step_size_strategy == StepSizeStrategy::ADAPTIVE) {
                scalar_t ratio = (d_res > 1e-12) ? (p_res / d_res) : 1.0;
                if (ratio > 2.0) {
                    scalar_t factor = std::min(static_cast<scalar_t>(1.25), std::sqrt(ratio));
                    for (scalar_t& s : sigma) s *= factor;
                    for (scalar_t& t : tau) t /= factor;
                } else if (ratio < 0.5) {
                    scalar_t factor = std::min(static_cast<scalar_t>(1.25), std::sqrt(1.0 / ratio));
                    for (scalar_t& s : sigma) s /= factor;
                    for (scalar_t& t : tau) t *= factor;
                }
            }

            // Adaptive Restart Mechanism
            if (config_.restart_strategy == RestartStrategy::ADAPTIVE) {
                if (iter - last_restart_iter >= config_.restart_interval) {
                    if (comb_res > 0.9 * best_comb_res) {
                        // Progress stalled: restart momentum
                        for (index_t j = 0; j < n; ++j) {
                            x_bar[j] = x[j];
                            x_prev[j] = x[j];
                        }
                        restart_count++;
                        last_restart_iter = iter;
                        if (!result.history.empty()) {
                            result.history.back().restarted = true;
                        }
                    }
                    best_comb_res = std::min(best_comb_res, comb_res);
                }
            }
        }
    }

    solve_timer.stop();
    total_timer.stop();

    result.restarts = restart_count;
    result.timing.pure_solve_ms = solve_timer.elapsed_milliseconds();
    result.timing.total_time_ms = total_timer.elapsed_milliseconds();

    // Store final solution vectors
    result.x = std::move(x);
    result.y = std::move(y);
    result.s.resize(n, 0.0);
    for (index_t j = 0; j < n; ++j) {
        result.s[j] = lp.c[j] + Aty[j];
    }

    return result;
}

} // namespace pipepye::solver
