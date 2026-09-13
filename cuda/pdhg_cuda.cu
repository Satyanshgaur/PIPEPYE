#include <pipepye/solver/pdhg_solver_cuda.hpp>
#include <pipepye/solver/pdhg_solver.hpp>
#include <pipepye/cuda/spmv.cuh>
#include <pipepye/cuda/reductions.cuh>
#include <pipepye/cuda/cuda_check.cuh>
#include "kernels/pdhg_kernels.cuh"
#include <pipepye/utils/timer.hpp>
#include <cmath>
#include <algorithm>
#include <iostream>

namespace pipepye::solver {

void init_cuda_solver() {
    PDHGSolver::register_cuda_solver_factory([](const model::LinearProgram& lp, const SolverConfig& config) {
        CudaPDPOptimizer opt(config);
        return opt.solve(lp);
    });
}

namespace {

struct AutoRegisterCudaSolver {
    AutoRegisterCudaSolver() {
        init_cuda_solver();
    }
};
static AutoRegisterCudaSolver g_auto_register;

const scalar_t kInf = 1e15;

inline scalar_t host_clip(scalar_t val, scalar_t lb, scalar_t ub) noexcept {
    if (lb > -kInf && val < lb) return lb;
    if (ub < kInf && val > ub) return ub;
    return val;
}

} // namespace

SolverResult CudaPDPOptimizer::solve(const model::LinearProgram& lp) const {
    SolverResult result;
    index_t m = lp.num_rows();
    index_t n = lp.num_cols();

    utils::CPUTimer total_timer;
    total_timer.start();

    // 0. Handle 0-dimension edge cases on host
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

    // 1. Host Precomputations: Normalization Constants & Initial Step Sizes
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

    // Initial primal state on host
    std::vector<scalar_t> h_x(n, 0.0);
    for (index_t j = 0; j < n; ++j) {
        h_x[j] = host_clip(0.0, lp.col_lower[j], lp.col_upper[j]);
    }

    // Pock-Chambolle step-sizes
    std::vector<scalar_t> h_tau(n, 1.0);
    std::vector<scalar_t> h_sigma(m, 1.0);

    if (config_.step_size_strategy == StepSizeStrategy::CONSTANT) {
        // Scalar step based on spectral proxy estimate
        scalar_t max_row_norm = 1.0;
        for (index_t i = 0; i < m; ++i) {
            scalar_t sum = 0.0;
            index_t start = lp.csr_row_ptr[i];
            index_t end = lp.csr_row_ptr[i + 1];
            for (index_t p = start; p < end; ++p) sum += std::abs(lp.csr_values[p]);
            max_row_norm = std::max(max_row_norm, sum);
        }
        scalar_t base_step = config_.omega / max_row_norm;
        std::fill(h_tau.begin(), h_tau.end(), base_step * config_.step_size_ratio);
        std::fill(h_sigma.begin(), h_sigma.end(), base_step / config_.step_size_ratio);
    } else {
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
            h_tau[j] = (col_abs_sum[j] > 1e-12) ? ((config_.omega / col_abs_sum[j]) * config_.step_size_ratio)
                                                : (1.0 * config_.step_size_ratio);
        }
        for (index_t i = 0; i < m; ++i) {
            h_sigma[i] = (row_abs_sum[i] > 1e-12) ? ((config_.omega / row_abs_sum[i]) / config_.step_size_ratio)
                                                  : (1.0 / config_.step_size_ratio);
        }
    }

    // 2. Measure Host-to-Device (H2D) Transfers
    utils::CPUTimer h2d_timer;
    h2d_timer.start();

    // Construct A (CSR) and A^T (CSC of A converted to CSR of A^T)
    sparse::CSRMatrix host_csr(m, n, lp.csr_row_ptr, lp.csr_col_ind, lp.csr_values);
    cuda::DeviceCSRMatrix d_A(host_csr);

    // If CSC arrays are available, construct A^T as a DeviceCSRMatrix of dimensions n x m
    sparse::CSRMatrix host_csr_t;
    if (!lp.csc_col_ptr.empty() && static_cast<index_t>(lp.csc_col_ptr.size()) == n + 1) {
        host_csr_t = sparse::CSRMatrix(n, m, lp.csc_col_ptr, lp.csc_row_ind, lp.csc_values);
    } else {
        auto csc = host_csr.to_csc();
        std::vector<index_t> at_ptr(csc.col_ptr().begin(), csc.col_ptr().end());
        std::vector<index_t> at_ind(csc.row_ind().begin(), csc.row_ind().end());
        std::vector<scalar_t> at_val(csc.values().begin(), csc.values().end());
        host_csr_t = sparse::CSRMatrix(n, m, std::move(at_ptr), std::move(at_ind), std::move(at_val));
    }
    cuda::DeviceCSRMatrix d_At(host_csr_t);

    // Allocate and transfer device state vectors
    cuda::DeviceVector d_x(sparse::ConstVectorView(h_x.data(), n));
    cuda::DeviceVector d_x_prev(sparse::ConstVectorView(h_x.data(), n));
    cuda::DeviceVector d_x_bar(sparse::ConstVectorView(h_x.data(), n));

    cuda::DeviceVector d_y(m);
    d_y.set_zero();

    cuda::DeviceVector d_c(sparse::ConstVectorView(lp.c.data(), n));
    cuda::DeviceVector d_col_lower(sparse::ConstVectorView(lp.col_lower.data(), n));
    cuda::DeviceVector d_col_upper(sparse::ConstVectorView(lp.col_upper.data(), n));
    cuda::DeviceVector d_row_lower(sparse::ConstVectorView(lp.row_lower.data(), m));
    cuda::DeviceVector d_row_upper(sparse::ConstVectorView(lp.row_upper.data(), m));

    cuda::DeviceVector d_tau(sparse::ConstVectorView(h_tau.data(), n));
    cuda::DeviceVector d_sigma(sparse::ConstVectorView(h_sigma.data(), m));

    cuda::DeviceVector d_Ax(m);
    cuda::DeviceVector d_Aty(n);
    cuda::DeviceVector d_rp(m);
    cuda::DeviceVector d_rd(n);

    h2d_timer.stop();
    result.timing.h2d_transfer_ms = h2d_timer.elapsed_milliseconds();

    // 3. Resident GPU Iteration Loop
    utils::CPUTimer solve_timer;
    solve_timer.start();

    cuda::SpMVKernelVariant spmv_variant = cuda::SpMVKernelVariant::Vector;
    if (lp.num_nonzeros() > 100000) {
        spmv_variant = cuda::SpMVKernelVariant::Adaptive;
    }

    scalar_t best_comb_res = 1e30;
    int last_restart_iter = 0;
    int restart_count = 0;
    scalar_t theta = config_.theta;

    for (int iter = 1; iter <= config_.max_iterations; ++iter) {
        result.iterations = iter;

        // Step 3a: SpMV d_Ax = A * d_x_bar
        d_A.spmv(1.0, d_x_bar, 0.0, d_Ax, spmv_variant);

        // Step 3b: Dual update kernel on GPU
        cuda::kernels::launch_dual_step(
            m, d_Ax.data(), d_sigma.data(), d_row_lower.data(), d_row_upper.data(), d_y.data());

        // Step 3c: Transposed SpMV d_Aty = A^T * d_y (via d_At SpMV)
        d_At.spmv(1.0, d_y, 0.0, d_Aty, spmv_variant);

        // Step 3d: Primal update, bound projection, and extrapolation kernel on GPU
        cuda::kernels::launch_primal_step_extrapolate(
            n, d_Aty.data(), d_c.data(), d_tau.data(), d_col_lower.data(), d_col_upper.data(),
            theta, d_x.data(), d_x_prev.data(), d_x_bar.data());

        // Step 3e: Periodic Convergence & Residuals Evaluation
        if (iter % config_.check_interval == 0 || iter == config_.max_iterations) {
            // Primal residual on true iterate d_x
            d_A.spmv(1.0, d_x, 0.0, d_Ax, spmv_variant);
            cuda::kernels::launch_primal_residual(
                m, d_Ax.data(), d_row_lower.data(), d_row_upper.data(), d_rp.data());

            // Dual residual on true iterate d_x
            cuda::kernels::launch_dual_residual(
                n, d_x.data(), d_Aty.data(), d_c.data(), d_col_lower.data(), d_col_upper.data(), d_rd.data());

            // Compute Euclidean norms on device (only scalar floats returned to CPU)
            scalar_t p_res_raw = cuda::cuda_norm_2(d_rp.data(), m);
            scalar_t d_res_raw = cuda::cuda_norm_2(d_rd.data(), n);

            scalar_t p_res = p_res_raw / b_norm;
            scalar_t d_res = d_res_raw / c_norm;

            // Primal objective: c^T x + c_0
            scalar_t cx = cuda::cuda_dot(d_c.data(), d_x.data(), n);
            scalar_t p_obj = cx + lp.obj_offset;

            result.primal_residual = p_res;
            result.dual_residual = d_res;
            result.primal_objective = p_obj;

            scalar_t comb_res = std::sqrt(p_res * p_res + d_res * d_res);

            if (config_.record_history) {
                IterationLog entry;
                entry.iteration = iter;
                entry.elapsed_ms = solve_timer.elapsed_milliseconds();
                entry.objective = p_obj;
                entry.primal_residual = p_res;
                entry.dual_residual = d_res;
                entry.step_primal = h_tau[0];
                entry.step_dual = h_sigma[0];
                entry.restarted = false;
                result.history.push_back(entry);
            }

            // Numerical divergence check
            if (std::isnan(p_res) || std::isinf(p_res) || p_res > 1e10 ||
                std::isnan(d_res) || std::isinf(d_res) || d_res > 1e10) {
                result.status = TerminationStatus::NUMERICAL_FAILURE;
                break;
            }

            // Convergence check
            if (p_res < config_.primal_tol && d_res < config_.dual_tol) {
                result.status = TerminationStatus::OPTIMAL;
                break;
            }

            // Time limit check
            if (solve_timer.elapsed_seconds() > config_.time_limit_sec) {
                result.status = TerminationStatus::TIME_LIMIT;
                break;
            }

            // Adaptive step size adjustment
            if (config_.step_size_strategy == StepSizeStrategy::ADAPTIVE) {
                scalar_t ratio = (d_res > 1e-12) ? (p_res / d_res) : 1.0;
                if (ratio > 2.0) {
                    scalar_t factor = std::min(static_cast<scalar_t>(1.25), std::sqrt(ratio));
                    cuda::kernels::launch_scale_steps(m, n, factor, d_sigma.data(), d_tau.data());
                } else if (ratio < 0.5) {
                    scalar_t factor = std::min(static_cast<scalar_t>(1.25), std::sqrt(1.0 / ratio));
                    cuda::kernels::launch_scale_steps(m, n, 1.0 / factor, d_sigma.data(), d_tau.data());
                }
            }

            // Adaptive restart
            if (config_.restart_strategy == RestartStrategy::ADAPTIVE) {
                if (iter - last_restart_iter >= config_.restart_interval) {
                    if (comb_res > 0.9 * best_comb_res) {
                        cuda::kernels::launch_restart_momentum(n, d_x.data(), d_x_bar.data(), d_x_prev.data());
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
    result.restarts = restart_count;
    result.timing.pure_solve_ms = solve_timer.elapsed_milliseconds();

    // 4. Measure Device-to-Host (D2H) Transfer
    utils::CPUTimer d2h_timer;
    d2h_timer.start();

    result.x.resize(n);
    result.y.resize(m);
    result.s.resize(n);

    d_x.copy_to_host(sparse::MutableVectorView(result.x.data(), n));
    d_y.copy_to_host(sparse::MutableVectorView(result.y.data(), m));

    d2h_timer.stop();
    total_timer.stop();

    result.timing.d2h_transfer_ms = d2h_timer.elapsed_milliseconds();
    result.timing.total_time_ms = total_timer.elapsed_milliseconds();

    // Compute slacks s = c + A^T y on host
    std::vector<scalar_t> host_Aty(n, 0.0);
    if (!lp.csc_col_ptr.empty() && static_cast<index_t>(lp.csc_col_ptr.size()) == n + 1) {
        for (index_t j = 0; j < n; ++j) {
            index_t start = lp.csc_col_ptr[j];
            index_t end = lp.csc_col_ptr[j + 1];
            scalar_t sum = 0.0;
            for (index_t p = start; p < end; ++p) {
                sum += lp.csc_values[p] * result.y[lp.csc_row_ind[p]];
            }
            result.s[j] = lp.c[j] + sum;
        }
    } else {
        for (index_t j = 0; j < n; ++j) result.s[j] = lp.c[j];
    }

    return result;
}

} // namespace pipepye::solver
