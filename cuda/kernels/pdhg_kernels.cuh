#pragma once

#include <cuda_runtime.h>
#include <pipepye/core/types.hpp>

namespace pipepye::cuda::kernels {

/// @brief Launches dual proximal update kernel on GPU.
/// y_i = sigma_i * (v_i - clip(v_i, l_r[i], u_r[i])), where v_i = y_i / sigma_i + Ax_bar[i]
void launch_dual_step(
    index_t m,
    const scalar_t* d_Ax_bar,
    const scalar_t* d_sigma,
    const scalar_t* d_row_lower,
    const scalar_t* d_row_upper,
    scalar_t* d_y,
    cudaStream_t stream = nullptr);

/// @brief Launches primal update, bound projection, and extrapolation on GPU.
/// x_prev = x, x = clip(x - tau * (Aty + c), l_c, u_c), x_bar = x + theta * (x - x_prev)
void launch_primal_step_extrapolate(
    index_t n,
    const scalar_t* d_Aty,
    const scalar_t* d_c,
    const scalar_t* d_tau,
    const scalar_t* d_col_lower,
    const scalar_t* d_col_upper,
    scalar_t theta,
    scalar_t* d_x,
    scalar_t* d_x_prev,
    scalar_t* d_x_bar,
    cudaStream_t stream = nullptr);

/// @brief Computes primal constraint residual vector: rp_i = Ax_i - clip(Ax_i, l_r[i], u_r[i])
void launch_primal_residual(
    index_t m,
    const scalar_t* d_Ax,
    const scalar_t* d_row_lower,
    const scalar_t* d_row_upper,
    scalar_t* d_rp,
    cudaStream_t stream = nullptr);

/// @brief Computes dual stationarity residual vector: rd_j = x_j - clip(x_j - (Aty_j + c_j), l_c[j], u_c[j])
void launch_dual_residual(
    index_t n,
    const scalar_t* d_x,
    const scalar_t* d_Aty,
    const scalar_t* d_c,
    const scalar_t* d_col_lower,
    const scalar_t* d_col_upper,
    scalar_t* d_rd,
    cudaStream_t stream = nullptr);

/// @brief Scales step sizes: sigma = sigma * factor, tau = tau / factor
void launch_scale_steps(
    index_t m, index_t n,
    scalar_t factor,
    scalar_t* d_sigma,
    scalar_t* d_tau,
    cudaStream_t stream = nullptr);

/// @brief Resets momentum for adaptive restart: x_bar = x, x_prev = x
void launch_restart_momentum(
    index_t n,
    const scalar_t* d_x,
    scalar_t* d_x_bar,
    scalar_t* d_x_prev,
    cudaStream_t stream = nullptr);

} // namespace pipepye::cuda::kernels
