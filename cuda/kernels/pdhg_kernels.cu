#include "pdhg_kernels.cuh"
#include <pipepye/cuda/cuda_check.cuh>

namespace pipepye::cuda::kernels {

namespace {

__device__ constexpr scalar_t kDevInf = 1e15;

__device__ inline scalar_t d_clip(scalar_t val, scalar_t lb, scalar_t ub) noexcept {
    if (lb > -kDevInf && val < lb) return lb;
    if (ub < kDevInf && val > ub) return ub;
    return val;
}

__global__ void dual_step_kernel(
    index_t m,
    const scalar_t* __restrict__ d_Ax_bar,
    const scalar_t* __restrict__ d_sigma,
    const scalar_t* __restrict__ d_row_lower,
    const scalar_t* __restrict__ d_row_upper,
    scalar_t* __restrict__ d_y) {

    index_t i = static_cast<index_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= m) return;

    scalar_t sig = d_sigma[i];
    scalar_t v = (d_y[i] / sig) + d_Ax_bar[i];
    scalar_t proj = d_clip(v, d_row_lower[i], d_row_upper[i]);
    d_y[i] = sig * (v - proj);
}

__global__ void primal_step_extrapolate_kernel(
    index_t n,
    const scalar_t* __restrict__ d_Aty,
    const scalar_t* __restrict__ d_c,
    const scalar_t* __restrict__ d_tau,
    const scalar_t* __restrict__ d_col_lower,
    const scalar_t* __restrict__ d_col_upper,
    scalar_t theta,
    scalar_t* __restrict__ d_x,
    scalar_t* __restrict__ d_x_prev,
    scalar_t* __restrict__ d_x_bar) {

    index_t j = static_cast<index_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (j >= n) return;

    scalar_t xj = d_x[j];
    d_x_prev[j] = xj;

    scalar_t w = xj - d_tau[j] * (d_Aty[j] + d_c[j]);
    scalar_t x_new = d_clip(w, d_col_lower[j], d_col_upper[j]);
    d_x[j] = x_new;
    d_x_bar[j] = x_new + theta * (x_new - xj);
}

__global__ void primal_residual_kernel(
    index_t m,
    const scalar_t* __restrict__ d_Ax,
    const scalar_t* __restrict__ d_row_lower,
    const scalar_t* __restrict__ d_row_upper,
    scalar_t* __restrict__ d_rp) {

    index_t i = static_cast<index_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= m) return;

    scalar_t axi = d_Ax[i];
    scalar_t proj = d_clip(axi, d_row_lower[i], d_row_upper[i]);
    d_rp[i] = axi - proj;
}

__global__ void dual_residual_kernel(
    index_t n,
    const scalar_t* __restrict__ d_x,
    const scalar_t* __restrict__ d_Aty,
    const scalar_t* __restrict__ d_c,
    const scalar_t* __restrict__ d_col_lower,
    const scalar_t* __restrict__ d_col_upper,
    scalar_t* __restrict__ d_rd) {

    index_t j = static_cast<index_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (j >= n) return;

    scalar_t grad = d_Aty[j] + d_c[j];
    scalar_t xj = d_x[j];
    scalar_t proj = d_clip(xj - grad, d_col_lower[j], d_col_upper[j]);
    d_rd[j] = xj - proj;
}

__global__ void scale_steps_kernel(
    index_t m, index_t n,
    scalar_t factor,
    scalar_t* __restrict__ d_sigma,
    scalar_t* __restrict__ d_tau) {

    index_t idx = static_cast<index_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx < m) {
        d_sigma[idx] *= factor;
    }
    if (idx < n) {
        d_tau[idx] /= factor;
    }
}

__global__ void restart_momentum_kernel(
    index_t n,
    const scalar_t* __restrict__ d_x,
    scalar_t* __restrict__ d_x_bar,
    scalar_t* __restrict__ d_x_prev) {

    index_t j = static_cast<index_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (j >= n) return;

    scalar_t xj = d_x[j];
    d_x_bar[j] = xj;
    d_x_prev[j] = xj;
}

} // namespace

void launch_dual_step(
    index_t m,
    const scalar_t* d_Ax_bar,
    const scalar_t* d_sigma,
    const scalar_t* d_row_lower,
    const scalar_t* d_row_upper,
    scalar_t* d_y,
    cudaStream_t stream) {

    if (m <= 0) return;
    constexpr int kBlock = 256;
    int grid = static_cast<int>((m + kBlock - 1) / kBlock);
    dual_step_kernel<<<grid, kBlock, 0, stream>>>(m, d_Ax_bar, d_sigma, d_row_lower, d_row_upper, d_y);
    CUDA_CHECK_LAST_ERROR();
}

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
    cudaStream_t stream) {

    if (n <= 0) return;
    constexpr int kBlock = 256;
    int grid = static_cast<int>((n + kBlock - 1) / kBlock);
    primal_step_extrapolate_kernel<<<grid, kBlock, 0, stream>>>(
        n, d_Aty, d_c, d_tau, d_col_lower, d_col_upper, theta, d_x, d_x_prev, d_x_bar);
    CUDA_CHECK_LAST_ERROR();
}

void launch_primal_residual(
    index_t m,
    const scalar_t* d_Ax,
    const scalar_t* d_row_lower,
    const scalar_t* d_row_upper,
    scalar_t* d_rp,
    cudaStream_t stream) {

    if (m <= 0) return;
    constexpr int kBlock = 256;
    int grid = static_cast<int>((m + kBlock - 1) / kBlock);
    primal_residual_kernel<<<grid, kBlock, 0, stream>>>(m, d_Ax, d_row_lower, d_row_upper, d_rp);
    CUDA_CHECK_LAST_ERROR();
}

void launch_dual_residual(
    index_t n,
    const scalar_t* d_x,
    const scalar_t* d_Aty,
    const scalar_t* d_c,
    const scalar_t* d_col_lower,
    const scalar_t* d_col_upper,
    scalar_t* d_rd,
    cudaStream_t stream) {

    if (n <= 0) return;
    constexpr int kBlock = 256;
    int grid = static_cast<int>((n + kBlock - 1) / kBlock);
    dual_residual_kernel<<<grid, kBlock, 0, stream>>>(n, d_x, d_Aty, d_c, d_col_lower, d_col_upper, d_rd);
    CUDA_CHECK_LAST_ERROR();
}

void launch_scale_steps(
    index_t m, index_t n,
    scalar_t factor,
    scalar_t* d_sigma,
    scalar_t* d_tau,
    cudaStream_t stream) {

    index_t total = std::max(m, n);
    if (total <= 0) return;
    constexpr int kBlock = 256;
    int grid = static_cast<int>((total + kBlock - 1) / kBlock);
    scale_steps_kernel<<<grid, kBlock, 0, stream>>>(m, n, factor, d_sigma, d_tau);
    CUDA_CHECK_LAST_ERROR();
}

void launch_restart_momentum(
    index_t n,
    const scalar_t* d_x,
    scalar_t* d_x_bar,
    scalar_t* d_x_prev,
    cudaStream_t stream) {

    if (n <= 0) return;
    constexpr int kBlock = 256;
    int grid = static_cast<int>((n + kBlock - 1) / kBlock);
    restart_momentum_kernel<<<grid, kBlock, 0, stream>>>(n, d_x, d_x_bar, d_x_prev);
    CUDA_CHECK_LAST_ERROR();
}

} // namespace pipepye::cuda::kernels
