#pragma once

#include <cuda_runtime.h>
#include <pipepye/core/types.hpp>
#include <cstddef>

namespace pipepye::cuda {

// =============================================================================
// Synchronous Host-Returning Reductions (Copies scalar result to CPU host)
// =============================================================================

/// @brief Computes vector dot product: x^T y = sum(x_i * y_i)
[[nodiscard]] scalar_t cuda_dot(const scalar_t* d_x, const scalar_t* d_y, size_t n, cudaStream_t stream = nullptr);

/// @brief Computes vector Euclidean L2 norm: sqrt(sum(x_i^2))
[[nodiscard]] scalar_t cuda_norm_2(const scalar_t* d_x, size_t n, cudaStream_t stream = nullptr);

/// @brief Computes vector L1 norm: sum(|x_i|)
[[nodiscard]] scalar_t cuda_norm_1(const scalar_t* d_x, size_t n, cudaStream_t stream = nullptr);

/// @brief Computes vector Linf (maximum absolute value) norm: max(|x_i|)
[[nodiscard]] scalar_t cuda_norm_inf(const scalar_t* d_x, size_t n, cudaStream_t stream = nullptr);

/// @brief Computes vector sum: sum(x_i)
[[nodiscard]] scalar_t cuda_sum(const scalar_t* d_x, size_t n, cudaStream_t stream = nullptr);

// =============================================================================
// Asynchronous Device-Resident Reductions (Keeps scalar in GPU VRAM for solver)
// =============================================================================

void launch_cuda_dot(const scalar_t* d_x, const scalar_t* d_y, scalar_t* d_out, size_t n, cudaStream_t stream = nullptr);
void launch_cuda_norm_2(const scalar_t* d_x, scalar_t* d_out, size_t n, cudaStream_t stream = nullptr);
void launch_cuda_norm_1(const scalar_t* d_x, scalar_t* d_out, size_t n, cudaStream_t stream = nullptr);
void launch_cuda_norm_inf(const scalar_t* d_x, scalar_t* d_out, size_t n, cudaStream_t stream = nullptr);
void launch_cuda_sum(const scalar_t* d_x, scalar_t* d_out, size_t n, cudaStream_t stream = nullptr);

} // namespace pipepye::cuda
