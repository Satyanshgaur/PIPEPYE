#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <pipepye/core/types.hpp>

namespace pipepye::cuda::kernels {

/// @brief Launches double-precision AXPY kernel: y[i] = alpha * x[i] + y[i]
/// @param d_x Pointer to device memory input vector x
/// @param d_y Pointer to device memory in-out vector y
/// @param alpha Scaling scalar
/// @param n Number of elements
/// @param stream CUDA stream (defaults to default stream)
void launch_daxpy(const double* d_x, double* d_y, double alpha, size_t n, cudaStream_t stream = nullptr);

/// @brief Launches single-precision AXPY kernel: y[i] = alpha * x[i] + y[i]
void launch_saxpy(const float* d_x, float* d_y, float alpha, size_t n, cudaStream_t stream = nullptr);

} // namespace pipepye::cuda::kernels
