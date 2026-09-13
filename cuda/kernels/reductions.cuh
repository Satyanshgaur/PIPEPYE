#pragma once

#include <cuda_runtime.h>
#include <pipepye/core/types.hpp>

namespace pipepye::cuda::kernels {

void launch_dot_impl(const scalar_t* d_x, const scalar_t* d_y, scalar_t* d_out, size_t n, cudaStream_t stream = nullptr);
void launch_norm_2_impl(const scalar_t* d_x, scalar_t* d_out, size_t n, cudaStream_t stream = nullptr);
void launch_norm_1_impl(const scalar_t* d_x, scalar_t* d_out, size_t n, cudaStream_t stream = nullptr);
void launch_norm_inf_impl(const scalar_t* d_x, scalar_t* d_out, size_t n, cudaStream_t stream = nullptr);
void launch_sum_impl(const scalar_t* d_x, scalar_t* d_out, size_t n, cudaStream_t stream = nullptr);

} // namespace pipepye::cuda::kernels
