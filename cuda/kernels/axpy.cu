#include "axpy.cuh"
#include <pipepye/cuda/cuda_check.cuh>

namespace pipepye::cuda::kernels {

__global__ void daxpy_kernel(const double* __restrict__ x, double* __restrict__ y, double alpha, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (size_t i = idx; i < n; i += stride) {
        y[i] = alpha * x[i] + y[i];
    }
}

__global__ void saxpy_kernel(const float* __restrict__ x, float* __restrict__ y, float alpha, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (size_t i = idx; i < n; i += stride) {
        y[i] = alpha * x[i] + y[i];
    }
}

void launch_daxpy(const double* d_x, double* d_y, double alpha, size_t n, cudaStream_t stream) {
    if (n == 0) return;
    constexpr int block_size = 256;
    int num_blocks = static_cast<int>((n + block_size - 1) / block_size);
    // Cap blocks to maximum reasonable grid dimension to utilize grid-stride loop efficiently
    if (num_blocks > 65535) {
        num_blocks = 65535;
    }
    daxpy_kernel<<<num_blocks, block_size, 0, stream>>>(d_x, d_y, alpha, n);
    CUDA_CHECK_LAST_ERROR();
}

void launch_saxpy(const float* d_x, float* d_y, float alpha, size_t n, cudaStream_t stream) {
    if (n == 0) return;
    constexpr int block_size = 256;
    int num_blocks = static_cast<int>((n + block_size - 1) / block_size);
    if (num_blocks > 65535) {
        num_blocks = 65535;
    }
    saxpy_kernel<<<num_blocks, block_size, 0, stream>>>(d_x, d_y, alpha, n);
    CUDA_CHECK_LAST_ERROR();
}

} // namespace pipepye::cuda::kernels
