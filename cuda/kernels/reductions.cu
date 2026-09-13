#include "reductions.cuh"
#include <pipepye/cuda/reductions.cuh>
#include <pipepye/cuda/cuda_check.cuh>

#include <cmath>
#include <algorithm>

namespace pipepye::cuda::kernels {

// =============================================================================
// Device Warp/Block Reduction Helpers
// =============================================================================

__device__ inline double warp_reduce_sum(double val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ inline double warp_reduce_max(double val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmax(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ inline double block_reduce_sum(double val) {
    __shared__ double s_warps[8]; // Up to 8 warps per 256-thread block
    int warp_id = threadIdx.x / 32;
    int lane    = threadIdx.x & 31;

    val = warp_reduce_sum(val);

    if (lane == 0) {
        s_warps[warp_id] = val;
    }
    __syncthreads();

    // First warp reduces the warp sums
    double block_sum = (threadIdx.x < 8) ? s_warps[lane] : 0.0;
    if (warp_id == 0) {
        #pragma unroll
        for (int offset = 4; offset > 0; offset >>= 1) {
            block_sum += __shfl_down_sync(0xffffffff, block_sum, offset);
        }
    }
    return block_sum;
}

__device__ inline double block_reduce_max(double val) {
    __shared__ double s_warps[8];
    int warp_id = threadIdx.x / 32;
    int lane    = threadIdx.x & 31;

    val = warp_reduce_max(val);

    if (lane == 0) {
        s_warps[warp_id] = val;
    }
    __syncthreads();

    double block_max = (threadIdx.x < 8) ? s_warps[lane] : 0.0;
    if (warp_id == 0) {
        #pragma unroll
        for (int offset = 4; offset > 0; offset >>= 1) {
            block_max = fmax(block_max, __shfl_down_sync(0xffffffff, block_max, offset));
        }
    }
    return block_max;
}

// =============================================================================
// Reduction Kernels
// =============================================================================

__global__ void dot_kernel(const scalar_t* __restrict__ x, const scalar_t* __restrict__ y, scalar_t* __restrict__ out, size_t n) {
    double local_sum = 0.0;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = idx; i < n; i += stride) {
        local_sum += x[i] * y[i];
    }

    double block_sum = block_reduce_sum(local_sum);
    if (threadIdx.x == 0) {
        atomicAdd(out, block_sum);
    }
}

__global__ void norm2_sq_kernel(const scalar_t* __restrict__ x, scalar_t* __restrict__ out, size_t n) {
    double local_sum = 0.0;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = idx; i < n; i += stride) {
        double v = x[i];
        local_sum += v * v;
    }

    double block_sum = block_reduce_sum(local_sum);
    if (threadIdx.x == 0) {
        atomicAdd(out, block_sum);
    }
}

__global__ void norm1_kernel(const scalar_t* __restrict__ x, scalar_t* __restrict__ out, size_t n) {
    double local_sum = 0.0;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = idx; i < n; i += stride) {
        local_sum += fabs(x[i]);
    }

    double block_sum = block_reduce_sum(local_sum);
    if (threadIdx.x == 0) {
        atomicAdd(out, block_sum);
    }
}

__global__ void sum_kernel(const scalar_t* __restrict__ x, scalar_t* __restrict__ out, size_t n) {
    double local_sum = 0.0;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = idx; i < n; i += stride) {
        local_sum += x[i];
    }

    double block_sum = block_reduce_sum(local_sum);
    if (threadIdx.x == 0) {
        atomicAdd(out, block_sum);
    }
}

__global__ void norm_inf_stage1_kernel(const scalar_t* __restrict__ x, scalar_t* __restrict__ block_maxes, size_t n) {
    double local_max = 0.0;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = idx; i < n; i += stride) {
        local_max = fmax(local_max, fabs(x[i]));
    }

    double b_max = block_reduce_max(local_max);
    if (threadIdx.x == 0) {
        block_maxes[blockIdx.x] = b_max;
    }
}

__global__ void norm_inf_stage2_kernel(const scalar_t* __restrict__ block_maxes, scalar_t* __restrict__ out, size_t num_blocks) {
    double local_max = 0.0;
    for (size_t i = threadIdx.x; i < num_blocks; i += blockDim.x) {
        local_max = fmax(local_max, block_maxes[i]);
    }

    double final_max = block_reduce_max(local_max);
    if (threadIdx.x == 0) {
        *out = final_max;
    }
}

__global__ void sqrt_scalar_kernel(scalar_t* __restrict__ val) {
    if (threadIdx.x == 0) {
        *val = sqrt(*val);
    }
}

void launch_dot_impl(const scalar_t* d_x, const scalar_t* d_y, scalar_t* d_out, size_t n, cudaStream_t stream) {
    CUDA_CHECK(cudaMemsetAsync(d_out, 0, sizeof(scalar_t), stream));
    if (n == 0) return;
    constexpr int block_size = 256;
    int num_blocks = static_cast<int>((n + block_size - 1) / block_size);
    if (num_blocks > 512) num_blocks = 512;
    dot_kernel<<<num_blocks, block_size, 0, stream>>>(d_x, d_y, d_out, n);
    CUDA_CHECK_LAST_ERROR();
}

void launch_norm_2_impl(const scalar_t* d_x, scalar_t* d_out, size_t n, cudaStream_t stream) {
    CUDA_CHECK(cudaMemsetAsync(d_out, 0, sizeof(scalar_t), stream));
    if (n == 0) return;
    constexpr int block_size = 256;
    int num_blocks = static_cast<int>((n + block_size - 1) / block_size);
    if (num_blocks > 512) num_blocks = 512;
    norm2_sq_kernel<<<num_blocks, block_size, 0, stream>>>(d_x, d_out, n);
    sqrt_scalar_kernel<<<1, 1, 0, stream>>>(d_out);
    CUDA_CHECK_LAST_ERROR();
}

void launch_norm_1_impl(const scalar_t* d_x, scalar_t* d_out, size_t n, cudaStream_t stream) {
    CUDA_CHECK(cudaMemsetAsync(d_out, 0, sizeof(scalar_t), stream));
    if (n == 0) return;
    constexpr int block_size = 256;
    int num_blocks = static_cast<int>((n + block_size - 1) / block_size);
    if (num_blocks > 512) num_blocks = 512;
    norm1_kernel<<<num_blocks, block_size, 0, stream>>>(d_x, d_out, n);
    CUDA_CHECK_LAST_ERROR();
}

void launch_norm_inf_impl(const scalar_t* d_x, scalar_t* d_out, size_t n, cudaStream_t stream) {
    if (n == 0) {
        CUDA_CHECK(cudaMemsetAsync(d_out, 0, sizeof(scalar_t), stream));
        return;
    }
    constexpr int block_size = 256;
    int num_blocks = static_cast<int>((n + block_size - 1) / block_size);
    if (num_blocks > 512) num_blocks = 512;

    // Temporary workspace for block maxes
    scalar_t* d_block_maxes = nullptr;
    CUDA_CHECK(cudaMallocAsync(&d_block_maxes, num_blocks * sizeof(scalar_t), stream));

    norm_inf_stage1_kernel<<<num_blocks, block_size, 0, stream>>>(d_x, d_block_maxes, n);
    norm_inf_stage2_kernel<<<1, block_size, 0, stream>>>(d_block_maxes, d_out, num_blocks);
    CUDA_CHECK(cudaFreeAsync(d_block_maxes, stream));
    CUDA_CHECK_LAST_ERROR();
}

void launch_sum_impl(const scalar_t* d_x, scalar_t* d_out, size_t n, cudaStream_t stream) {
    CUDA_CHECK(cudaMemsetAsync(d_out, 0, sizeof(scalar_t), stream));
    if (n == 0) return;
    constexpr int block_size = 256;
    int num_blocks = static_cast<int>((n + block_size - 1) / block_size);
    if (num_blocks > 512) num_blocks = 512;
    sum_kernel<<<num_blocks, block_size, 0, stream>>>(d_x, d_out, n);
    CUDA_CHECK_LAST_ERROR();
}

} // namespace pipepye::cuda::kernels

namespace pipepye::cuda {

void launch_cuda_dot(const scalar_t* d_x, const scalar_t* d_y, scalar_t* d_out, size_t n, cudaStream_t stream) {
    kernels::launch_dot_impl(d_x, d_y, d_out, n, stream);
}

void launch_cuda_norm_2(const scalar_t* d_x, scalar_t* d_out, size_t n, cudaStream_t stream) {
    kernels::launch_norm_2_impl(d_x, d_out, n, stream);
}

void launch_cuda_norm_1(const scalar_t* d_x, scalar_t* d_out, size_t n, cudaStream_t stream) {
    kernels::launch_norm_1_impl(d_x, d_out, n, stream);
}

void launch_cuda_norm_inf(const scalar_t* d_x, scalar_t* d_out, size_t n, cudaStream_t stream) {
    kernels::launch_norm_inf_impl(d_x, d_out, n, stream);
}

void launch_cuda_sum(const scalar_t* d_x, scalar_t* d_out, size_t n, cudaStream_t stream) {
    kernels::launch_sum_impl(d_x, d_out, n, stream);
}

scalar_t cuda_dot(const scalar_t* d_x, const scalar_t* d_y, size_t n, cudaStream_t stream) {
    scalar_t* d_res = nullptr;
    scalar_t h_res = 0.0;
    CUDA_CHECK(cudaMallocAsync(&d_res, sizeof(scalar_t), stream));
    kernels::launch_dot_impl(d_x, d_y, d_res, n, stream);
    CUDA_CHECK(cudaMemcpyAsync(&h_res, d_res, sizeof(scalar_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFreeAsync(d_res, stream));
    return h_res;
}

scalar_t cuda_norm_2(const scalar_t* d_x, size_t n, cudaStream_t stream) {
    scalar_t* d_res = nullptr;
    scalar_t h_res = 0.0;
    CUDA_CHECK(cudaMallocAsync(&d_res, sizeof(scalar_t), stream));
    kernels::launch_norm_2_impl(d_x, d_res, n, stream);
    CUDA_CHECK(cudaMemcpyAsync(&h_res, d_res, sizeof(scalar_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFreeAsync(d_res, stream));
    return h_res;
}

scalar_t cuda_norm_1(const scalar_t* d_x, size_t n, cudaStream_t stream) {
    scalar_t* d_res = nullptr;
    scalar_t h_res = 0.0;
    CUDA_CHECK(cudaMallocAsync(&d_res, sizeof(scalar_t), stream));
    kernels::launch_norm_1_impl(d_x, d_res, n, stream);
    CUDA_CHECK(cudaMemcpyAsync(&h_res, d_res, sizeof(scalar_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFreeAsync(d_res, stream));
    return h_res;
}

scalar_t cuda_norm_inf(const scalar_t* d_x, size_t n, cudaStream_t stream) {
    scalar_t* d_res = nullptr;
    scalar_t h_res = 0.0;
    CUDA_CHECK(cudaMallocAsync(&d_res, sizeof(scalar_t), stream));
    kernels::launch_norm_inf_impl(d_x, d_res, n, stream);
    CUDA_CHECK(cudaMemcpyAsync(&h_res, d_res, sizeof(scalar_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFreeAsync(d_res, stream));
    return h_res;
}

scalar_t cuda_sum(const scalar_t* d_x, size_t n, cudaStream_t stream) {
    scalar_t* d_res = nullptr;
    scalar_t h_res = 0.0;
    CUDA_CHECK(cudaMallocAsync(&d_res, sizeof(scalar_t), stream));
    kernels::launch_sum_impl(d_x, d_res, n, stream);
    CUDA_CHECK(cudaMemcpyAsync(&h_res, d_res, sizeof(scalar_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFreeAsync(d_res, stream));
    return h_res;
}

} // namespace pipepye::cuda
