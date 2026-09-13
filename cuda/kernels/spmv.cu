#include "spmv.cuh"
#include <pipepye/cuda/spmv.cuh>
#include <pipepye/cuda/cuda_check.cuh>

#include <stdexcept>
#include <string>
#include <algorithm>

namespace pipepye::cuda {

std::string to_string(SpMVKernelVariant variant) {
    switch (variant) {
        case SpMVKernelVariant::Scalar:   return "Scalar (1 thread/row)";
        case SpMVKernelVariant::Vector:   return "Vector (1 warp/row)";
        case SpMVKernelVariant::Adaptive: return "Adaptive (Sub-warp 8)";
        case SpMVKernelVariant::Balanced: return "Balanced (Work-partitioned)";
        default: return "Unknown";
    }
}

// =============================================================================
// DeviceVector RAII Implementation
// =============================================================================

DeviceVector::DeviceVector() = default;

DeviceVector::DeviceVector(size_t size, cudaStream_t stream) : size_(size) {
    if (size_ > 0) {
        CUDA_CHECK(cudaMalloc(&d_data_, size_ * sizeof(scalar_t)));
        CUDA_CHECK(cudaMemsetAsync(d_data_, 0, size_ * sizeof(scalar_t), stream));
    }
}

DeviceVector::DeviceVector(sparse::ConstVectorView host_view, cudaStream_t stream)
    : size_(host_view.size()) {
    if (size_ > 0) {
        CUDA_CHECK(cudaMalloc(&d_data_, size_ * sizeof(scalar_t)));
        CUDA_CHECK(cudaMemcpyAsync(d_data_, host_view.data(), size_ * sizeof(scalar_t),
                                   cudaMemcpyHostToDevice, stream));
    }
}

DeviceVector::~DeviceVector() {
    if (d_data_ != nullptr) {
        cudaFree(d_data_);
        d_data_ = nullptr;
    }
    size_ = 0;
}

DeviceVector::DeviceVector(DeviceVector&& other) noexcept
    : d_data_(other.d_data_), size_(other.size_) {
    other.d_data_ = nullptr;
    other.size_ = 0;
}

DeviceVector& DeviceVector::operator=(DeviceVector&& other) noexcept {
    if (this != &other) {
        if (d_data_ != nullptr) {
            cudaFree(d_data_);
        }
        d_data_ = other.d_data_;
        size_ = other.size_;
        other.d_data_ = nullptr;
        other.size_ = 0;
    }
    return *this;
}

void DeviceVector::copy_from_host(sparse::ConstVectorView host_view, cudaStream_t stream) {
    if (static_cast<size_t>(host_view.size()) != size_) {
        throw std::invalid_argument("DeviceVector::copy_from_host size mismatch");
    }
    if (size_ > 0) {
        CUDA_CHECK(cudaMemcpyAsync(d_data_, host_view.data(), size_ * sizeof(scalar_t),
                                   cudaMemcpyHostToDevice, stream));
    }
}

void DeviceVector::copy_to_host(sparse::MutableVectorView host_view, cudaStream_t stream) const {
    if (static_cast<size_t>(host_view.size()) != size_) {
        throw std::invalid_argument("DeviceVector::copy_to_host size mismatch");
    }
    if (size_ > 0) {
        CUDA_CHECK(cudaMemcpyAsync(host_view.data(), d_data_, size_ * sizeof(scalar_t),
                                   cudaMemcpyDeviceToHost, stream));
    }
}

void DeviceVector::set_zero(cudaStream_t stream) {
    if (size_ > 0) {
        CUDA_CHECK(cudaMemsetAsync(d_data_, 0, size_ * sizeof(scalar_t), stream));
    }
}

__global__ void fill_kernel(scalar_t* __restrict__ data, scalar_t value, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (size_t i = idx; i < n; i += stride) {
        data[i] = value;
    }
}

void DeviceVector::fill(scalar_t value, cudaStream_t stream) {
    if (size_ == 0) return;
    constexpr int block_size = 256;
    int num_blocks = static_cast<int>((size_ + block_size - 1) / block_size);
    if (num_blocks > 65535) num_blocks = 65535;
    fill_kernel<<<num_blocks, block_size, 0, stream>>>(d_data_, value, size_);
    CUDA_CHECK_LAST_ERROR();
}

// =============================================================================
// DeviceCSRMatrix RAII Implementation
// =============================================================================

DeviceCSRMatrix::DeviceCSRMatrix() = default;

DeviceCSRMatrix::DeviceCSRMatrix(const sparse::CSRMatrix& host_csr, cudaStream_t stream)
    : num_rows_(host_csr.num_rows()),
      num_cols_(host_csr.num_cols()),
      nnz_(host_csr.num_nonzeros()) {
    
    size_t row_ptr_bytes = (num_rows_ + 1) * sizeof(index_t);
    size_t col_ind_bytes = nnz_ * sizeof(index_t);
    size_t values_bytes  = nnz_ * sizeof(scalar_t);

    CUDA_CHECK(cudaMalloc(&d_row_ptr_, row_ptr_bytes));
    CUDA_CHECK(cudaMemcpyAsync(d_row_ptr_, host_csr.row_ptr().data(), row_ptr_bytes,
                               cudaMemcpyHostToDevice, stream));

    if (nnz_ > 0) {
        CUDA_CHECK(cudaMalloc(&d_col_ind_, col_ind_bytes));
        CUDA_CHECK(cudaMalloc(&d_values_, values_bytes));

        CUDA_CHECK(cudaMemcpyAsync(d_col_ind_, host_csr.col_ind().data(), col_ind_bytes,
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_values_, host_csr.values().data(), values_bytes,
                                   cudaMemcpyHostToDevice, stream));
    }
}

DeviceCSRMatrix::~DeviceCSRMatrix() {
    if (d_row_ptr_ != nullptr) { cudaFree(d_row_ptr_); d_row_ptr_ = nullptr; }
    if (d_col_ind_ != nullptr) { cudaFree(d_col_ind_); d_col_ind_ = nullptr; }
    if (d_values_ != nullptr)  { cudaFree(d_values_);  d_values_ = nullptr; }
    num_rows_ = 0;
    num_cols_ = 0;
    nnz_ = 0;
}

DeviceCSRMatrix::DeviceCSRMatrix(DeviceCSRMatrix&& other) noexcept
    : num_rows_(other.num_rows_),
      num_cols_(other.num_cols_),
      nnz_(other.nnz_),
      d_row_ptr_(other.d_row_ptr_),
      d_col_ind_(other.d_col_ind_),
      d_values_(other.d_values_) {
    other.d_row_ptr_ = nullptr;
    other.d_col_ind_ = nullptr;
    other.d_values_ = nullptr;
    other.num_rows_ = 0;
    other.num_cols_ = 0;
    other.nnz_ = 0;
}

DeviceCSRMatrix& DeviceCSRMatrix::operator=(DeviceCSRMatrix&& other) noexcept {
    if (this != &other) {
        if (d_row_ptr_ != nullptr) cudaFree(d_row_ptr_);
        if (d_col_ind_ != nullptr) cudaFree(d_col_ind_);
        if (d_values_ != nullptr)  cudaFree(d_values_);

        num_rows_ = other.num_rows_;
        num_cols_ = other.num_cols_;
        nnz_ = other.nnz_;
        d_row_ptr_ = other.d_row_ptr_;
        d_col_ind_ = other.d_col_ind_;
        d_values_ = other.d_values_;

        other.d_row_ptr_ = nullptr;
        other.d_col_ind_ = nullptr;
        other.d_values_ = nullptr;
        other.num_rows_ = 0;
        other.num_cols_ = 0;
        other.nnz_ = 0;
    }
    return *this;
}

void DeviceCSRMatrix::spmv(scalar_t alpha, const DeviceVector& x, scalar_t beta, DeviceVector& y,
                           SpMVKernelVariant variant, cudaStream_t stream) const {
    if (x.size() != static_cast<size_t>(num_cols_)) {
        throw std::invalid_argument("DeviceCSRMatrix::spmv x size mismatch: expected " +
                                    std::to_string(num_cols_) + ", got " + std::to_string(x.size()));
    }
    if (y.size() != static_cast<size_t>(num_rows_)) {
        throw std::invalid_argument("DeviceCSRMatrix::spmv y size mismatch: expected " +
                                    std::to_string(num_rows_) + ", got " + std::to_string(y.size()));
    }
    spmv_raw(alpha, x.data(), beta, y.data(), variant, stream);
}

void DeviceCSRMatrix::spmv_raw(scalar_t alpha, const scalar_t* d_x, scalar_t beta, scalar_t* d_y,
                               SpMVKernelVariant variant, cudaStream_t stream) const {
    if (num_rows_ == 0 || num_cols_ == 0) return;

    switch (variant) {
        case SpMVKernelVariant::Scalar:
            launch_spmv_csr_scalar(alpha, num_rows_, num_cols_,
                                   d_row_ptr_, d_col_ind_, d_values_,
                                   d_x, beta, d_y, stream);
            break;
        case SpMVKernelVariant::Vector:
            launch_spmv_csr_vector(alpha, num_rows_, num_cols_,
                                   d_row_ptr_, d_col_ind_, d_values_,
                                   d_x, beta, d_y, stream);
            break;
        case SpMVKernelVariant::Adaptive:
            launch_spmv_csr_adaptive(alpha, num_rows_, num_cols_,
                                     d_row_ptr_, d_col_ind_, d_values_,
                                     d_x, beta, d_y, stream);
            break;
        case SpMVKernelVariant::Balanced:
            launch_spmv_csr_balanced(alpha, num_rows_, num_cols_, nnz_,
                                     d_row_ptr_, d_col_ind_, d_values_,
                                     d_x, beta, d_y, stream);
            break;
    }
}

} // namespace pipepye::cuda

// =============================================================================
// Kernel Implementations (Device Code)
// =============================================================================

namespace pipepye::cuda::kernels {

// 1. Scalar Kernel: 1 thread per row
__global__ void spmv_csr_scalar_kernel(
    scalar_t alpha,
    index_t num_rows,
    const index_t* __restrict__ row_ptr,
    const index_t* __restrict__ col_ind,
    const scalar_t* __restrict__ values,
    const scalar_t* __restrict__ x,
    scalar_t beta,
    scalar_t* __restrict__ y)
{
    index_t row = blockIdx.x * blockDim.x + threadIdx.x;
    index_t stride = blockDim.x * gridDim.x;

    for (; row < num_rows; row += stride) {
        index_t start = row_ptr[row];
        index_t end = row_ptr[row + 1];
        scalar_t sum = 0.0;

        for (index_t k = start; k < end; ++k) {
            sum += values[k] * x[col_ind[k]];
        }

        if (beta == 0.0) {
            y[row] = alpha * sum;
        } else {
            y[row] = alpha * sum + beta * y[row];
        }
    }
}

// 2. Vector Kernel: 1 warp (32 threads) per row
__global__ void spmv_csr_vector_kernel(
    scalar_t alpha,
    index_t num_rows,
    const index_t* __restrict__ row_ptr,
    const index_t* __restrict__ col_ind,
    const scalar_t* __restrict__ values,
    const scalar_t* __restrict__ x,
    scalar_t beta,
    scalar_t* __restrict__ y)
{
    int global_warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int total_warps = (gridDim.x * blockDim.x) / 32;
    int lane = threadIdx.x & 31;

    for (index_t row = global_warp_id; row < num_rows; row += total_warps) {
        index_t start = row_ptr[row];
        index_t end = row_ptr[row + 1];
        scalar_t sum = 0.0;

        for (index_t k = start + lane; k < end; k += 32) {
            sum += values[k] * x[col_ind[k]];
        }

        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }

        if (lane == 0) {
            if (beta == 0.0) {
                y[row] = alpha * sum;
            } else {
                y[row] = alpha * sum + beta * y[row];
            }
        }
    }
}

// 3. Adaptive Kernel: Sub-warp 8 threads per row
__global__ void spmv_csr_adaptive_subwarp8_kernel(
    scalar_t alpha,
    index_t num_rows,
    const index_t* __restrict__ row_ptr,
    const index_t* __restrict__ col_ind,
    const scalar_t* __restrict__ values,
    const scalar_t* __restrict__ x,
    scalar_t beta,
    scalar_t* __restrict__ y)
{
    constexpr int SUBWARP_SIZE = 8;
    int global_subwarp_id = (blockIdx.x * blockDim.x + threadIdx.x) / SUBWARP_SIZE;
    int total_subwarps = (gridDim.x * blockDim.x) / SUBWARP_SIZE;
    int lane = threadIdx.x & (SUBWARP_SIZE - 1);

    for (index_t row = global_subwarp_id; row < num_rows; row += total_subwarps) {
        index_t start = row_ptr[row];
        index_t end = row_ptr[row + 1];
        scalar_t sum = 0.0;

        for (index_t k = start + lane; k < end; k += SUBWARP_SIZE) {
            sum += values[k] * x[col_ind[k]];
        }

        #pragma unroll
        for (int offset = SUBWARP_SIZE / 2; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }

        if (lane == 0) {
            if (beta == 0.0) {
                y[row] = alpha * sum;
            } else {
                y[row] = alpha * sum + beta * y[row];
            }
        }
    }
}

// Helper: Binary search for row given nonzero index k
__device__ inline index_t binary_search_row(const index_t* __restrict__ row_ptr, size_t k, index_t num_rows) {
    index_t low = 0;
    index_t high = num_rows - 1;
    index_t ans = 0;

    while (low <= high) {
        index_t mid = low + (high - low) / 2;
        if (static_cast<size_t>(row_ptr[mid + 1]) > k) {
            ans = mid;
            high = mid - 1;
        } else {
            low = mid + 1;
        }
    }
    return ans;
}

// 4. Balanced Kernel: Partitions NNZ equally across threads
__global__ void spmv_csr_balanced_kernel(
    scalar_t alpha,
    index_t num_rows,
    size_t nnz,
    const index_t* __restrict__ row_ptr,
    const index_t* __restrict__ col_ind,
    const scalar_t* __restrict__ values,
    const scalar_t* __restrict__ x,
    scalar_t* __restrict__ y)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_threads = gridDim.x * blockDim.x;

    size_t k_start = (tid * nnz) / total_threads;
    size_t k_end   = ((tid + 1) * nnz) / total_threads;

    if (k_start >= k_end) return;

    index_t current_row = binary_search_row(row_ptr, k_start, num_rows);
    scalar_t row_sum = 0.0;

    for (size_t k = k_start; k < k_end; ++k) {
        while (k >= static_cast<size_t>(row_ptr[current_row + 1])) {
            // Write partial or full sum for completed row
            if (row_sum != 0.0) {
                atomicAdd(&y[current_row], alpha * row_sum);
                row_sum = 0.0;
            }
            ++current_row;
        }
        row_sum += values[k] * x[col_ind[k]];
    }

    if (row_sum != 0.0) {
        atomicAdd(&y[current_row], alpha * row_sum);
    }
}

// Pre-scale y kernel for balanced SpMV when beta != 0
__global__ void scale_vector_kernel(scalar_t* __restrict__ y, scalar_t beta, index_t n) {
    index_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    index_t stride = blockDim.x * gridDim.x;
    for (index_t i = idx; i < n; i += stride) {
        y[i] *= beta;
    }
}

__global__ void zero_vector_kernel(scalar_t* __restrict__ y, index_t n) {
    index_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    index_t stride = blockDim.x * gridDim.x;
    for (index_t i = idx; i < n; i += stride) {
        y[i] = 0.0;
    }
}

void launch_spmv_csr_scalar_impl(scalar_t alpha, index_t num_rows, index_t num_cols,
                                 const index_t* d_row_ptr, const index_t* d_col_ind, const scalar_t* d_values,
                                 const scalar_t* d_x, scalar_t beta, scalar_t* d_y,
                                 cudaStream_t stream) {
    (void)num_cols;
    if (num_rows == 0) return;
    constexpr int block_size = 256;
    int num_blocks = (num_rows + block_size - 1) / block_size;
    if (num_blocks > 65535) num_blocks = 65535;

    spmv_csr_scalar_kernel<<<num_blocks, block_size, 0, stream>>>(
        alpha, num_rows, d_row_ptr, d_col_ind, d_values, d_x, beta, d_y);
    CUDA_CHECK_LAST_ERROR();
}

void launch_spmv_csr_vector_impl(scalar_t alpha, index_t num_rows, index_t num_cols,
                                 const index_t* d_row_ptr, const index_t* d_col_ind, const scalar_t* d_values,
                                 const scalar_t* d_x, scalar_t beta, scalar_t* d_y,
                                 cudaStream_t stream) {
    (void)num_cols;
    if (num_rows == 0) return;
    constexpr int block_size = 256; // 8 warps per block
    constexpr int warps_per_block = block_size / 32;
    int num_blocks = (num_rows + warps_per_block - 1) / warps_per_block;
    if (num_blocks > 65535) num_blocks = 65535;

    spmv_csr_vector_kernel<<<num_blocks, block_size, 0, stream>>>(
        alpha, num_rows, d_row_ptr, d_col_ind, d_values, d_x, beta, d_y);
    CUDA_CHECK_LAST_ERROR();
}

void launch_spmv_csr_adaptive_impl(scalar_t alpha, index_t num_rows, index_t num_cols,
                                   const index_t* d_row_ptr, const index_t* d_col_ind, const scalar_t* d_values,
                                   const scalar_t* d_x, scalar_t beta, scalar_t* d_y,
                                   cudaStream_t stream) {
    (void)num_cols;
    if (num_rows == 0) return;
    constexpr int block_size = 256;
    constexpr int subwarps_per_block = block_size / 8; // 32 subwarps per block
    int num_blocks = (num_rows + subwarps_per_block - 1) / subwarps_per_block;
    if (num_blocks > 65535) num_blocks = 65535;

    spmv_csr_adaptive_subwarp8_kernel<<<num_blocks, block_size, 0, stream>>>(
        alpha, num_rows, d_row_ptr, d_col_ind, d_values, d_x, beta, d_y);
    CUDA_CHECK_LAST_ERROR();
}

void launch_spmv_csr_balanced_impl(scalar_t alpha, index_t num_rows, index_t num_cols, size_t nnz,
                                   const index_t* d_row_ptr, const index_t* d_col_ind, const scalar_t* d_values,
                                   const scalar_t* d_x, scalar_t beta, scalar_t* d_y,
                                   cudaStream_t stream) {
    (void)num_cols;
    if (num_rows == 0) return;
    if (nnz == 0) {
        if (beta == 0.0) {
            constexpr int bs = 256;
            int blocks = (num_rows + bs - 1) / bs;
            if (blocks > 65535) blocks = 65535;
            zero_vector_kernel<<<blocks, bs, 0, stream>>>(d_y, num_rows);
        } else if (beta != 1.0) {
            constexpr int bs = 256;
            int blocks = (num_rows + bs - 1) / bs;
            if (blocks > 65535) blocks = 65535;
            scale_vector_kernel<<<blocks, bs, 0, stream>>>(d_y, beta, num_rows);
        }
        CUDA_CHECK_LAST_ERROR();
        return;
    }

    constexpr int block_size = 256;
    // Scale or zero y first before atomic accumulation
    int blocks_y = (num_rows + block_size - 1) / block_size;
    if (blocks_y > 65535) blocks_y = 65535;
    if (beta == 0.0) {
        zero_vector_kernel<<<blocks_y, block_size, 0, stream>>>(d_y, num_rows);
    } else if (beta != 1.0) {
        scale_vector_kernel<<<blocks_y, block_size, 0, stream>>>(d_y, beta, num_rows);
    }

    // Grid sized to saturate SMs (e.g. 128 blocks * 256 threads = 32768 threads)
    int num_blocks = static_cast<int>((nnz + block_size - 1) / block_size);
    if (num_blocks > 1024) num_blocks = 1024;
    if (num_blocks < 1) num_blocks = 1;

    spmv_csr_balanced_kernel<<<num_blocks, block_size, 0, stream>>>(
        alpha, num_rows, nnz, d_row_ptr, d_col_ind, d_values, d_x, d_y);
    CUDA_CHECK_LAST_ERROR();
}

} // namespace pipepye::cuda::kernels

namespace pipepye::cuda {

void launch_spmv_csr_scalar(scalar_t alpha, index_t num_rows, index_t num_cols,
                            const index_t* d_row_ptr, const index_t* d_col_ind, const scalar_t* d_values,
                            const scalar_t* d_x, scalar_t beta, scalar_t* d_y,
                            cudaStream_t stream) {
    kernels::launch_spmv_csr_scalar_impl(alpha, num_rows, num_cols, d_row_ptr, d_col_ind, d_values,
                                         d_x, beta, d_y, stream);
}

void launch_spmv_csr_vector(scalar_t alpha, index_t num_rows, index_t num_cols,
                            const index_t* d_row_ptr, const index_t* d_col_ind, const scalar_t* d_values,
                            const scalar_t* d_x, scalar_t beta, scalar_t* d_y,
                            cudaStream_t stream) {
    kernels::launch_spmv_csr_vector_impl(alpha, num_rows, num_cols, d_row_ptr, d_col_ind, d_values,
                                         d_x, beta, d_y, stream);
}

void launch_spmv_csr_adaptive(scalar_t alpha, index_t num_rows, index_t num_cols,
                              const index_t* d_row_ptr, const index_t* d_col_ind, const scalar_t* d_values,
                              const scalar_t* d_x, scalar_t beta, scalar_t* d_y,
                              cudaStream_t stream) {
    kernels::launch_spmv_csr_adaptive_impl(alpha, num_rows, num_cols, d_row_ptr, d_col_ind, d_values,
                                           d_x, beta, d_y, stream);
}

void launch_spmv_csr_balanced(scalar_t alpha, index_t num_rows, index_t num_cols, size_t nnz,
                              const index_t* d_row_ptr, const index_t* d_col_ind, const scalar_t* d_values,
                              const scalar_t* d_x, scalar_t beta, scalar_t* d_y,
                              cudaStream_t stream) {
    kernels::launch_spmv_csr_balanced_impl(alpha, num_rows, num_cols, nnz, d_row_ptr, d_col_ind, d_values,
                                           d_x, beta, d_y, stream);
}

} // namespace pipepye::cuda
