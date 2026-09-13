#pragma once

#include <cuda_runtime.h>
#include <pipepye/core/types.hpp>
#include <pipepye/sparse/csr_matrix.hpp>
#include <pipepye/sparse/vector.hpp>
#include <memory>
#include <string>

namespace pipepye::cuda {

/// @brief SpMV GPU Execution Strategy / Kernel Variant
enum class SpMVKernelVariant {
    Scalar = 0,     // Baseline: 1 thread per row (best for very short, uniform rows)
    Vector,         // 1 warp (32 threads) per row (best for medium/long rows, memory coalesced)
    Adaptive,       // Dynamic row-partitioned (hybrid sub-warp / block reduction for hub rows)
    Balanced        // Work-balanced / merge-path chunked (partitions NNZ evenly across threads)
};

[[nodiscard]] std::string to_string(SpMVKernelVariant variant);

/// @brief RAII managed device vector allocated in GPU VRAM
class DeviceVector {
public:
    DeviceVector();
    explicit DeviceVector(size_t size, cudaStream_t stream = nullptr);
    explicit DeviceVector(sparse::ConstVectorView host_view, cudaStream_t stream = nullptr);
    ~DeviceVector();

    // Move semantics
    DeviceVector(DeviceVector&& other) noexcept;
    DeviceVector& operator=(DeviceVector&& other) noexcept;

    // Disallow copying to prevent accidental PCIe traffic
    DeviceVector(const DeviceVector&) = delete;
    DeviceVector& operator=(const DeviceVector&) = delete;

    [[nodiscard]] size_t size() const noexcept { return size_; }
    [[nodiscard]] scalar_t* data() noexcept { return d_data_; }
    [[nodiscard]] const scalar_t* data() const noexcept { return d_data_; }

    void copy_from_host(sparse::ConstVectorView host_view, cudaStream_t stream = nullptr);
    void copy_to_host(sparse::MutableVectorView host_view, cudaStream_t stream = nullptr) const;
    void fill(scalar_t value, cudaStream_t stream = nullptr);
    void set_zero(cudaStream_t stream = nullptr);

private:
    scalar_t* d_data_{nullptr};
    size_t size_{0};
};

/// @brief RAII managed Compressed Sparse Row matrix resident in GPU VRAM
class DeviceCSRMatrix {
public:
    DeviceCSRMatrix();
    explicit DeviceCSRMatrix(const sparse::CSRMatrix& host_csr, cudaStream_t stream = nullptr);
    ~DeviceCSRMatrix();

    // Move semantics
    DeviceCSRMatrix(DeviceCSRMatrix&& other) noexcept;
    DeviceCSRMatrix& operator=(DeviceCSRMatrix&& other) noexcept;

    DeviceCSRMatrix(const DeviceCSRMatrix&) = delete;
    DeviceCSRMatrix& operator=(const DeviceCSRMatrix&) = delete;

    [[nodiscard]] index_t num_rows() const noexcept { return num_rows_; }
    [[nodiscard]] index_t num_cols() const noexcept { return num_cols_; }
    [[nodiscard]] size_t num_nonzeros() const noexcept { return nnz_; }

    [[nodiscard]] const index_t* row_ptr() const noexcept { return d_row_ptr_; }
    [[nodiscard]] const index_t* col_ind() const noexcept { return d_col_ind_; }
    [[nodiscard]] const scalar_t* values() const noexcept { return d_values_; }

    /// @brief Executes SpMV: y = alpha * A * x + beta * y using specified kernel variant
    void spmv(scalar_t alpha, const DeviceVector& x, scalar_t beta, DeviceVector& y,
              SpMVKernelVariant variant = SpMVKernelVariant::Vector,
              cudaStream_t stream = nullptr) const;

    /// @brief Executes SpMV using raw device pointers
    void spmv_raw(scalar_t alpha, const scalar_t* d_x, scalar_t beta, scalar_t* d_y,
                  SpMVKernelVariant variant = SpMVKernelVariant::Vector,
                  cudaStream_t stream = nullptr) const;

private:
    index_t num_rows_{0};
    index_t num_cols_{0};
    size_t nnz_{0};

    index_t* d_row_ptr_{nullptr};
    index_t* d_col_ind_{nullptr};
    scalar_t* d_values_{nullptr};
};

// =============================================================================
// Low-Level Kernel Launch Wrappers
// =============================================================================

void launch_spmv_csr_scalar(scalar_t alpha, index_t num_rows, index_t num_cols,
                            const index_t* d_row_ptr, const index_t* d_col_ind, const scalar_t* d_values,
                            const scalar_t* d_x, scalar_t beta, scalar_t* d_y,
                            cudaStream_t stream = nullptr);

void launch_spmv_csr_vector(scalar_t alpha, index_t num_rows, index_t num_cols,
                            const index_t* d_row_ptr, const index_t* d_col_ind, const scalar_t* d_values,
                            const scalar_t* d_x, scalar_t beta, scalar_t* d_y,
                            cudaStream_t stream = nullptr);

void launch_spmv_csr_adaptive(scalar_t alpha, index_t num_rows, index_t num_cols,
                              const index_t* d_row_ptr, const index_t* d_col_ind, const scalar_t* d_values,
                              const scalar_t* d_x, scalar_t beta, scalar_t* d_y,
                              cudaStream_t stream = nullptr);

void launch_spmv_csr_balanced(scalar_t alpha, index_t num_rows, index_t num_cols, size_t nnz,
                              const index_t* d_row_ptr, const index_t* d_col_ind, const scalar_t* d_values,
                              const scalar_t* d_x, scalar_t beta, scalar_t* d_y,
                              cudaStream_t stream = nullptr);

} // namespace pipepye::cuda
