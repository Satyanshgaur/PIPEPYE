#pragma once

#include <cuda_runtime.h>
#include <pipepye/core/types.hpp>

namespace pipepye::cuda::kernels {

void launch_spmv_csr_scalar_impl(scalar_t alpha, index_t num_rows, index_t num_cols,
                                 const index_t* d_row_ptr, const index_t* d_col_ind, const scalar_t* d_values,
                                 const scalar_t* d_x, scalar_t beta, scalar_t* d_y,
                                 cudaStream_t stream = nullptr);

void launch_spmv_csr_vector_impl(scalar_t alpha, index_t num_rows, index_t num_cols,
                                 const index_t* d_row_ptr, const index_t* d_col_ind, const scalar_t* d_values,
                                 const scalar_t* d_x, scalar_t beta, scalar_t* d_y,
                                 cudaStream_t stream = nullptr);

void launch_spmv_csr_adaptive_impl(scalar_t alpha, index_t num_rows, index_t num_cols,
                                   const index_t* d_row_ptr, const index_t* d_col_ind, const scalar_t* d_values,
                                   const scalar_t* d_x, scalar_t beta, scalar_t* d_y,
                                   cudaStream_t stream = nullptr);

void launch_spmv_csr_balanced_impl(scalar_t alpha, index_t num_rows, index_t num_cols, size_t nnz,
                                   const index_t* d_row_ptr, const index_t* d_col_ind, const scalar_t* d_values,
                                   const scalar_t* d_x, scalar_t beta, scalar_t* d_y,
                                   cudaStream_t stream = nullptr);

} // namespace pipepye::cuda::kernels
