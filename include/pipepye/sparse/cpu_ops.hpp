#pragma once

#include <pipepye/sparse/vector.hpp>
#include <pipepye/sparse/csr_matrix.hpp>
#include <pipepye/sparse/csc_matrix.hpp>

namespace pipepye::sparse::cpu_ops {

// =============================================================================
// Vector Primitives
// =============================================================================

/// @brief Computes y = alpha * x + y
inline void axpy(scalar_t alpha, ConstVectorView x, MutableVectorView y) {
    y.axpy(alpha, x);
}

/// @brief Computes y = alpha * x + beta * y
inline void axpby(scalar_t alpha, ConstVectorView x, scalar_t beta, MutableVectorView y) {
    y.axpby(alpha, x, beta);
}

/// @brief Computes dot product x^T y
[[nodiscard]] inline scalar_t dot(ConstVectorView x, ConstVectorView y) {
    return x.dot(y);
}

/// @brief Computes L1 norm: sum(|x_i|)
[[nodiscard]] inline scalar_t norm_1(ConstVectorView x) noexcept {
    return x.norm_1();
}

/// @brief Computes L2 (Euclidean) norm: sqrt(sum(x_i^2))
[[nodiscard]] inline scalar_t norm_2(ConstVectorView x) noexcept {
    return x.norm_2();
}

/// @brief Computes squared L2 norm: sum(x_i^2)
[[nodiscard]] inline scalar_t norm_2_sq(ConstVectorView x) noexcept {
    return x.norm_2_sq();
}

/// @brief Computes L_infinity norm: max(|x_i|)
[[nodiscard]] inline scalar_t norm_inf(ConstVectorView x) noexcept {
    return x.norm_inf();
}

/// @brief Computes sum(x_i)
[[nodiscard]] inline scalar_t sum(ConstVectorView x) noexcept {
    return x.sum();
}

/// @brief Computes mean(x)
[[nodiscard]] inline scalar_t mean(ConstVectorView x) {
    return x.mean();
}

/// @brief Returns minimum element
[[nodiscard]] inline scalar_t min(ConstVectorView x) {
    return x.min();
}

/// @brief Returns maximum element
[[nodiscard]] inline scalar_t max(ConstVectorView x) {
    return x.max();
}

/// @brief Returns index of minimum element
[[nodiscard]] inline index_t argmin(ConstVectorView x) {
    return x.argmin();
}

/// @brief Returns index of maximum element
[[nodiscard]] inline index_t argmax(ConstVectorView x) {
    return x.argmax();
}

/// @brief Scales vector in-place: x = alpha * x
inline void scale(scalar_t alpha, MutableVectorView x) {
    x.scale(alpha);
}

/// @brief Fills vector with constant value
inline void fill(MutableVectorView x, scalar_t val) {
    x.fill(val);
}

/// @brief Sets all elements to zero
inline void set_zero(MutableVectorView x) {
    x.set_zero();
}

/// @brief Copies elements from src to dst
inline void copy(ConstVectorView src, MutableVectorView dst) {
    dst.copy_from(src);
}

/// @brief Clamps elements to box bounds [lower, upper]
inline void project_box(MutableVectorView x, ConstVectorView lower, ConstVectorView upper) {
    x.project_bounds(lower, upper);
}

/// @brief Element-wise product: out_i = x_i * y_i
void hadamard(ConstVectorView x, ConstVectorView y, MutableVectorView out);

/// @brief Maximum absolute difference: max(|a_i - b_i|)
[[nodiscard]] inline scalar_t abs_diff_inf(ConstVectorView a, ConstVectorView b) {
    return a.abs_diff_inf(b);
}

/// @brief Euclidean difference: sqrt(sum((a_i - b_i)^2))
[[nodiscard]] inline scalar_t abs_diff_2(ConstVectorView a, ConstVectorView b) {
    return a.abs_diff_2(b);
}

// =============================================================================
// SpMV and SpMV Transpose Primitives
// =============================================================================

/// @brief CPU SpMV using CSR: y = alpha * A * x + beta * y
void spmv_csr(scalar_t alpha, const CSRMatrix& A, ConstVectorView x, scalar_t beta, MutableVectorView y);

/// @brief CPU SpMV Transpose using CSR: y = alpha * A^T * x + beta * y
void spmv_transpose_csr(scalar_t alpha, const CSRMatrix& A, ConstVectorView x, scalar_t beta, MutableVectorView y);

/// @brief CPU SpMV using CSC: y = alpha * A * x + beta * y
void spmv_csc(scalar_t alpha, const CSCMatrix& A, ConstVectorView x, scalar_t beta, MutableVectorView y);

/// @brief CPU SpMV Transpose using CSC: y = alpha * A^T * x + beta * y
void spmv_transpose_csc(scalar_t alpha, const CSCMatrix& A, ConstVectorView x, scalar_t beta, MutableVectorView y);

// =============================================================================
// Multithreaded Parallel Operations (OpenMP / Core-scaling)
// =============================================================================

/// @brief Returns the maximum number of hardware execution threads available
[[nodiscard]] int get_max_threads() noexcept;

/// @brief Multithreaded parallel CPU SpMV using CSR: y = alpha * A * x + beta * y
void spmv_csr_parallel(scalar_t alpha, const CSRMatrix& A, ConstVectorView x, scalar_t beta, MutableVectorView y, int num_threads = 0);

/// @brief Multithreaded parallel CPU SpMV Transpose using CSC: y = alpha * A^T * x + beta * y
void spmv_transpose_csc_parallel(scalar_t alpha, const CSCMatrix& A, ConstVectorView x, scalar_t beta, MutableVectorView y, int num_threads = 0);

/// @brief Multithreaded parallel AXPY: y = alpha * x + y
void axpy_parallel(scalar_t alpha, ConstVectorView x, MutableVectorView y, int num_threads = 0);

/// @brief Multithreaded parallel dot product: x^T y
[[nodiscard]] scalar_t dot_parallel(ConstVectorView x, ConstVectorView y, int num_threads = 0);

} // namespace pipepye::sparse::cpu_ops
