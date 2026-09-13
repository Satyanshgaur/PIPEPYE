#pragma once

#include <pipepye/core/types.hpp>
#include <pipepye/core/status.hpp>
#include <pipepye/sparse/csr_matrix.hpp>
#include <pipepye/sparse/csc_matrix.hpp>
#include <vector>
#include <span>

namespace pipepye::factorization {

/// @brief Telemetry metrics recorded during sparse LU factorization.
struct SparseLUMetrics {
    index_t dim{0};
    size_t orig_nnz{0};
    size_t l_nnz{0};
    size_t u_nnz{0};
    double fill_in_ratio{0.0};
    double factorization_ms{0.0};
    scalar_t min_pivot{0.0};
    scalar_t max_pivot{0.0};
};

/// @brief High-performance Sparse LU Factorization with Markowitz Threshold Partial Pivoting.
/// Decomposes P * B * Q = L * U where L is unit lower triangular and U is upper triangular.
class SparseLU {
public:
    SparseLU() = default;
    explicit SparseLU(index_t dim);

    /// @brief Computes sparse LU factorization of an m x m basis matrix B from CSC format.
    /// @param dim Matrix dimension m
    /// @param col_ptr Column pointers (size dim + 1)
    /// @param row_ind Row indices
    /// @param values Nonzero values
    /// @param markowitz_threshold Relative pivot threshold u in [0.01, 0.5] (default 0.1)
    /// @param singularity_tol Minimum absolute pivot magnitude (default 1e-12)
    [[nodiscard]] Status factorize(
        index_t dim,
        std::span<const index_t> col_ptr,
        std::span<const index_t> row_ind,
        std::span<const scalar_t> values,
        scalar_t markowitz_threshold = 0.1,
        scalar_t singularity_tol = 1e-12);

    /// @brief Overload taking CSR basis matrix.
    [[nodiscard]] Status factorize(
        const sparse::CSRMatrix& B,
        scalar_t markowitz_threshold = 0.1,
        scalar_t singularity_tol = 1e-12);

    /// @brief Overload taking CSC basis matrix.
    [[nodiscard]] Status factorize(
        const sparse::CSCMatrix& B,
        scalar_t markowitz_threshold = 0.1,
        scalar_t singularity_tol = 1e-12);

    /// @brief FTRAN solve: computes x = B^{-1} b via L U (Q^T x) = P b.
    [[nodiscard]] std::vector<scalar_t> solve_ftran(const std::vector<scalar_t>& b) const;

    /// @brief BTRAN solve: computes y = B^{-T} b via Q U^T L^T (P y) = b.
    [[nodiscard]] std::vector<scalar_t> solve_btran(const std::vector<scalar_t>& b) const;

    [[nodiscard]] index_t dimension() const noexcept { return dim_; }
    [[nodiscard]] bool is_factorized() const noexcept { return is_factorized_; }
    [[nodiscard]] const SparseLUMetrics& metrics() const noexcept { return metrics_; }
    [[nodiscard]] size_t total_nnz() const noexcept { return metrics_.l_nnz + metrics_.u_nnz; }

private:
    index_t dim_{0};
    bool is_factorized_{false};
    SparseLUMetrics metrics_;

    // Permutations: row_perm_[i] is row position in P, col_perm_[j] is col position in Q
    std::vector<index_t> p_row_;        ///< Map step k -> row index in original matrix
    std::vector<index_t> p_row_inv_;    ///< Map original row -> step k
    std::vector<index_t> q_col_;        ///< Map step k -> col index in original matrix
    std::vector<index_t> q_col_inv_;    ///< Map original col -> step k

    // Sparse L representation (unit lower triangular, stored column-by-column)
    std::vector<index_t> l_col_ptr_;
    std::vector<index_t> l_row_ind_;
    std::vector<scalar_t> l_values_;

    // Sparse U representation (upper triangular, stored row-by-row)
    std::vector<index_t> u_row_ptr_;
    std::vector<index_t> u_col_ind_;
    std::vector<scalar_t> u_values_;
    std::vector<scalar_t> u_diag_;      ///< U[k, k] diagonal elements for rapid FTRAN/BTRAN
};

} // namespace pipepye::factorization
