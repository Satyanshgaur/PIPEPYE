#pragma once

#include <pipepye/core/types.hpp>
#include <pipepye/core/status.hpp>
#include <pipepye/sparse/csc_matrix.hpp>
#include <pipepye/factorization/sparse_lu.hpp>
#include <pipepye/factorization/dense_lu.hpp>
#include <pipepye/simplex/simplex_types.hpp>
#include <vector>
#include <span>

namespace pipepye::factorization {

/// @brief Represents a single elementary eta matrix for Product Form of Inverse (PFI).
/// E = I - ((v - e_p) / v_p) e_p^T
/// Storage: pivot row index p, pivot diagonal scale eta_diag = 1 / v_p,
/// and sparse off-diagonal entries val[k] = -v_{i_k} for i_k != p.
struct EtaVector {
    index_t pivot_row{0};
    scalar_t eta_diag{0.0};             ///< 1.0 / v_p
    std::vector<index_t> ind;           ///< Row indices i != pivot_row
    std::vector<scalar_t> val;          ///< Values -v_i
};

/// @brief Basis Factorization Engine.
/// Manages sparse LU factorization of the base matrix B_0, along with a sequence of PFI eta vectors.
/// Provides forward transformation (FTRAN) and backward transformation (BTRAN), as well as refactorization policies.
class BasisFactorization {
public:
    BasisFactorization() = default;
    explicit BasisFactorization(index_t num_rows);

    /// @brief Refactorizes the base basis matrix B_0 from scratch.
    /// @param A Augmented or original constraint matrix in CSC format
    /// @param basic_vars Array of length m indicating which variable is in each basis position
    /// @param num_structurals Number of structural variables n (vars >= n are slacks)
    /// @param markowitz_threshold Relative pivot threshold for sparse LU (default 0.1)
    /// @param singularity_tol Absolute pivot cutoff (default 1e-12)
    [[nodiscard]] Status factorize(
        const sparse::CSCMatrix& A,
        std::span<const index_t> basic_vars,
        index_t num_structurals,
        scalar_t markowitz_threshold = 0.1,
        scalar_t singularity_tol = 1e-12);

    /// @brief Appends a PFI eta update when column pivot_row of B is replaced by entering column with FTRAN v.
    /// @param pivot_row Position in basis (0 <= pivot_row < m)
    /// @param ftran_col Pivot column v = B^{-1} a_q
    /// @param pivot_tol Minimum absolute pivot magnitude allowed
    [[nodiscard]] Status update(
        index_t pivot_row,
        const std::vector<scalar_t>& ftran_col,
        scalar_t pivot_tol = 1e-8);

    /// @brief FTRAN solve: computes x = B^{-1} b.
    [[nodiscard]] std::vector<scalar_t> solve_ftran(const std::vector<scalar_t>& b);

    /// @brief BTRAN solve: computes y = B^{-T} b.
    [[nodiscard]] std::vector<scalar_t> solve_btran(const std::vector<scalar_t>& b);

    /// @brief Returns whether a refactorization is recommended due to eta count or numerical growth.
    [[nodiscard]] bool needs_refactorize(int max_updates = 60) const noexcept;

    /// @brief Resets all eta vectors without re-factorizing base LU (e.g. after fresh factorize).
    void clear_updates() noexcept;

    /// @brief Enable or disable dense oracle validation.
    void set_use_dense_oracle(bool enable) noexcept { use_dense_oracle_ = enable; }
    [[nodiscard]] bool use_dense_oracle() const noexcept { return use_dense_oracle_; }

    [[nodiscard]] index_t dimension() const noexcept { return num_rows_; }
    [[nodiscard]] bool is_factorized() const noexcept { return is_factorized_; }
    [[nodiscard]] size_t num_updates() const noexcept { return etas_.size(); }
    [[nodiscard]] int factorization_count() const noexcept { return factorizations_; }
    [[nodiscard]] int update_count() const noexcept { return updates_; }
    [[nodiscard]] int ftran_count() const noexcept { return ftran_count_; }
    [[nodiscard]] int btran_count() const noexcept { return btran_count_; }

    [[nodiscard]] const SparseLU& sparse_lu() const noexcept { return sparse_lu_; }
    [[nodiscard]] const DenseLU& dense_lu() const noexcept { return dense_lu_; }

    /// @brief Utility to extract a column of the augmented matrix [A -I].
    /// @param var_idx Variable index in [0, n + m)
    /// @param num_structurals n
    /// @param A Original constraint matrix CSC
    /// @param out_row_ind Output row indices
    /// @param out_values Output nonzero values
    static void extract_column(
        index_t var_idx,
        index_t num_structurals,
        const sparse::CSCMatrix& A,
        std::vector<index_t>& out_row_ind,
        std::vector<scalar_t>& out_values);

    /// @brief Utility to expand a sparse column into a dense vector of size m.
    static std::vector<scalar_t> expand_to_dense(
        index_t m,
        std::span<const index_t> row_ind,
        std::span<const scalar_t> values);

private:
    index_t num_rows_{0};
    bool is_factorized_{false};
    bool use_dense_oracle_{false};

    SparseLU sparse_lu_;
    DenseLU dense_lu_;
    std::vector<EtaVector> etas_;

    int factorizations_{0};
    int updates_{0};
    int ftran_count_{0};
    int btran_count_{0};
};

} // namespace pipepye::factorization
