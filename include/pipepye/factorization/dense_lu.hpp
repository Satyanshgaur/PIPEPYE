#pragma once

#include <pipepye/core/types.hpp>
#include <pipepye/core/status.hpp>
#include <vector>

namespace pipepye::factorization {

/// @brief Dense LU Factorization with partial row pivoting (PA = LU).
/// Serves as the numerical correctness oracle for small basis solves, condition estimation, and testing.
class DenseLU {
public:
    DenseLU() = default;
    explicit DenseLU(index_t dim);

    /// @brief Computes dense LU factorization with partial row pivoting: P * A = L * U.
    /// @param A_row_major Flat row-major matrix of dimension dim x dim.
    /// @param tol Singularity tolerance (default 1e-12).
    [[nodiscard]] Status factorize(const std::vector<scalar_t>& A_row_major, scalar_t tol = 1e-12);

    /// @brief FTRAN solve: computes x = B^{-1} b via forward/backward substitution (L U x = P b).
    [[nodiscard]] std::vector<scalar_t> solve_ftran(const std::vector<scalar_t>& b) const;

    /// @brief BTRAN solve: computes y = B^{-T} b via backward/forward substitution (U^T L^T P x = b).
    [[nodiscard]] std::vector<scalar_t> solve_btran(const std::vector<scalar_t>& b) const;

    [[nodiscard]] index_t dimension() const noexcept { return dim_; }
    [[nodiscard]] bool is_factorized() const noexcept { return is_factorized_; }
    [[nodiscard]] scalar_t min_pivot() const noexcept { return min_pivot_; }

private:
    index_t dim_{0};
    std::vector<scalar_t> lu_;          ///< Fused L and U factors (dim x dim)
    std::vector<index_t> piv_;          ///< Permutation vector P (size dim)
    bool is_factorized_{false};
    scalar_t min_pivot_{0.0};
};

} // namespace pipepye::factorization
