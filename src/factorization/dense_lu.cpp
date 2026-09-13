#include <pipepye/factorization/dense_lu.hpp>
#include <cmath>
#include <algorithm>
#include <numeric>

namespace pipepye::factorization {

DenseLU::DenseLU(index_t dim)
    : dim_(dim),
      lu_(static_cast<size_t>(dim * dim), 0.0),
      piv_(static_cast<size_t>(dim), 0),
      is_factorized_(false),
      min_pivot_(0.0) {}

Status DenseLU::factorize(const std::vector<scalar_t>& A_row_major, scalar_t tol) {
    if (dim_ <= 0) {
        return Status::InvalidArgument("DenseLU dimension must be positive");
    }
    if (static_cast<index_t>(A_row_major.size()) != dim_ * dim_) {
        return Status::InvalidArgument("DenseLU input size does not match dim * dim");
    }

    lu_ = A_row_major;
    piv_.resize(dim_);
    std::iota(piv_.begin(), piv_.end(), 0);

    min_pivot_ = 1e30;

    for (index_t k = 0; k < dim_; ++k) {
        // Find pivot in column k below diagonal
        index_t max_row = k;
        scalar_t max_val = std::abs(lu_[k * dim_ + k]);

        for (index_t i = k + 1; i < dim_; ++i) {
            scalar_t val = std::abs(lu_[i * dim_ + k]);
            if (val > max_val) {
                max_val = val;
                max_row = i;
            }
        }

        if (max_val <= tol) {
            is_factorized_ = false;
            return Status::NumericalFailure("DenseLU: singular or near-singular pivot detected at column " +
                                          std::to_string(k) + " (pivot = " + std::to_string(max_val) + ")");
        }

        min_pivot_ = std::min(min_pivot_, max_val);

        // Swap rows k and max_row if needed
        if (max_row != k) {
            std::swap(piv_[k], piv_[max_row]);
            for (index_t j = 0; j < dim_; ++j) {
                std::swap(lu_[k * dim_ + j], lu_[max_row * dim_ + j]);
            }
        }

        scalar_t diag = lu_[k * dim_ + k];

        // Elimination
        for (index_t i = k + 1; i < dim_; ++i) {
            scalar_t factor = lu_[i * dim_ + k] / diag;
            lu_[i * dim_ + k] = factor; // Store L entry

            for (index_t j = k + 1; j < dim_; ++j) {
                lu_[i * dim_ + j] -= factor * lu_[k * dim_ + j];
            }
        }
    }

    is_factorized_ = true;
    return Status::OK();
}

std::vector<scalar_t> DenseLU::solve_ftran(const std::vector<scalar_t>& b) const {
    if (!is_factorized_ || static_cast<index_t>(b.size()) != dim_) {
        return {};
    }

    // Apply row permutation: y = P * b
    std::vector<scalar_t> y(dim_);
    for (index_t i = 0; i < dim_; ++i) {
        y[i] = b[piv_[i]];
    }

    // Forward substitution: L * y = P * b (L has unit diagonal)
    for (index_t i = 0; i < dim_; ++i) {
        for (index_t j = 0; j < i; ++j) {
            y[i] -= lu_[i * dim_ + j] * y[j];
        }
    }

    // Backward substitution: U * x = y
    std::vector<scalar_t> x = y;
    for (index_t i = dim_ - 1; i >= 0; --i) {
        for (index_t j = i + 1; j < dim_; ++j) {
            x[i] -= lu_[i * dim_ + j] * x[j];
        }
        x[i] /= lu_[i * dim_ + i];
    }

    return x;
}

std::vector<scalar_t> DenseLU::solve_btran(const std::vector<scalar_t>& b) const {
    if (!is_factorized_ || static_cast<index_t>(b.size()) != dim_) {
        return {};
    }

    // Solve B^T x = b <=> (P^T L U)^T x = b <=> U^T L^T P x = b
    // 1. Forward substitution: U^T * y = b
    std::vector<scalar_t> y = b;
    for (index_t i = 0; i < dim_; ++i) {
        for (index_t j = 0; j < i; ++j) {
            y[i] -= lu_[j * dim_ + i] * y[j];
        }
        y[i] /= lu_[i * dim_ + i];
    }

    // 2. Backward substitution: L^T * z = y (L has unit diagonal)
    std::vector<scalar_t> z = y;
    for (index_t i = dim_ - 1; i >= 0; --i) {
        for (index_t j = i + 1; j < dim_; ++j) {
            z[i] -= lu_[j * dim_ + i] * z[j];
        }
    }

    // 3. Inverse permutation: x = P^T * z
    std::vector<scalar_t> x(dim_);
    for (index_t i = 0; i < dim_; ++i) {
        x[piv_[i]] = z[i];
    }

    return x;
}

} // namespace pipepye::factorization
