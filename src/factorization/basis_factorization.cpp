#include <pipepye/factorization/basis_factorization.hpp>
#include <cmath>
#include <algorithm>

namespace pipepye::factorization {

BasisFactorization::BasisFactorization(index_t num_rows)
    : num_rows_(num_rows), sparse_lu_(num_rows), dense_lu_(num_rows) {}

void BasisFactorization::extract_column(
    index_t var_idx,
    index_t num_structurals,
    const sparse::CSCMatrix& A,
    std::vector<index_t>& out_row_ind,
    std::vector<scalar_t>& out_values) {
    out_row_ind.clear();
    out_values.clear();

    if (var_idx < num_structurals) {
        auto col = A.col(var_idx);
        out_row_ind.assign(col.row_indices.begin(), col.row_indices.end());
        out_values.assign(col.values.begin(), col.values.end());
    } else {
        index_t slack_row = var_idx - num_structurals;
        out_row_ind.push_back(slack_row);
        out_values.push_back(-1.0);
    }
}

std::vector<scalar_t> BasisFactorization::expand_to_dense(
    index_t m,
    std::span<const index_t> row_ind,
    std::span<const scalar_t> values) {
    std::vector<scalar_t> dense(m, 0.0);
    for (size_t k = 0; k < row_ind.size(); ++k) {
        dense[row_ind[k]] = values[k];
    }
    return dense;
}

Status BasisFactorization::factorize(
    const sparse::CSCMatrix& A,
    std::span<const index_t> basic_vars,
    index_t num_structurals,
    scalar_t markowitz_threshold,
    scalar_t singularity_tol) {
    if (static_cast<index_t>(basic_vars.size()) != num_rows_) {
        return Status::InvalidArgument("basic_vars size does not match basis dimension m");
    }

    // Build CSC representation of m x m basis matrix B
    std::vector<index_t> col_ptr;
    col_ptr.reserve(num_rows_ + 1);
    col_ptr.push_back(0);

    std::vector<index_t> row_ind;
    std::vector<scalar_t> values;

    std::vector<index_t> temp_rows;
    std::vector<scalar_t> temp_vals;

    for (index_t k = 0; k < num_rows_; ++k) {
        index_t var = basic_vars[k];
        extract_column(var, num_structurals, A, temp_rows, temp_vals);

        row_ind.insert(row_ind.end(), temp_rows.begin(), temp_rows.end());
        values.insert(values.end(), temp_vals.begin(), temp_vals.end());
        col_ptr.push_back(static_cast<index_t>(row_ind.size()));
    }

    // Factorize with SparseLU
    Status s = sparse_lu_.factorize(num_rows_, col_ptr, row_ind, values, markowitz_threshold, singularity_tol);
    if (!s.is_ok()) {
        is_factorized_ = false;
        return s;
    }

    // Optionally factorize DenseLU oracle
    if (use_dense_oracle_) {
        std::vector<scalar_t> dense_b(static_cast<size_t>(num_rows_) * num_rows_, 0.0);
        for (index_t col = 0; col < num_rows_; ++col) {
            index_t start = col_ptr[col];
            index_t end = col_ptr[col + 1];
            for (index_t idx = start; idx < end; ++idx) {
                index_t row = row_ind[idx];
                dense_b[static_cast<size_t>(row) * num_rows_ + col] = values[idx];
            }
        }
        Status ds = dense_lu_.factorize(dense_b, singularity_tol);
        if (!ds.is_ok()) {
            return ds;
        }
    }

    clear_updates();
    is_factorized_ = true;
    factorizations_++;
    return Status::OK();
}

Status BasisFactorization::update(
    index_t pivot_row,
    const std::vector<scalar_t>& ftran_col,
    scalar_t pivot_tol) {
    if (pivot_row < 0 || pivot_row >= num_rows_) {
        return Status::InvalidArgument("pivot_row out of bounds");
    }
    if (static_cast<index_t>(ftran_col.size()) != num_rows_) {
        return Status::InvalidArgument("ftran_col size does not match basis dimension m");
    }

    scalar_t vp = ftran_col[pivot_row];
    if (std::abs(vp) < pivot_tol) {
        return Status::NumericalFailure("Pivot element below numerical tolerance: |v_p| < pivot_tol");
    }

    EtaVector eta;
    eta.pivot_row = pivot_row;
    eta.eta_diag = 1.0 / vp;

    for (index_t i = 0; i < num_rows_; ++i) {
        if (i == pivot_row) continue;
        scalar_t vi = ftran_col[i];
        if (std::abs(vi) > 1e-15) {
            eta.ind.push_back(i);
            eta.val.push_back(-vi);
        }
    }

    etas_.push_back(std::move(eta));
    updates_++;
    return Status::OK();
}

std::vector<scalar_t> BasisFactorization::solve_ftran(const std::vector<scalar_t>& b) {
    ftran_count_++;
    std::vector<scalar_t> x;
    if (use_dense_oracle_ && dense_lu_.is_factorized()) {
        x = dense_lu_.solve_ftran(b);
    } else {
        x = sparse_lu_.solve_ftran(b);
    }

    // Apply eta matrices forward: x_{t} = E_t * x_{t-1}
    for (const auto& eta : etas_) {
        index_t p = eta.pivot_row;
        scalar_t new_xp = x[p] * eta.eta_diag;
        const size_t nnz = eta.ind.size();
        for (size_t k = 0; k < nnz; ++k) {
            x[eta.ind[k]] += eta.val[k] * new_xp;
        }
        x[p] = new_xp;
    }

    return x;
}

std::vector<scalar_t> BasisFactorization::solve_btran(const std::vector<scalar_t>& b) {
    btran_count_++;
    std::vector<scalar_t> w = b;

    // Apply eta matrices backward: w_{t-1} = E_t^T * w_t
    for (auto it = etas_.rbegin(); it != etas_.rend(); ++it) {
        const auto& eta = *it;
        index_t p = eta.pivot_row;
        scalar_t sum = w[p];
        const size_t nnz = eta.ind.size();
        for (size_t k = 0; k < nnz; ++k) {
            sum += eta.val[k] * w[eta.ind[k]];
        }
        w[p] = sum * eta.eta_diag;
    }

    // Solve base system transpose
    if (use_dense_oracle_ && dense_lu_.is_factorized()) {
        return dense_lu_.solve_btran(w);
    }
    return sparse_lu_.solve_btran(w);
}

bool BasisFactorization::needs_refactorize(int max_updates) const noexcept {
    return static_cast<int>(etas_.size()) >= max_updates;
}

void BasisFactorization::clear_updates() noexcept {
    etas_.clear();
}

} // namespace pipepye::factorization
