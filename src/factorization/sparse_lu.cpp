#include <pipepye/factorization/sparse_lu.hpp>
#include <pipepye/utils/timer.hpp>
#include <cmath>
#include <algorithm>
#include <unordered_map>
#include <numeric>

namespace pipepye::factorization {

SparseLU::SparseLU(index_t dim)
    : dim_(dim), is_factorized_(false) {}

Status SparseLU::factorize(
    const sparse::CSRMatrix& B,
    scalar_t markowitz_threshold,
    scalar_t singularity_tol) {
    auto csc = B.to_csc();
    return factorize(B.num_rows(), csc.col_ptr(), csc.row_ind(), csc.values(),
                     markowitz_threshold, singularity_tol);
}

Status SparseLU::factorize(
    const sparse::CSCMatrix& B,
    scalar_t markowitz_threshold,
    scalar_t singularity_tol) {
    return factorize(B.num_rows(), B.col_ptr(), B.row_ind(), B.values(),
                     markowitz_threshold, singularity_tol);
}

Status SparseLU::factorize(
    index_t dim,
    std::span<const index_t> col_ptr,
    std::span<const index_t> row_ind,
    std::span<const scalar_t> values,
    scalar_t markowitz_threshold,
    scalar_t singularity_tol) {

    utils::CPUTimer timer;
    timer.start();

    if (dim <= 0) {
        return Status::InvalidArgument("SparseLU: dimension must be positive");
    }
    dim_ = dim;
    is_factorized_ = false;

    metrics_ = SparseLUMetrics{};
    metrics_.dim = dim;
    metrics_.orig_nnz = values.size();
    metrics_.min_pivot = 1e30;
    metrics_.max_pivot = 0.0;

    // Build dynamic row and column structures for Gaussian elimination
    // Each row i stores map: col -> value
    std::vector<std::unordered_map<index_t, scalar_t>> rows(dim);
    std::vector<std::unordered_map<index_t, scalar_t>> cols(dim);

    for (index_t j = 0; j < dim; ++j) {
        index_t start = col_ptr[j];
        index_t end = col_ptr[j + 1];
        for (index_t p = start; p < end; ++p) {
            index_t i = row_ind[p];
            scalar_t val = values[p];
            if (std::abs(val) > 1e-15) {
                rows[i][j] = val;
                cols[j][i] = val;
            }
        }
    }

    std::vector<bool> row_eliminated(dim, false);
    std::vector<bool> col_eliminated(dim, false);

    p_row_.assign(dim, -1);
    p_row_inv_.assign(dim, -1);
    q_col_.assign(dim, -1);
    q_col_inv_.assign(dim, -1);

    l_col_ptr_.assign(dim + 1, 0);
    l_row_ind_.clear();
    l_values_.clear();

    u_row_ptr_.assign(dim + 1, 0);
    u_col_ind_.clear();
    u_values_.clear();
    u_diag_.assign(dim, 0.0);

    // Elimination loop: k = 0 to dim - 1
    for (index_t k = 0; k < dim; ++k) {
        // Markowitz search for pivot (best_r, best_c)
        index_t best_r = -1;
        index_t best_c = -1;
        long best_cost = std::numeric_limits<long>::max();
        scalar_t best_val = 0.0;

        // Search active columns
        for (index_t j = 0; j < dim; ++j) {
            if (col_eliminated[j]) continue;

            // Find column maximum among uneliminated rows
            scalar_t col_max = 0.0;
            long col_degree = 0;
            for (const auto& [r, val] : cols[j]) {
                if (!row_eliminated[r]) {
                    col_max = std::max(col_max, std::abs(val));
                    col_degree++;
                }
            }

            if (col_max <= singularity_tol || col_degree == 0) continue;

            scalar_t thresh = markowitz_threshold * col_max;

            for (const auto& [r, val] : cols[j]) {
                if (row_eliminated[r]) continue;
                scalar_t abs_v = std::abs(val);
                if (abs_v >= thresh) {
                    long row_degree = 0;
                    for (const auto& [c, v] : rows[r]) {
                        if (!col_eliminated[c]) row_degree++;
                    }
                    long cost = (row_degree - 1) * (col_degree - 1);

                    if (cost < best_cost || (cost == best_cost && abs_v > std::abs(best_val))) {
                        best_cost = cost;
                        best_r = r;
                        best_c = j;
                        best_val = val;
                        if (best_cost == 0) break; // Singleton: optimal Markowitz cost
                    }
                }
            }
            if (best_cost == 0) break;
        }

        if (best_r < 0 || best_c < 0 || std::abs(best_val) <= singularity_tol) {
            timer.stop();
            metrics_.factorization_ms = timer.elapsed_milliseconds();
            return Status::NumericalFailure("SparseLU: matrix is structurally or numerically singular at step " +
                                            std::to_string(k) + ", no valid Markowitz pivot found.");
        }

        // Record pivot
        p_row_[k] = best_r;
        p_row_inv_[best_r] = k;
        q_col_[k] = best_c;
        q_col_inv_[best_c] = k;

        scalar_t pivot = best_val;
        metrics_.min_pivot = std::min(metrics_.min_pivot, std::abs(pivot));
        metrics_.max_pivot = std::max(metrics_.max_pivot, std::abs(pivot));
        u_diag_[k] = pivot;

        row_eliminated[best_r] = true;
        col_eliminated[best_c] = true;

        // Construct row k of U from remaining active entries of row best_r
        u_row_ptr_[k] = static_cast<index_t>(u_col_ind_.size());
        for (const auto& [c, val] : rows[best_r]) {
            if (c != best_c && !col_eliminated[c]) {
                u_col_ind_.push_back(c);
                u_values_.push_back(val);
            }
        }

        // Construct column k of L: multipliers m_i = A[i, best_c] / pivot
        l_col_ptr_[k] = static_cast<index_t>(l_row_ind_.size());
        std::vector<std::pair<index_t, scalar_t>> multipliers;
        for (const auto& [r, val] : cols[best_c]) {
            if (r != best_r && !row_eliminated[r]) {
                scalar_t m = val / pivot;
                multipliers.push_back({r, m});
                l_row_ind_.push_back(r);
                l_values_.push_back(m);
            }
        }

        // Clean up pivot row and col from dynamic structures
        for (const auto& [c, val] : rows[best_r]) {
            cols[c].erase(best_r);
        }
        rows[best_r].clear();

        for (const auto& [r, val] : cols[best_c]) {
            rows[r].erase(best_c);
        }
        cols[best_c].clear();

        // Update active submatrix: A[r, c] -= m * A[best_r, c]
        index_t u_start = u_row_ptr_[k];
        index_t u_end = static_cast<index_t>(u_col_ind_.size());

        for (const auto& [r, m] : multipliers) {
            for (index_t p = u_start; p < u_end; ++p) {
                index_t c = u_col_ind_[p];
                scalar_t u_val = u_values_[p];
                scalar_t update = m * u_val;

                auto it = rows[r].find(c);
                if (it != rows[r].end()) {
                    it->second -= update;
                    cols[c][r] = it->second;
                    if (std::abs(it->second) <= 1e-15) {
                        rows[r].erase(it);
                        cols[c].erase(r);
                    }
                } else if (std::abs(update) > 1e-15) {
                    rows[r][c] = -update;
                    cols[c][r] = -update;
                }
            }
        }
    }

    l_col_ptr_[dim] = static_cast<index_t>(l_row_ind_.size());
    u_row_ptr_[dim] = static_cast<index_t>(u_col_ind_.size());

    metrics_.l_nnz = l_values_.size();
    metrics_.u_nnz = u_values_.size() + dim; // plus dim diagonal entries
    metrics_.fill_in_ratio = (metrics_.orig_nnz > 0)
        ? (static_cast<double>(metrics_.l_nnz + metrics_.u_nnz) / metrics_.orig_nnz)
        : 1.0;

    timer.stop();
    metrics_.factorization_ms = timer.elapsed_milliseconds();
    is_factorized_ = true;

    return Status::OK();
}

std::vector<scalar_t> SparseLU::solve_ftran(const std::vector<scalar_t>& b) const {
    if (!is_factorized_ || static_cast<index_t>(b.size()) != dim_) {
        return {};
    }

    // Solve B x = b <=> P^T L U Q^T x = b <=> L (U Q^T x) = P b
    // 1. Permute RHS: y_step[k] = b[p_row_[k]]
    std::vector<scalar_t> y(dim_);
    for (index_t k = 0; k < dim_; ++k) {
        y[k] = b[p_row_[k]];
    }

    // 2. Forward substitution with unit lower triangular L (stored by column k)
    for (index_t k = 0; k < dim_; ++k) {
        scalar_t yk = y[k];
        if (std::abs(yk) < 1e-15) continue;

        index_t start = l_col_ptr_[k];
        index_t end = l_col_ptr_[k + 1];
        for (index_t p = start; p < end; ++p) {
            index_t r_orig = l_row_ind_[p];
            index_t r_step = p_row_inv_[r_orig];
            y[r_step] -= l_values_[p] * yk;
        }
    }

    // 3. Backward substitution with upper triangular U (stored by row k): U z = y
    std::vector<scalar_t> z(dim_);
    for (index_t k = dim_ - 1; k >= 0; --k) {
        scalar_t sum = y[k];
        index_t start = u_row_ptr_[k];
        index_t end = u_row_ptr_[k + 1];
        for (index_t p = start; p < end; ++p) {
            index_t c_orig = u_col_ind_[p];
            index_t c_step = q_col_inv_[c_orig];
            sum -= u_values_[p] * z[c_step];
        }
        z[k] = sum / u_diag_[k];
    }

    // 4. Inverse column permutation: x[q_col_[k]] = z[k]
    std::vector<scalar_t> x(dim_);
    for (index_t k = 0; k < dim_; ++k) {
        x[q_col_[k]] = z[k];
    }

    return x;
}

std::vector<scalar_t> SparseLU::solve_btran(const std::vector<scalar_t>& b) const {
    if (!is_factorized_ || static_cast<index_t>(b.size()) != dim_) {
        return {};
    }

    // Solve B^T y = b <=> Q U^T L^T P y = b
    // 1. Column permutation: w[k] = b[q_col_[k]]
    std::vector<scalar_t> w(dim_);
    for (index_t k = 0; k < dim_; ++k) {
        w[k] = b[q_col_[k]];
    }

    // 2. Forward solve on U^T: U^T v = w (U row k is column k of U^T)
    std::vector<scalar_t> v(dim_, 0.0);
    for (index_t k = 0; k < dim_; ++k) {
        scalar_t vk = (w[k] - v[k]) / u_diag_[k];
        v[k] = vk;

        index_t start = u_row_ptr_[k];
        index_t end = u_row_ptr_[k + 1];
        for (index_t p = start; p < end; ++p) {
            index_t c_orig = u_col_ind_[p];
            index_t c_step = q_col_inv_[c_orig];
            v[c_step] += u_values_[p] * vk;
        }
    }

    // 3. Backward solve on L^T: L^T z = v
    std::vector<scalar_t> z = v;
    for (index_t k = dim_ - 1; k >= 0; --k) {
        index_t start = l_col_ptr_[k];
        index_t end = l_col_ptr_[k + 1];
        for (index_t p = start; p < end; ++p) {
            index_t r_orig = l_row_ind_[p];
            index_t r_step = p_row_inv_[r_orig];
            z[k] -= l_values_[p] * z[r_step];
        }
    }

    // 4. Row permutation: y[p_row_[k]] = z[k]
    std::vector<scalar_t> y_out(dim_);
    for (index_t k = 0; k < dim_; ++k) {
        y_out[p_row_[k]] = z[k];
    }

    return y_out;
}

} // namespace pipepye::factorization
