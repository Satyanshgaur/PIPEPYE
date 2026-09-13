#include <pipepye/sparse/csr_matrix.hpp>
#include <pipepye/sparse/csc_matrix.hpp>
#include <pipepye/sparse/coo_matrix.hpp>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace pipepye::sparse {

CSRMatrix::CSRMatrix(index_t num_rows, index_t num_cols,
                     std::vector<index_t> row_ptr,
                     std::vector<index_t> col_ind,
                     std::vector<scalar_t> values)
    : num_rows_(num_rows), num_cols_(num_cols),
      row_ptr_(std::move(row_ptr)),
      col_ind_(std::move(col_ind)),
      values_(std::move(values)) {
    validate();
}

void CSRMatrix::validate() const {
    if (num_rows_ < 0 || num_cols_ < 0) {
        throw std::invalid_argument("CSRMatrix dimensions cannot be negative: (" +
                                    std::to_string(num_rows_) + "x" + std::to_string(num_cols_) + ")");
    }
    if (row_ptr_.empty()) {
        throw std::invalid_argument("CSRMatrix row_ptr cannot be empty");
    }
    if (static_cast<index_t>(row_ptr_.size()) != num_rows_ + 1) {
        throw std::invalid_argument("CSRMatrix row_ptr size must equal num_rows + 1: got " +
                                    std::to_string(row_ptr_.size()) + ", expected " +
                                    std::to_string(num_rows_ + 1));
    }
    if (row_ptr_[0] != 0) {
        throw std::invalid_argument("CSRMatrix row_ptr[0] must be 0, got " + std::to_string(row_ptr_[0]));
    }
    if (col_ind_.size() != values_.size()) {
        throw std::invalid_argument("CSRMatrix col_ind and values size mismatch: col_ind=" +
                                    std::to_string(col_ind_.size()) + ", values=" +
                                    std::to_string(values_.size()));
    }
    if (row_ptr_.back() != static_cast<index_t>(values_.size())) {
        throw std::invalid_argument("CSRMatrix row_ptr.back() must equal values.size(): row_ptr.back()=" +
                                    std::to_string(row_ptr_.back()) + ", values.size()=" +
                                    std::to_string(values_.size()));
    }

    for (index_t i = 0; i < num_rows_; ++i) {
        if (row_ptr_[i] > row_ptr_[i + 1]) {
            throw std::invalid_argument("CSRMatrix row_ptr is not monotonic at row " + std::to_string(i));
        }
        index_t start = row_ptr_[i];
        index_t end = row_ptr_[i + 1];
        for (index_t k = start; k < end; ++k) {
            if (col_ind_[k] < 0 || col_ind_[k] >= num_cols_) {
                throw std::invalid_argument("CSRMatrix column index out of bounds: " +
                                            std::to_string(col_ind_[k]) + " for num_cols " +
                                            std::to_string(num_cols_));
            }
            if (k > start && col_ind_[k] <= col_ind_[k - 1]) {
                throw std::invalid_argument("CSRMatrix column indices in row " + std::to_string(i) +
                                            " are not strictly ascending: " + std::to_string(col_ind_[k - 1]) +
                                            " followed by " + std::to_string(col_ind_[k]));
            }
        }
    }
}

CSRMatrix::RowView CSRMatrix::row(index_t i) const {
    if (i < 0 || i >= num_rows_) {
        throw std::out_of_range("CSRMatrix::row index out of range: " + std::to_string(i) +
                                " for num_rows=" + std::to_string(num_rows_));
    }
    index_t start = row_ptr_[i];
    index_t end = row_ptr_[i + 1];
    return RowView{
        .col_indices = std::span<const index_t>(col_ind_.data() + start, end - start),
        .values = std::span<const scalar_t>(values_.data() + start, end - start)
    };
}

index_t CSRMatrix::row_nnz(index_t i) const {
    if (i < 0 || i >= num_rows_) {
        throw std::out_of_range("CSRMatrix::row_nnz index out of range: " + std::to_string(i) +
                                " for num_rows=" + std::to_string(num_rows_));
    }
    return row_ptr_[i + 1] - row_ptr_[i];
}

scalar_t CSRMatrix::coeff(index_t row, index_t col) const {
    check_bounds(row, col);
    index_t start = row_ptr_[row];
    index_t end = row_ptr_[row + 1];
    auto it = std::lower_bound(col_ind_.begin() + start, col_ind_.begin() + end, col);
    if (it != col_ind_.begin() + end && *it == col) {
        return values_[std::distance(col_ind_.begin(), it)];
    }
    return 0.0;
}

CSCMatrix CSRMatrix::to_csc() const {
    if (num_rows_ == 0 || num_cols_ == 0 || values_.empty()) {
        return CSCMatrix(num_rows_, num_cols_, std::vector<index_t>(num_cols_ + 1, 0), {}, {});
    }

    std::vector<index_t> col_ptr(num_cols_ + 1, 0);
    for (index_t col : col_ind_) {
        col_ptr[col + 1]++;
    }
    for (index_t j = 0; j < num_cols_; ++j) {
        col_ptr[j + 1] += col_ptr[j];
    }

    std::vector<index_t> row_ind(values_.size());
    std::vector<scalar_t> csc_vals(values_.size());
    std::vector<index_t> current_col_pos = col_ptr;

    for (index_t i = 0; i < num_rows_; ++i) {
        index_t start = row_ptr_[i];
        index_t end = row_ptr_[i + 1];
        for (index_t k = start; k < end; ++k) {
            index_t j = col_ind_[k];
            index_t dest = current_col_pos[j]++;
            row_ind[dest] = i;
            csc_vals[dest] = values_[k];
        }
    }

    return CSCMatrix(num_rows_, num_cols_, std::move(col_ptr), std::move(row_ind), std::move(csc_vals));
}

COOMatrix CSRMatrix::to_coo() const {
    std::vector<TripletF64> triplets;
    triplets.reserve(values_.size());
    for (index_t i = 0; i < num_rows_; ++i) {
        index_t start = row_ptr_[i];
        index_t end = row_ptr_[i + 1];
        for (index_t k = start; k < end; ++k) {
            triplets.emplace_back(i, col_ind_[k], values_[k]);
        }
    }
    COOMatrix coo(num_rows_, num_cols_, std::move(triplets));
    coo.sort(StorageOrder::RowMajor);
    return coo;
}

std::vector<scalar_t> CSRMatrix::to_dense() const {
    std::vector<scalar_t> dense(static_cast<size_t>(num_rows_) * static_cast<size_t>(num_cols_), 0.0);
    for (index_t i = 0; i < num_rows_; ++i) {
        index_t start = row_ptr_[i];
        index_t end = row_ptr_[i + 1];
        for (index_t k = start; k < end; ++k) {
            dense[static_cast<size_t>(i) * static_cast<size_t>(num_cols_) + static_cast<size_t>(col_ind_[k])] = values_[k];
        }
    }
    return dense;
}

void CSRMatrix::spmv(scalar_t alpha, ConstVectorView x, scalar_t beta, MutableVectorView y) const {
    if (x.size() != num_cols_) {
        throw std::invalid_argument("CSRMatrix::spmv x size mismatch: expected " +
                                    std::to_string(num_cols_) + ", got " + std::to_string(x.size()));
    }
    if (y.size() != num_rows_) {
        throw std::invalid_argument("CSRMatrix::spmv y size mismatch: expected " +
                                    std::to_string(num_rows_) + ", got " + std::to_string(y.size()));
    }

    for (index_t i = 0; i < num_rows_; ++i) {
        scalar_t row_dot = 0.0;
        index_t start = row_ptr_[i];
        index_t end = row_ptr_[i + 1];
        for (index_t k = start; k < end; ++k) {
            row_dot += values_[k] * x[col_ind_[k]];
        }
        if (beta == 0.0) {
            y[i] = alpha * row_dot;
        } else {
            y[i] = alpha * row_dot + beta * y[i];
        }
    }
}

void CSRMatrix::spmv_transpose(scalar_t alpha, ConstVectorView x, scalar_t beta, MutableVectorView y) const {
    if (x.size() != num_rows_) {
        throw std::invalid_argument("CSRMatrix::spmv_transpose x size mismatch: expected " +
                                    std::to_string(num_rows_) + ", got " + std::to_string(x.size()));
    }
    if (y.size() != num_cols_) {
        throw std::invalid_argument("CSRMatrix::spmv_transpose y size mismatch: expected " +
                                    std::to_string(num_cols_) + ", got " + std::to_string(y.size()));
    }

    if (beta == 0.0) {
        y.fill(0.0);
    } else if (beta != 1.0) {
        y.scale(beta);
    }

    if (alpha == 0.0) return;

    for (index_t i = 0; i < num_rows_; ++i) {
        scalar_t xi = x[i];
        if (xi == 0.0) continue;
        scalar_t scaled_xi = alpha * xi;
        index_t start = row_ptr_[i];
        index_t end = row_ptr_[i + 1];
        for (index_t k = start; k < end; ++k) {
            y[col_ind_[k]] += scaled_xi * values_[k];
        }
    }
}

scalar_t CSRMatrix::norm_inf() const {
    if (num_rows_ == 0 || num_cols_ == 0 || values_.empty()) return 0.0;
    scalar_t max_sum = 0.0;
    for (index_t i = 0; i < num_rows_; ++i) {
        scalar_t row_sum = 0.0;
        index_t start = row_ptr_[i];
        index_t end = row_ptr_[i + 1];
        for (index_t k = start; k < end; ++k) {
            row_sum += std::abs(values_[k]);
        }
        if (row_sum > max_sum) max_sum = row_sum;
    }
    return max_sum;
}

scalar_t CSRMatrix::norm_1() const {
    if (num_rows_ == 0 || num_cols_ == 0 || values_.empty()) return 0.0;
    std::vector<scalar_t> col_sums(num_cols_, 0.0);
    for (size_t k = 0; k < values_.size(); ++k) {
        col_sums[col_ind_[k]] += std::abs(values_[k]);
    }
    scalar_t max_sum = 0.0;
    for (index_t j = 0; j < num_cols_; ++j) {
        if (col_sums[j] > max_sum) max_sum = col_sums[j];
    }
    return max_sum;
}

scalar_t CSRMatrix::norm_frobenius() const {
    scalar_t sum_sq = 0.0;
    for (scalar_t v : values_) {
        sum_sq += v * v;
    }
    return std::sqrt(sum_sq);
}

} // namespace pipepye::sparse
