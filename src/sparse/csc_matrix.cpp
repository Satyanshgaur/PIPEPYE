#include <pipepye/sparse/csc_matrix.hpp>
#include <pipepye/sparse/csr_matrix.hpp>
#include <pipepye/sparse/coo_matrix.hpp>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace pipepye::sparse {

CSCMatrix::CSCMatrix(index_t num_rows, index_t num_cols,
                     std::vector<index_t> col_ptr,
                     std::vector<index_t> row_ind,
                     std::vector<scalar_t> values)
    : num_rows_(num_rows), num_cols_(num_cols),
      col_ptr_(std::move(col_ptr)),
      row_ind_(std::move(row_ind)),
      values_(std::move(values)) {
    validate();
}

void CSCMatrix::validate() const {
    if (num_rows_ < 0 || num_cols_ < 0) {
        throw std::invalid_argument("CSCMatrix dimensions cannot be negative: (" +
                                    std::to_string(num_rows_) + "x" + std::to_string(num_cols_) + ")");
    }
    if (col_ptr_.empty()) {
        throw std::invalid_argument("CSCMatrix col_ptr cannot be empty");
    }
    if (static_cast<index_t>(col_ptr_.size()) != num_cols_ + 1) {
        throw std::invalid_argument("CSCMatrix col_ptr size must equal num_cols + 1: got " +
                                    std::to_string(col_ptr_.size()) + ", expected " +
                                    std::to_string(num_cols_ + 1));
    }
    if (col_ptr_[0] != 0) {
        throw std::invalid_argument("CSCMatrix col_ptr[0] must be 0, got " + std::to_string(col_ptr_[0]));
    }
    if (row_ind_.size() != values_.size()) {
        throw std::invalid_argument("CSCMatrix row_ind and values size mismatch: row_ind=" +
                                    std::to_string(row_ind_.size()) + ", values=" +
                                    std::to_string(values_.size()));
    }
    if (col_ptr_.back() != static_cast<index_t>(values_.size())) {
        throw std::invalid_argument("CSCMatrix col_ptr.back() must equal values.size(): col_ptr.back()=" +
                                    std::to_string(col_ptr_.back()) + ", values.size()=" +
                                    std::to_string(values_.size()));
    }

    for (index_t j = 0; j < num_cols_; ++j) {
        if (col_ptr_[j] > col_ptr_[j + 1]) {
            throw std::invalid_argument("CSCMatrix col_ptr is not monotonic at col " + std::to_string(j));
        }
        index_t start = col_ptr_[j];
        index_t end = col_ptr_[j + 1];
        for (index_t k = start; k < end; ++k) {
            if (row_ind_[k] < 0 || row_ind_[k] >= num_rows_) {
                throw std::invalid_argument("CSCMatrix row index out of bounds: " +
                                            std::to_string(row_ind_[k]) + " for num_rows " +
                                            std::to_string(num_rows_));
            }
            if (k > start && row_ind_[k] <= row_ind_[k - 1]) {
                throw std::invalid_argument("CSCMatrix row indices in col " + std::to_string(j) +
                                            " are not strictly ascending: " + std::to_string(row_ind_[k - 1]) +
                                            " followed by " + std::to_string(row_ind_[k]));
            }
        }
    }
}

CSCMatrix::ColView CSCMatrix::col(index_t j) const {
    if (j < 0 || j >= num_cols_) {
        throw std::out_of_range("CSCMatrix::col index out of range: " + std::to_string(j) +
                                " for num_cols=" + std::to_string(num_cols_));
    }
    index_t start = col_ptr_[j];
    index_t end = col_ptr_[j + 1];
    return ColView{
        .row_indices = std::span<const index_t>(row_ind_.data() + start, end - start),
        .values = std::span<const scalar_t>(values_.data() + start, end - start)
    };
}

index_t CSCMatrix::col_nnz(index_t j) const {
    if (j < 0 || j >= num_cols_) {
        throw std::out_of_range("CSCMatrix::col_nnz index out of range: " + std::to_string(j) +
                                " for num_cols=" + std::to_string(num_cols_));
    }
    return col_ptr_[j + 1] - col_ptr_[j];
}

scalar_t CSCMatrix::coeff(index_t row, index_t col) const {
    check_bounds(row, col);
    index_t start = col_ptr_[col];
    index_t end = col_ptr_[col + 1];
    auto it = std::lower_bound(row_ind_.begin() + start, row_ind_.begin() + end, row);
    if (it != row_ind_.begin() + end && *it == row) {
        return values_[std::distance(row_ind_.begin(), it)];
    }
    return 0.0;
}

CSRMatrix CSCMatrix::to_csr() const {
    if (num_rows_ == 0 || num_cols_ == 0 || values_.empty()) {
        return CSRMatrix(num_rows_, num_cols_, std::vector<index_t>(num_rows_ + 1, 0), {}, {});
    }

    std::vector<index_t> row_ptr(num_rows_ + 1, 0);
    for (index_t row : row_ind_) {
        row_ptr[row + 1]++;
    }
    for (index_t i = 0; i < num_rows_; ++i) {
        row_ptr[i + 1] += row_ptr[i];
    }

    std::vector<index_t> col_ind(values_.size());
    std::vector<scalar_t> csr_vals(values_.size());
    std::vector<index_t> current_row_pos = row_ptr;

    for (index_t j = 0; j < num_cols_; ++j) {
        index_t start = col_ptr_[j];
        index_t end = col_ptr_[j + 1];
        for (index_t k = start; k < end; ++k) {
            index_t i = row_ind_[k];
            index_t dest = current_row_pos[i]++;
            col_ind[dest] = j;
            csr_vals[dest] = values_[k];
        }
    }

    return CSRMatrix(num_rows_, num_cols_, std::move(row_ptr), std::move(col_ind), std::move(csr_vals));
}

COOMatrix CSCMatrix::to_coo() const {
    std::vector<TripletF64> triplets;
    triplets.reserve(values_.size());
    for (index_t j = 0; j < num_cols_; ++j) {
        index_t start = col_ptr_[j];
        index_t end = col_ptr_[j + 1];
        for (index_t k = start; k < end; ++k) {
            triplets.emplace_back(row_ind_[k], j, values_[k]);
        }
    }
    COOMatrix coo(num_rows_, num_cols_, std::move(triplets));
    coo.sort(StorageOrder::ColMajor);
    return coo;
}

std::vector<scalar_t> CSCMatrix::to_dense() const {
    std::vector<scalar_t> dense(static_cast<size_t>(num_rows_) * static_cast<size_t>(num_cols_), 0.0);
    for (index_t j = 0; j < num_cols_; ++j) {
        index_t start = col_ptr_[j];
        index_t end = col_ptr_[j + 1];
        for (index_t k = start; k < end; ++k) {
            dense[static_cast<size_t>(row_ind_[k]) * static_cast<size_t>(num_cols_) + static_cast<size_t>(j)] = values_[k];
        }
    }
    return dense;
}

void CSCMatrix::spmv(scalar_t alpha, ConstVectorView x, scalar_t beta, MutableVectorView y) const {
    if (x.size() != num_cols_) {
        throw std::invalid_argument("CSCMatrix::spmv x size mismatch: expected " +
                                    std::to_string(num_cols_) + ", got " + std::to_string(x.size()));
    }
    if (y.size() != num_rows_) {
        throw std::invalid_argument("CSCMatrix::spmv y size mismatch: expected " +
                                    std::to_string(num_rows_) + ", got " + std::to_string(y.size()));
    }

    if (beta == 0.0) {
        y.fill(0.0);
    } else if (beta != 1.0) {
        y.scale(beta);
    }

    if (alpha == 0.0) return;

    for (index_t j = 0; j < num_cols_; ++j) {
        scalar_t xj = x[j];
        if (xj == 0.0) continue;
        scalar_t scaled_xj = alpha * xj;
        index_t start = col_ptr_[j];
        index_t end = col_ptr_[j + 1];
        for (index_t k = start; k < end; ++k) {
            y[row_ind_[k]] += scaled_xj * values_[k];
        }
    }
}

void CSCMatrix::spmv_transpose(scalar_t alpha, ConstVectorView x, scalar_t beta, MutableVectorView y) const {
    if (x.size() != num_rows_) {
        throw std::invalid_argument("CSCMatrix::spmv_transpose x size mismatch: expected " +
                                    std::to_string(num_rows_) + ", got " + std::to_string(x.size()));
    }
    if (y.size() != num_cols_) {
        throw std::invalid_argument("CSCMatrix::spmv_transpose y size mismatch: expected " +
                                    std::to_string(num_cols_) + ", got " + std::to_string(y.size()));
    }

    for (index_t j = 0; j < num_cols_; ++j) {
        scalar_t col_dot = 0.0;
        index_t start = col_ptr_[j];
        index_t end = col_ptr_[j + 1];
        for (index_t k = start; k < end; ++k) {
            col_dot += values_[k] * x[row_ind_[k]];
        }
        if (beta == 0.0) {
            y[j] = alpha * col_dot;
        } else {
            y[j] = alpha * col_dot + beta * y[j];
        }
    }
}

scalar_t CSCMatrix::norm_1() const {
    if (num_rows_ == 0 || num_cols_ == 0 || values_.empty()) return 0.0;
    scalar_t max_sum = 0.0;
    for (index_t j = 0; j < num_cols_; ++j) {
        scalar_t col_sum = 0.0;
        index_t start = col_ptr_[j];
        index_t end = col_ptr_[j + 1];
        for (index_t k = start; k < end; ++k) {
            col_sum += std::abs(values_[k]);
        }
        if (col_sum > max_sum) max_sum = col_sum;
    }
    return max_sum;
}

scalar_t CSCMatrix::norm_inf() const {
    if (num_rows_ == 0 || num_cols_ == 0 || values_.empty()) return 0.0;
    std::vector<scalar_t> row_sums(num_rows_, 0.0);
    for (size_t k = 0; k < values_.size(); ++k) {
        row_sums[row_ind_[k]] += std::abs(values_[k]);
    }
    scalar_t max_sum = 0.0;
    for (index_t i = 0; i < num_rows_; ++i) {
        if (row_sums[i] > max_sum) max_sum = row_sums[i];
    }
    return max_sum;
}

scalar_t CSCMatrix::norm_frobenius() const {
    scalar_t sum_sq = 0.0;
    for (scalar_t v : values_) {
        sum_sq += v * v;
    }
    return std::sqrt(sum_sq);
}

} // namespace pipepye::sparse
