#include <pipepye/sparse/coo_matrix.hpp>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace pipepye::sparse {

COOMatrix::COOMatrix(index_t num_rows, index_t num_cols, size_t initial_capacity)
    : num_rows_(num_rows), num_cols_(num_cols) {
    if (num_rows < 0 || num_cols < 0) {
        throw std::invalid_argument("COOMatrix dimensions cannot be negative: (" +
                                    std::to_string(num_rows) + "x" + std::to_string(num_cols) + ")");
    }
    if (initial_capacity > 0) {
        triplets_.reserve(initial_capacity);
    }
}

COOMatrix::COOMatrix(index_t num_rows, index_t num_cols, std::vector<TripletF64> triplets)
    : num_rows_(num_rows), num_cols_(num_cols), triplets_(std::move(triplets)) {
    if (num_rows < 0 || num_cols < 0) {
        throw std::invalid_argument("COOMatrix dimensions cannot be negative");
    }
    for (const auto& t : triplets_) {
        check_bounds(t.row, t.col);
    }
    is_sorted_ = false;
    has_duplicates_ = true;
}

COOMatrix::COOMatrix(index_t num_rows, index_t num_cols,
                     std::span<const index_t> rows,
                     std::span<const index_t> cols,
                     std::span<const scalar_t> vals)
    : num_rows_(num_rows), num_cols_(num_cols) {
    if (num_rows < 0 || num_cols < 0) {
        throw std::invalid_argument("COOMatrix dimensions cannot be negative");
    }
    if (rows.size() != cols.size() || rows.size() != vals.size()) {
        throw std::invalid_argument("COOMatrix parallel spans must have identical sizes: rows=" +
                                    std::to_string(rows.size()) + ", cols=" +
                                    std::to_string(cols.size()) + ", vals=" +
                                    std::to_string(vals.size()));
    }
    triplets_.reserve(rows.size());
    for (size_t i = 0; i < rows.size(); ++i) {
        check_bounds(rows[i], cols[i]);
        triplets_.emplace_back(rows[i], cols[i], vals[i]);
    }
    is_sorted_ = false;
    has_duplicates_ = true;
}

void COOMatrix::set_dimensions(index_t num_rows, index_t num_cols) {
    if (num_rows < 0 || num_cols < 0) {
        throw std::invalid_argument("COOMatrix dimensions cannot be negative");
    }
    for (const auto& t : triplets_) {
        if (t.row >= num_rows || t.col >= num_cols) {
            throw std::out_of_range("Existing triplet (" + std::to_string(t.row) + ", " +
                                    std::to_string(t.col) + ") lies outside new dimensions (" +
                                    std::to_string(num_rows) + "x" + std::to_string(num_cols) + ")");
        }
    }
    num_rows_ = num_rows;
    num_cols_ = num_cols;
}

void COOMatrix::add_entry(index_t row, index_t col, scalar_t value) {
    check_bounds(row, col);
    triplets_.emplace_back(row, col, value);
    is_sorted_ = false;
    has_duplicates_ = true;
}

void COOMatrix::add_entry_unchecked(index_t row, index_t col, scalar_t value) {
    if (row + 1 > num_rows_) num_rows_ = row + 1;
    if (col + 1 > num_cols_) num_cols_ = col + 1;
    triplets_.emplace_back(row, col, value);
    is_sorted_ = false;
    has_duplicates_ = true;
}

void COOMatrix::reserve(size_t nnz) {
    triplets_.reserve(nnz);
}

void COOMatrix::clear() noexcept {
    triplets_.clear();
    is_sorted_ = true;
    has_duplicates_ = false;
}

void COOMatrix::sort(StorageOrder order) {
    if (order == StorageOrder::RowMajor) {
        std::stable_sort(triplets_.begin(), triplets_.end(), TripletF64::RowMajorLess{});
    } else {
        std::stable_sort(triplets_.begin(), triplets_.end(), TripletF64::ColMajorLess{});
    }
    is_sorted_ = true;
    storage_order_ = order;
}

void COOMatrix::sum_duplicates() {
    if (triplets_.size() <= 1) {
        has_duplicates_ = false;
        return;
    }
    if (!is_sorted_) {
        sort(storage_order_);
    }

    std::vector<TripletF64> deduped;
    deduped.reserve(triplets_.size());
    for (const auto& t : triplets_) {
        if (!deduped.empty() && deduped.back().row == t.row && deduped.back().col == t.col) {
            deduped.back().val += t.val;
        } else {
            deduped.push_back(t);
        }
    }
    triplets_ = std::move(deduped);
    has_duplicates_ = false;
}

void COOMatrix::drop_zeros(scalar_t tolerance) {
    std::erase_if(triplets_, [tolerance](const TripletF64& t) {
        return std::abs(t.val) <= tolerance;
    });
}

CSRMatrix COOMatrix::to_csr() const {
    if (num_rows_ == 0 || num_cols_ == 0) {
        return CSRMatrix(num_rows_, num_cols_, std::vector<index_t>(num_rows_ + 1, 0), {}, {});
    }

    std::vector<TripletF64> working_triplets;
    const std::vector<TripletF64>* p_triplets = &triplets_;

    if (!is_sorted_ || storage_order_ != StorageOrder::RowMajor || has_duplicates_) {
        working_triplets = triplets_;
        std::stable_sort(working_triplets.begin(), working_triplets.end(), TripletF64::RowMajorLess{});

        std::vector<TripletF64> deduped;
        deduped.reserve(working_triplets.size());
        for (const auto& t : working_triplets) {
            if (!deduped.empty() && deduped.back().row == t.row && deduped.back().col == t.col) {
                deduped.back().val += t.val;
            } else {
                deduped.push_back(t);
            }
        }
        working_triplets = std::move(deduped);
        p_triplets = &working_triplets;
    }

    std::vector<index_t> row_ptr(num_rows_ + 1, 0);
    for (const auto& t : *p_triplets) {
        row_ptr[t.row + 1]++;
    }
    for (index_t i = 0; i < num_rows_; ++i) {
        row_ptr[i + 1] += row_ptr[i];
    }

    std::vector<index_t> col_ind(p_triplets->size());
    std::vector<scalar_t> values(p_triplets->size());

    for (size_t k = 0; k < p_triplets->size(); ++k) {
        col_ind[k] = (*p_triplets)[k].col;
        values[k] = (*p_triplets)[k].val;
    }

    return CSRMatrix(num_rows_, num_cols_, std::move(row_ptr), std::move(col_ind), std::move(values));
}

CSCMatrix COOMatrix::to_csc() const {
    if (num_rows_ == 0 || num_cols_ == 0) {
        return CSCMatrix(num_rows_, num_cols_, std::vector<index_t>(num_cols_ + 1, 0), {}, {});
    }

    std::vector<TripletF64> working_triplets;
    const std::vector<TripletF64>* p_triplets = &triplets_;

    if (!is_sorted_ || storage_order_ != StorageOrder::ColMajor || has_duplicates_) {
        working_triplets = triplets_;
        std::stable_sort(working_triplets.begin(), working_triplets.end(), TripletF64::ColMajorLess{});

        std::vector<TripletF64> deduped;
        deduped.reserve(working_triplets.size());
        for (const auto& t : working_triplets) {
            if (!deduped.empty() && deduped.back().row == t.row && deduped.back().col == t.col) {
                deduped.back().val += t.val;
            } else {
                deduped.push_back(t);
            }
        }
        working_triplets = std::move(deduped);
        p_triplets = &working_triplets;
    }

    std::vector<index_t> col_ptr(num_cols_ + 1, 0);
    for (const auto& t : *p_triplets) {
        col_ptr[t.col + 1]++;
    }
    for (index_t j = 0; j < num_cols_; ++j) {
        col_ptr[j + 1] += col_ptr[j];
    }

    std::vector<index_t> row_ind(p_triplets->size());
    std::vector<scalar_t> values(p_triplets->size());

    for (size_t k = 0; k < p_triplets->size(); ++k) {
        row_ind[k] = (*p_triplets)[k].row;
        values[k] = (*p_triplets)[k].val;
    }

    return CSCMatrix(num_rows_, num_cols_, std::move(col_ptr), std::move(row_ind), std::move(values));
}

std::vector<scalar_t> COOMatrix::to_dense() const {
    std::vector<scalar_t> dense(static_cast<size_t>(num_rows_) * static_cast<size_t>(num_cols_), 0.0);
    for (const auto& t : triplets_) {
        dense[static_cast<size_t>(t.row) * static_cast<size_t>(num_cols_) + static_cast<size_t>(t.col)] += t.val;
    }
    return dense;
}

scalar_t COOMatrix::coeff(index_t row, index_t col) const {
    check_bounds(row, col);
    scalar_t sum = 0.0;
    for (const auto& t : triplets_) {
        if (t.row == row && t.col == col) {
            sum += t.val;
        }
    }
    return sum;
}

void COOMatrix::spmv(scalar_t alpha, ConstVectorView x, scalar_t beta, MutableVectorView y) const {
    if (x.size() != num_cols_) {
        throw std::invalid_argument("COOMatrix::spmv x size mismatch: expected " +
                                    std::to_string(num_cols_) + ", got " + std::to_string(x.size()));
    }
    if (y.size() != num_rows_) {
        throw std::invalid_argument("COOMatrix::spmv y size mismatch: expected " +
                                    std::to_string(num_rows_) + ", got " + std::to_string(y.size()));
    }

    if (beta == 0.0) {
        y.fill(0.0);
    } else if (beta != 1.0) {
        y.scale(beta);
    }

    if (alpha == 0.0) return;

    for (const auto& t : triplets_) {
        y[t.row] += alpha * t.val * x[t.col];
    }
}

void COOMatrix::spmv_transpose(scalar_t alpha, ConstVectorView x, scalar_t beta, MutableVectorView y) const {
    if (x.size() != num_rows_) {
        throw std::invalid_argument("COOMatrix::spmv_transpose x size mismatch: expected " +
                                    std::to_string(num_rows_) + ", got " + std::to_string(x.size()));
    }
    if (y.size() != num_cols_) {
        throw std::invalid_argument("COOMatrix::spmv_transpose y size mismatch: expected " +
                                    std::to_string(num_cols_) + ", got " + std::to_string(y.size()));
    }

    if (beta == 0.0) {
        y.fill(0.0);
    } else if (beta != 1.0) {
        y.scale(beta);
    }

    if (alpha == 0.0) return;

    for (const auto& t : triplets_) {
        y[t.col] += alpha * t.val * x[t.row];
    }
}

scalar_t COOMatrix::norm_inf() const {
    if (num_rows_ == 0 || num_cols_ == 0 || triplets_.empty()) return 0.0;
    if (has_duplicates_) {
        return to_csr().norm_inf();
    }
    std::vector<scalar_t> row_sums(num_rows_, 0.0);
    for (const auto& t : triplets_) {
        row_sums[t.row] += std::abs(t.val);
    }
    scalar_t max_sum = 0.0;
    for (scalar_t s : row_sums) {
        if (s > max_sum) max_sum = s;
    }
    return max_sum;
}

scalar_t COOMatrix::norm_1() const {
    if (num_rows_ == 0 || num_cols_ == 0 || triplets_.empty()) return 0.0;
    if (has_duplicates_) {
        return to_csc().norm_1();
    }
    std::vector<scalar_t> col_sums(num_cols_, 0.0);
    for (const auto& t : triplets_) {
        col_sums[t.col] += std::abs(t.val);
    }
    scalar_t max_sum = 0.0;
    for (scalar_t s : col_sums) {
        if (s > max_sum) max_sum = s;
    }
    return max_sum;
}

scalar_t COOMatrix::norm_frobenius() const {
    if (triplets_.empty()) return 0.0;
    if (has_duplicates_) {
        return to_csr().norm_frobenius();
    }
    scalar_t sum_sq = 0.0;
    for (const auto& t : triplets_) {
        sum_sq += t.val * t.val;
    }
    return std::sqrt(sum_sq);
}

} // namespace pipepye::sparse
