#include <pipepye/sparse/dense_matrix.hpp>
#include <pipepye/sparse/csr_matrix.hpp>
#include <pipepye/sparse/csc_matrix.hpp>
#include <pipepye/sparse/coo_matrix.hpp>
#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <string>

namespace pipepye::sparse {

DenseMatrix::DenseMatrix(index_t num_rows, index_t num_cols, scalar_t init_val)
    : num_rows_(num_rows), num_cols_(num_cols),
      data_(static_cast<size_t>(num_rows) * static_cast<size_t>(num_cols), init_val) {
    if (num_rows < 0 || num_cols < 0) {
        throw std::invalid_argument("DenseMatrix dimensions cannot be negative");
    }
}

DenseMatrix::DenseMatrix(index_t num_rows, index_t num_cols, std::vector<scalar_t> data)
    : num_rows_(num_rows), num_cols_(num_cols), data_(std::move(data)) {
    if (num_rows < 0 || num_cols < 0) {
        throw std::invalid_argument("DenseMatrix dimensions cannot be negative");
    }
    if (data_.size() != static_cast<size_t>(num_rows) * static_cast<size_t>(num_cols)) {
        throw std::invalid_argument("DenseMatrix data size mismatch: got " +
                                    std::to_string(data_.size()) + ", expected " +
                                    std::to_string(num_rows * num_cols));
    }
}

DenseMatrix::DenseMatrix(index_t num_rows, index_t num_cols, std::initializer_list<scalar_t> data)
    : num_rows_(num_rows), num_cols_(num_cols), data_(data) {
    if (num_rows < 0 || num_cols < 0) {
        throw std::invalid_argument("DenseMatrix dimensions cannot be negative");
    }
    if (data_.size() != static_cast<size_t>(num_rows) * static_cast<size_t>(num_cols)) {
        throw std::invalid_argument("DenseMatrix data size mismatch: got " +
                                    std::to_string(data_.size()) + ", expected " +
                                    std::to_string(num_rows * num_cols));
    }
}

size_t DenseMatrix::num_nonzeros() const noexcept {
    size_t count = 0;
    for (scalar_t v : data_) {
        if (v != 0.0) ++count;
    }
    return count;
}

ConstVectorView DenseMatrix::row(index_t i) const {
    if (i < 0 || i >= num_rows_) {
        throw std::out_of_range("DenseMatrix::row index out of bounds: " + std::to_string(i));
    }
    return ConstVectorView(data_.data() + static_cast<size_t>(i) * static_cast<size_t>(num_cols_), num_cols_);
}

MutableVectorView DenseMatrix::row(index_t i) {
    if (i < 0 || i >= num_rows_) {
        throw std::out_of_range("DenseMatrix::row index out of bounds: " + std::to_string(i));
    }
    return MutableVectorView(data_.data() + static_cast<size_t>(i) * static_cast<size_t>(num_cols_), num_cols_);
}

void DenseMatrix::gemv(scalar_t alpha, ConstVectorView x, scalar_t beta, MutableVectorView y) const {
    if (x.size() != num_cols_) {
        throw std::invalid_argument("DenseMatrix::gemv x size mismatch: expected " +
                                    std::to_string(num_cols_) + ", got " + std::to_string(x.size()));
    }
    if (y.size() != num_rows_) {
        throw std::invalid_argument("DenseMatrix::gemv y size mismatch: expected " +
                                    std::to_string(num_rows_) + ", got " + std::to_string(y.size()));
    }

    for (index_t i = 0; i < num_rows_; ++i) {
        scalar_t dot = 0.0;
        size_t row_start = static_cast<size_t>(i) * static_cast<size_t>(num_cols_);
        for (index_t j = 0; j < num_cols_; ++j) {
            dot += data_[row_start + static_cast<size_t>(j)] * x[j];
        }
        if (beta == 0.0) {
            y[i] = alpha * dot;
        } else {
            y[i] = alpha * dot + beta * y[i];
        }
    }
}

void DenseMatrix::gemv_transpose(scalar_t alpha, ConstVectorView x, scalar_t beta, MutableVectorView y) const {
    if (x.size() != num_rows_) {
        throw std::invalid_argument("DenseMatrix::gemv_transpose x size mismatch: expected " +
                                    std::to_string(num_rows_) + ", got " + std::to_string(x.size()));
    }
    if (y.size() != num_cols_) {
        throw std::invalid_argument("DenseMatrix::gemv_transpose y size mismatch: expected " +
                                    std::to_string(num_cols_) + ", got " + std::to_string(y.size()));
    }

    if (beta == 0.0) {
        y.fill(0.0);
    } else if (beta != 1.0) {
        y.scale(beta);
    }

    if (alpha == 0.0) return;

    for (index_t j = 0; j < num_cols_; ++j) {
        scalar_t dot = 0.0;
        for (index_t i = 0; i < num_rows_; ++i) {
            dot += data_[static_cast<size_t>(i) * static_cast<size_t>(num_cols_) + static_cast<size_t>(j)] * x[i];
        }
        y[j] += alpha * dot;
    }
}

scalar_t DenseMatrix::norm_inf() const {
    if (num_rows_ == 0 || num_cols_ == 0) return 0.0;
    scalar_t max_sum = 0.0;
    for (index_t i = 0; i < num_rows_; ++i) {
        scalar_t row_sum = 0.0;
        size_t row_start = static_cast<size_t>(i) * static_cast<size_t>(num_cols_);
        for (index_t j = 0; j < num_cols_; ++j) {
            row_sum += std::abs(data_[row_start + static_cast<size_t>(j)]);
        }
        if (row_sum > max_sum) max_sum = row_sum;
    }
    return max_sum;
}

scalar_t DenseMatrix::norm_1() const {
    if (num_rows_ == 0 || num_cols_ == 0) return 0.0;
    scalar_t max_sum = 0.0;
    for (index_t j = 0; j < num_cols_; ++j) {
        scalar_t col_sum = 0.0;
        for (index_t i = 0; i < num_rows_; ++i) {
            col_sum += std::abs(data_[static_cast<size_t>(i) * static_cast<size_t>(num_cols_) + static_cast<size_t>(j)]);
        }
        if (col_sum > max_sum) max_sum = col_sum;
    }
    return max_sum;
}

scalar_t DenseMatrix::norm_frobenius() const {
    scalar_t sum_sq = 0.0;
    for (scalar_t v : data_) {
        sum_sq += v * v;
    }
    return std::sqrt(sum_sq);
}

DenseMatrix DenseMatrix::matmul(const DenseMatrix& B) const {
    if (num_cols_ != B.num_rows_) {
        throw std::invalid_argument("DenseMatrix::matmul dimension mismatch: A cols (" +
                                    std::to_string(num_cols_) + ") != B rows (" +
                                    std::to_string(B.num_rows_) + ")");
    }
    DenseMatrix C(num_rows_, B.num_cols_, 0.0);
    for (index_t i = 0; i < num_rows_; ++i) {
        for (index_t k = 0; k < num_cols_; ++k) {
            scalar_t aik = data_[static_cast<size_t>(i) * static_cast<size_t>(num_cols_) + static_cast<size_t>(k)];
            if (aik == 0.0) continue;
            for (index_t j = 0; j < B.num_cols_; ++j) {
                C(i, j) += aik * B(k, j);
            }
        }
    }
    return C;
}

DenseMatrix DenseMatrix::add(const DenseMatrix& B) const {
    if (num_rows_ != B.num_rows_ || num_cols_ != B.num_cols_) {
        throw std::invalid_argument("DenseMatrix::add dimension mismatch");
    }
    DenseMatrix C(num_rows_, num_cols_);
    for (size_t k = 0; k < data_.size(); ++k) {
        C.data_[k] = data_[k] + B.data_[k];
    }
    return C;
}

CSRMatrix DenseMatrix::to_csr(scalar_t zero_tol) const {
    COOMatrix coo = to_coo(zero_tol);
    return coo.to_csr();
}

CSCMatrix DenseMatrix::to_csc(scalar_t zero_tol) const {
    COOMatrix coo = to_coo(zero_tol);
    return coo.to_csc();
}

COOMatrix DenseMatrix::to_coo(scalar_t zero_tol) const {
    COOMatrix coo(num_rows_, num_cols_);
    for (index_t i = 0; i < num_rows_; ++i) {
        size_t row_start = static_cast<size_t>(i) * static_cast<size_t>(num_cols_);
        for (index_t j = 0; j < num_cols_; ++j) {
            scalar_t v = data_[row_start + static_cast<size_t>(j)];
            if (std::abs(v) > zero_tol) {
                coo.add_entry(i, j, v);
            }
        }
    }
    return coo;
}

DenseMatrix DenseMatrix::identity(index_t n) {
    DenseMatrix I(n, n, 0.0);
    for (index_t i = 0; i < n; ++i) {
        I(i, i) = 1.0;
    }
    return I;
}

DenseMatrix DenseMatrix::diagonal(ConstVectorView diag) {
    index_t n = diag.size();
    DenseMatrix D(n, n, 0.0);
    for (index_t i = 0; i < n; ++i) {
        D(i, i) = diag[i];
    }
    return D;
}

DenseMatrix DenseMatrix::from_sparse(const ISparseMatrix& sparse) {
    DenseMatrix D(sparse.num_rows(), sparse.num_cols(), 0.0);
    for (index_t i = 0; i < sparse.num_rows(); ++i) {
        for (index_t j = 0; j < sparse.num_cols(); ++j) {
            D(i, j) = sparse.coeff(i, j);
        }
    }
    return D;
}

DenseMatrix DenseMatrix::random(index_t num_rows, index_t num_cols,
                                scalar_t min_val, scalar_t max_val,
                                uint32_t seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<scalar_t> dist(min_val, max_val);

    DenseMatrix M(num_rows, num_cols);
    for (index_t i = 0; i < num_rows; ++i) {
        for (index_t j = 0; j < num_cols; ++j) {
            M(i, j) = dist(gen);
        }
    }
    return M;
}

} // namespace pipepye::sparse
