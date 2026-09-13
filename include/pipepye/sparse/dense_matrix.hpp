#pragma once

#include <pipepye/sparse/matrix_base.hpp>
#include <pipepye/sparse/vector.hpp>
#include <span>
#include <vector>

namespace pipepye::sparse {

class CSRMatrix;
class CSCMatrix;
class COOMatrix;

/// @brief Deliberately simple, canonical dense matrix reference oracle.
/// Used as a correctness oracle for SpMV, SpMV transpose, matrix multiplication,
/// and numerical comparisons before GPU and aggressive compiler optimizations.
class DenseMatrix : public ISparseMatrix {
public:
    DenseMatrix() = default;
    DenseMatrix(index_t num_rows, index_t num_cols, scalar_t init_val = 0.0);
    DenseMatrix(index_t num_rows, index_t num_cols, std::vector<scalar_t> data);
    DenseMatrix(index_t num_rows, index_t num_cols, std::initializer_list<scalar_t> data);

    [[nodiscard]] index_t num_rows() const noexcept override { return num_rows_; }
    [[nodiscard]] index_t num_cols() const noexcept override { return num_cols_; }
    [[nodiscard]] size_t num_nonzeros() const noexcept override;
    [[nodiscard]] StorageFormat storage_format() const noexcept override { return StorageFormat::Dense; }

    [[nodiscard]] scalar_t& operator()(index_t row, index_t col) {
        check_bounds(row, col);
        return data_[static_cast<size_t>(row) * static_cast<size_t>(num_cols_) + static_cast<size_t>(col)];
    }

    [[nodiscard]] scalar_t operator()(index_t row, index_t col) const {
        check_bounds(row, col);
        return data_[static_cast<size_t>(row) * static_cast<size_t>(num_cols_) + static_cast<size_t>(col)];
    }

    [[nodiscard]] scalar_t coeff(index_t row, index_t col) const override {
        check_bounds(row, col);
        return data_[static_cast<size_t>(row) * static_cast<size_t>(num_cols_) + static_cast<size_t>(col)];
    }

    [[nodiscard]] std::span<const scalar_t> data() const noexcept { return data_; }
    [[nodiscard]] std::span<scalar_t> data() noexcept { return data_; }
    [[nodiscard]] const std::vector<scalar_t>& std_vector() const noexcept { return data_; }

    [[nodiscard]] ConstVectorView row(index_t i) const;
    [[nodiscard]] MutableVectorView row(index_t i);

    /// @brief Reference GEMV: y = alpha * A * x + beta * y
    void gemv(scalar_t alpha, ConstVectorView x, scalar_t beta, MutableVectorView y) const;

    /// @brief Reference Transpose GEMV: y = alpha * A^T * x + beta * y
    void gemv_transpose(scalar_t alpha, ConstVectorView x, scalar_t beta, MutableVectorView y) const;

    // ISparseMatrix interface delegates directly to gemv / gemv_transpose
    void spmv(scalar_t alpha, ConstVectorView x, scalar_t beta, MutableVectorView y) const override {
        gemv(alpha, x, beta, y);
    }
    void spmv_transpose(scalar_t alpha, ConstVectorView x, scalar_t beta, MutableVectorView y) const override {
        gemv_transpose(alpha, x, beta, y);
    }

    [[nodiscard]] scalar_t norm_inf() const override;
    [[nodiscard]] scalar_t norm_1() const override;
    [[nodiscard]] scalar_t norm_frobenius() const override;

    /// @brief Reference matrix-matrix multiplication: C = A * B
    [[nodiscard]] DenseMatrix matmul(const DenseMatrix& B) const;

    /// @brief Reference matrix addition: C = A + B
    [[nodiscard]] DenseMatrix add(const DenseMatrix& B) const;

    /// @brief Conversions to sparse formats
    [[nodiscard]] CSRMatrix to_csr(scalar_t zero_tol = 0.0) const;
    [[nodiscard]] CSCMatrix to_csc(scalar_t zero_tol = 0.0) const;
    [[nodiscard]] COOMatrix to_coo(scalar_t zero_tol = 0.0) const;

    /// @brief Factory helpers for test oracle generation
    [[nodiscard]] static DenseMatrix identity(index_t n);
    [[nodiscard]] static DenseMatrix diagonal(ConstVectorView diag);
    [[nodiscard]] static DenseMatrix from_sparse(const ISparseMatrix& sparse);
    [[nodiscard]] static DenseMatrix random(index_t num_rows, index_t num_cols,
                                            scalar_t min_val = -1.0, scalar_t max_val = 1.0,
                                            uint32_t seed = 42);

private:
    index_t num_rows_{0};
    index_t num_cols_{0};
    std::vector<scalar_t> data_{};
};

} // namespace pipepye::sparse
