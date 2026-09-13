#pragma once

#include <pipepye/core/types.hpp>
#include <pipepye/sparse/sparse_types.hpp>
#include <pipepye/sparse/vector.hpp>
#include <concepts>
#include <stdexcept>
#include <string>

namespace pipepye::sparse {

/// @brief Mathematical interface describing the shape and behavior of any linear operator / matrix.
class ISparseMatrix {
public:
    virtual ~ISparseMatrix() = default;

    [[nodiscard]] virtual index_t num_rows() const noexcept = 0;
    [[nodiscard]] virtual index_t num_cols() const noexcept = 0;
    [[nodiscard]] virtual size_t num_nonzeros() const noexcept = 0;
    [[nodiscard]] virtual StorageFormat storage_format() const noexcept = 0;

    [[nodiscard]] bool is_empty() const noexcept {
        return num_rows() == 0 || num_cols() == 0 || num_nonzeros() == 0;
    }

    [[nodiscard]] bool is_square() const noexcept {
        return num_rows() == num_cols();
    }

    [[nodiscard]] double density() const noexcept {
        if (num_rows() == 0 || num_cols() == 0) return 0.0;
        return static_cast<double>(num_nonzeros()) / (static_cast<double>(num_rows()) * static_cast<double>(num_cols()));
    }

    /// @brief Validates index coordinates within matrix dimensions.
    void check_bounds(index_t row, index_t col) const {
        if (row < 0 || row >= num_rows() || col < 0 || col >= num_cols()) {
            throw std::out_of_range("Matrix index out of bounds: (" + std::to_string(row) +
                                    ", " + std::to_string(col) + ") for dimension (" +
                                    std::to_string(num_rows()) + "x" + std::to_string(num_cols()) + ")");
        }
    }

    /// @brief Computes y = alpha * A * x + beta * y
    virtual void spmv(scalar_t alpha, ConstVectorView x, scalar_t beta, MutableVectorView y) const = 0;

    /// @brief Computes y = alpha * A^T * x + beta * y
    virtual void spmv_transpose(scalar_t alpha, ConstVectorView x, scalar_t beta, MutableVectorView y) const = 0;

    /// @brief Direct coefficient access A(row, col). Returns 0.0 if not present.
    [[nodiscard]] virtual scalar_t coeff(index_t row, index_t col) const = 0;

    /// @brief Syntactic sugar for coeff(row, col)
    [[nodiscard]] scalar_t operator()(index_t row, index_t col) const {
        return coeff(row, col);
    }

    /// @brief Mathematically compares two matrices across all entries.
    [[nodiscard]] bool equals(const ISparseMatrix& other, scalar_t tol = 1e-12) const {
        if (num_rows() != other.num_rows() || num_cols() != other.num_cols()) {
            return false;
        }
        if (num_cols() == 0 || num_rows() == 0) {
            return true;
        }
        DenseVector<scalar_t> x(num_cols(), 0.0);
        DenseVector<scalar_t> y1(num_rows(), 0.0);
        DenseVector<scalar_t> y2(num_rows(), 0.0);
        for (index_t j = 0; j < num_cols(); ++j) {
            x[j] = 1.0;
            spmv(1.0, x.view(), 0.0, y1.view());
            other.spmv(1.0, x.view(), 0.0, y2.view());
            for (index_t i = 0; i < num_rows(); ++i) {
                if (std::abs(y1[i] - y2[i]) > tol) {
                    return false;
                }
            }
            x[j] = 0.0;
        }
        return true;
    }

    /// @brief Computes the matrix infinity norm: max row sum of absolute values
    [[nodiscard]] virtual scalar_t norm_inf() const = 0;

    /// @brief Computes the matrix 1-norm: max column sum of absolute values
    [[nodiscard]] virtual scalar_t norm_1() const = 0;

    /// @brief Computes the Frobenius norm of the matrix: sqrt(sum(A_ij^2))
    [[nodiscard]] virtual scalar_t norm_frobenius() const = 0;
};

/// @brief C++20 Concept defining a valid Sparse Matrix representation.
template <typename M>
concept SparseMatrixType = requires(const M& m, scalar_t alpha, ConstVectorView x, scalar_t beta, MutableVectorView y) {
    { m.num_rows() } noexcept -> std::same_as<index_t>;
    { m.num_cols() } noexcept -> std::same_as<index_t>;
    { m.num_nonzeros() } noexcept -> std::same_as<size_t>;
    { m.spmv(alpha, x, beta, y) };
    { m.spmv_transpose(alpha, x, beta, y) };
    { m.norm_inf() } -> std::same_as<scalar_t>;
};

} // namespace pipepye::sparse
