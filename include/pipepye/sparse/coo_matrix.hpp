#pragma once

#include <pipepye/sparse/matrix_base.hpp>
#include <pipepye/sparse/sparse_types.hpp>
#include <pipepye/sparse/vector.hpp>
#include <pipepye/sparse/csr_matrix.hpp>
#include <pipepye/sparse/csc_matrix.hpp>
#include <span>
#include <vector>

namespace pipepye::sparse {

/// @brief Coordinate (COO) sparse matrix format storing (row, col, value) triplets.
/// Natural intermediate assembly format produced during MPS parsing and model construction.
class COOMatrix : public ISparseMatrix {
public:
    COOMatrix() = default;

    /// @brief Construct an empty COO matrix with given shape and optional initial capacity.
    COOMatrix(index_t num_rows, index_t num_cols, size_t initial_capacity = 0);

    /// @brief Construct COO matrix with existing triplets.
    COOMatrix(index_t num_rows, index_t num_cols, std::vector<TripletF64> triplets);

    /// @brief Construct COO matrix from parallel row, col, and value spans.
    COOMatrix(index_t num_rows, index_t num_cols,
              std::span<const index_t> rows,
              std::span<const index_t> cols,
              std::span<const scalar_t> vals);

    [[nodiscard]] index_t num_rows() const noexcept override { return num_rows_; }
    [[nodiscard]] index_t num_cols() const noexcept override { return num_cols_; }
    [[nodiscard]] size_t num_nonzeros() const noexcept override { return triplets_.size(); }
    [[nodiscard]] StorageFormat storage_format() const noexcept override { return StorageFormat::COO; }

    [[nodiscard]] bool is_sorted() const noexcept { return is_sorted_; }
    [[nodiscard]] StorageOrder storage_order() const noexcept { return storage_order_; }
    [[nodiscard]] bool has_duplicates() const noexcept { return has_duplicates_; }

    /// @brief Update the matrix dimensions.
    void set_dimensions(index_t num_rows, index_t num_cols);

    /// @brief Appends a triplet (row, col, value). Validates indices against bounds.
    void add_entry(index_t row, index_t col, scalar_t value);

    /// @brief Appends a triplet (row, col, value) without bounds check, auto-expanding dimensions if needed.
    void add_entry_unchecked(index_t row, index_t col, scalar_t value);

    /// @brief Reserves storage capacity for triplets.
    void reserve(size_t nnz);

    /// @brief Clears all entries while retaining dimensions.
    void clear() noexcept;

    /// @brief Sorts triplets by RowMajor or ColMajor order.
    void sort(StorageOrder order = StorageOrder::RowMajor);

    /// @brief Combines duplicate entries at the same (row, col) coordinates by summing their values.
    /// If not already sorted, sorts first using the current storage order.
    void sum_duplicates();

    /// @brief Removes entries with absolute value <= tolerance.
    void drop_zeros(scalar_t tolerance = 1e-15);

    /// @brief Direct triplet accessors
    [[nodiscard]] std::span<const TripletF64> triplets() const noexcept { return triplets_; }
    [[nodiscard]] std::span<TripletF64> triplets() noexcept { return triplets_; }
    [[nodiscard]] const std::vector<TripletF64>& triplets_vector() const noexcept { return triplets_; }
    [[nodiscard]] std::vector<TripletF64>& triplets_vector() noexcept { return triplets_; }

    /// @brief Converts to Compressed Sparse Row (CSR) format.
    /// Guarantees canonical row-major order with summed duplicates.
    [[nodiscard]] CSRMatrix to_csr() const;

    /// @brief Converts to Compressed Sparse Column (CSC) format.
    /// Guarantees canonical column-major order with summed duplicates.
    [[nodiscard]] CSCMatrix to_csc() const;

    /// @brief Converts to dense row-major array of size num_rows * num_cols.
    [[nodiscard]] std::vector<scalar_t> to_dense() const;

    /// @brief Direct coefficient access A(row, col).
    [[nodiscard]] scalar_t coeff(index_t row, index_t col) const override;

    /// @brief SpMV: y = alpha * A * x + beta * y
    void spmv(scalar_t alpha, ConstVectorView x, scalar_t beta, MutableVectorView y) const override;

    /// @brief Transpose SpMV: y = alpha * A^T * x + beta * y
    void spmv_transpose(scalar_t alpha, ConstVectorView x, scalar_t beta, MutableVectorView y) const override;

    /// @brief Infinity norm: max row sum of absolute values
    [[nodiscard]] scalar_t norm_inf() const override;

    /// @brief 1-norm: max column sum of absolute values
    [[nodiscard]] scalar_t norm_1() const override;

    /// @brief Frobenius norm: sqrt(sum(A_ij^2))
    [[nodiscard]] scalar_t norm_frobenius() const override;

private:
    index_t num_rows_{0};
    index_t num_cols_{0};
    std::vector<TripletF64> triplets_{};
    bool is_sorted_{true};
    StorageOrder storage_order_{StorageOrder::RowMajor};
    bool has_duplicates_{false};
};

} // namespace pipepye::sparse
