#pragma once

#include <pipepye/sparse/matrix_base.hpp>
#include <span>
#include <vector>

namespace pipepye::sparse {

class CSCMatrix;
class COOMatrix;

/// @brief Compressed Sparse Row (CSR) matrix representation.
/// Optimal for row-wise presolve, row inequality checks, and forward SpMV (A * x).
class CSRMatrix : public ISparseMatrix {
public:
    struct RowView {
        std::span<const index_t> col_indices;
        std::span<const scalar_t> values;

        [[nodiscard]] index_t size() const noexcept {
            return static_cast<index_t>(col_indices.size());
        }
        [[nodiscard]] bool empty() const noexcept {
            return col_indices.empty();
        }
    };

    CSRMatrix() = default;
    CSRMatrix(index_t num_rows, index_t num_cols,
              std::vector<index_t> row_ptr,
              std::vector<index_t> col_ind,
              std::vector<scalar_t> values);

    [[nodiscard]] index_t num_rows() const noexcept override { return num_rows_; }
    [[nodiscard]] index_t num_cols() const noexcept override { return num_cols_; }
    [[nodiscard]] size_t num_nonzeros() const noexcept override { return values_.size(); }
    [[nodiscard]] StorageFormat storage_format() const noexcept override { return StorageFormat::CSR; }

    [[nodiscard]] std::span<const index_t> row_ptr() const noexcept { return row_ptr_; }
    [[nodiscard]] std::span<const index_t> col_ind() const noexcept { return col_ind_; }
    [[nodiscard]] std::span<const index_t> col_idx() const noexcept { return col_ind_; }
    [[nodiscard]] std::span<const scalar_t> values() const noexcept { return values_; }

    [[nodiscard]] std::span<index_t> row_ptr() noexcept { return row_ptr_; }
    [[nodiscard]] std::span<index_t> col_ind() noexcept { return col_ind_; }
    [[nodiscard]] std::span<index_t> col_idx() noexcept { return col_ind_; }
    [[nodiscard]] std::span<scalar_t> values() noexcept { return values_; }

    [[nodiscard]] const std::vector<index_t>& row_ptr_vector() const noexcept { return row_ptr_; }
    [[nodiscard]] const std::vector<index_t>& col_ind_vector() const noexcept { return col_ind_; }
    [[nodiscard]] const std::vector<index_t>& col_idx_vector() const noexcept { return col_ind_; }
    [[nodiscard]] const std::vector<scalar_t>& values_vector() const noexcept { return values_; }

    [[nodiscard]] RowView row(index_t i) const;
    [[nodiscard]] index_t row_nnz(index_t i) const;

    /// @brief Direct coefficient access A(row, col). Binary searches inside row.
    [[nodiscard]] scalar_t coeff(index_t row, index_t col) const override;

    /// @brief Verifies canonical CSR properties (monotonic row_ptr, sorted unique col_ind within row).
    void validate() const;

    /// @brief Direct linear-time O(M + N + NNZ) conversion to Compressed Sparse Column (CSC).
    [[nodiscard]] CSCMatrix to_csc() const;

    /// @brief Direct conversion to Coordinate (COO) representation.
    [[nodiscard]] COOMatrix to_coo() const;

    /// @brief Converts to dense row-major array of size num_rows * num_cols.
    [[nodiscard]] std::vector<scalar_t> to_dense() const;

    void spmv(scalar_t alpha, ConstVectorView x, scalar_t beta, MutableVectorView y) const override;
    void spmv_transpose(scalar_t alpha, ConstVectorView x, scalar_t beta, MutableVectorView y) const override;

    [[nodiscard]] scalar_t norm_inf() const override;
    [[nodiscard]] scalar_t norm_1() const override;
    [[nodiscard]] scalar_t norm_frobenius() const override;

private:
    index_t num_rows_{0};
    index_t num_cols_{0};
    std::vector<index_t> row_ptr_{0};
    std::vector<index_t> col_ind_{};
    std::vector<scalar_t> values_{};
};

} // namespace pipepye::sparse
