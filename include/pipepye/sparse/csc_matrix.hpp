#pragma once

#include <pipepye/sparse/matrix_base.hpp>
#include <span>
#include <vector>

namespace pipepye::sparse {

class CSRMatrix;
class COOMatrix;

/// @brief Compressed Sparse Column (CSC) matrix representation.
/// Essential for Revised Simplex (column pricing, basis operations) and transpose SpMV (Aᵀ * y).
class CSCMatrix : public ISparseMatrix {
public:
    struct ColView {
        std::span<const index_t> row_indices;
        std::span<const scalar_t> values;

        [[nodiscard]] index_t size() const noexcept {
            return static_cast<index_t>(row_indices.size());
        }
        [[nodiscard]] bool empty() const noexcept {
            return row_indices.empty();
        }
    };

    CSCMatrix() = default;
    CSCMatrix(index_t num_rows, index_t num_cols,
              std::vector<index_t> col_ptr,
              std::vector<index_t> row_ind,
              std::vector<scalar_t> values);

    [[nodiscard]] index_t num_rows() const noexcept override { return num_rows_; }
    [[nodiscard]] index_t num_cols() const noexcept override { return num_cols_; }
    [[nodiscard]] size_t num_nonzeros() const noexcept override { return values_.size(); }
    [[nodiscard]] StorageFormat storage_format() const noexcept override { return StorageFormat::CSC; }

    [[nodiscard]] std::span<const index_t> col_ptr() const noexcept { return col_ptr_; }
    [[nodiscard]] std::span<const index_t> row_ind() const noexcept { return row_ind_; }
    [[nodiscard]] std::span<const index_t> row_idx() const noexcept { return row_ind_; }
    [[nodiscard]] std::span<const scalar_t> values() const noexcept { return values_; }

    [[nodiscard]] std::span<index_t> col_ptr() noexcept { return col_ptr_; }
    [[nodiscard]] std::span<index_t> row_ind() noexcept { return row_ind_; }
    [[nodiscard]] std::span<index_t> row_idx() noexcept { return row_ind_; }
    [[nodiscard]] std::span<scalar_t> values() noexcept { return values_; }

    [[nodiscard]] const std::vector<index_t>& col_ptr_vector() const noexcept { return col_ptr_; }
    [[nodiscard]] const std::vector<index_t>& row_ind_vector() const noexcept { return row_ind_; }
    [[nodiscard]] const std::vector<index_t>& row_idx_vector() const noexcept { return row_ind_; }
    [[nodiscard]] const std::vector<scalar_t>& values_vector() const noexcept { return values_; }

    [[nodiscard]] ColView col(index_t j) const;
    [[nodiscard]] index_t col_nnz(index_t j) const;

    /// @brief Direct coefficient access A(row, col). Binary searches inside col.
    [[nodiscard]] scalar_t coeff(index_t row, index_t col) const override;

    /// @brief Verifies canonical CSC properties (monotonic col_ptr, sorted unique row_ind within col).
    void validate() const;

    /// @brief Direct linear-time O(M + N + NNZ) conversion to Compressed Sparse Row (CSR).
    [[nodiscard]] CSRMatrix to_csr() const;

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
    std::vector<index_t> col_ptr_{0};
    std::vector<index_t> row_ind_{};
    std::vector<scalar_t> values_{};
};

} // namespace pipepye::sparse
