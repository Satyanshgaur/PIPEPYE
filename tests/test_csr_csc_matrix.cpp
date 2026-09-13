#include <gtest/gtest.h>
#include <pipepye/sparse/csr_matrix.hpp>
#include <pipepye/sparse/csc_matrix.hpp>
#include <cmath>

using namespace pipepye::sparse;
using pipepye::index_t;
using pipepye::scalar_t;

TEST(CSRCSCMatrixTest, CSRValidationErrors) {
    // Negative dimensions
    EXPECT_THROW(CSRMatrix(-1, 2, {0}, {}, {}), std::invalid_argument);
    EXPECT_THROW(CSRMatrix(2, -1, {0, 0, 0}, {}, {}), std::invalid_argument);

    // Empty row_ptr for non-zero rows
    EXPECT_THROW(CSRMatrix(2, 2, {}, {}, {}), std::invalid_argument);

    // row_ptr size mismatch (expected 3 for 2 rows)
    EXPECT_THROW(CSRMatrix(2, 2, {0, 0}, {}, {}), std::invalid_argument);

    // col_ind vs values size mismatch
    EXPECT_THROW(CSRMatrix(1, 2, {0, 1}, {0, 1}, {1.0}), std::invalid_argument);

    // row_ptr.back() != values.size()
    EXPECT_THROW(CSRMatrix(1, 2, {0, 2}, {0}, {1.0}), std::invalid_argument);

    // Out of range row access
    CSRMatrix csr(2, 2, {0, 0, 0}, {}, {});
    EXPECT_THROW((void)csr.row(-1), std::out_of_range);
    EXPECT_THROW((void)csr.row(2), std::out_of_range);
    EXPECT_THROW((void)csr.row_nnz(-1), std::out_of_range);
    EXPECT_THROW((void)csr.row_nnz(2), std::out_of_range);
}

TEST(CSRCSCMatrixTest, CSCValidationErrors) {
    // Negative dimensions
    EXPECT_THROW(CSCMatrix(-1, 2, {0, 0, 0}, {}, {}), std::invalid_argument);
    EXPECT_THROW(CSCMatrix(2, -1, {0}, {}, {}), std::invalid_argument);

    // Empty col_ptr for non-zero cols
    EXPECT_THROW(CSCMatrix(2, 2, {}, {}, {}), std::invalid_argument);

    // col_ptr size mismatch (expected 3 for 2 cols)
    EXPECT_THROW(CSCMatrix(2, 2, {0, 0}, {}, {}), std::invalid_argument);

    // row_ind vs values size mismatch
    EXPECT_THROW(CSCMatrix(2, 1, {0, 1}, {0, 1}, {1.0}), std::invalid_argument);

    // col_ptr.back() != values.size()
    EXPECT_THROW(CSCMatrix(2, 1, {0, 2}, {0}, {1.0}), std::invalid_argument);

    // Out of range col access
    CSCMatrix csc(2, 2, {0, 0, 0}, {}, {});
    EXPECT_THROW((void)csc.col(-1), std::out_of_range);
    EXPECT_THROW((void)csc.col(2), std::out_of_range);
    EXPECT_THROW((void)csc.col_nnz(-1), std::out_of_range);
    EXPECT_THROW((void)csc.col_nnz(2), std::out_of_range);
}

TEST(CSRCSCMatrixTest, EmptyMatricesNormsAndSpMV) {
    CSRMatrix empty_csr(0, 0, {0}, {}, {});
    EXPECT_EQ(empty_csr.num_rows(), 0);
    EXPECT_EQ(empty_csr.num_cols(), 0);
    EXPECT_EQ(empty_csr.num_nonzeros(), 0);
    EXPECT_DOUBLE_EQ(empty_csr.norm_1(), 0.0);
    EXPECT_DOUBLE_EQ(empty_csr.norm_inf(), 0.0);
    EXPECT_DOUBLE_EQ(empty_csr.norm_frobenius(), 0.0);

    CSCMatrix empty_csc(0, 0, {0}, {}, {});
    EXPECT_EQ(empty_csc.num_rows(), 0);
    EXPECT_EQ(empty_csc.num_cols(), 0);
    EXPECT_EQ(empty_csc.num_nonzeros(), 0);
    EXPECT_DOUBLE_EQ(empty_csc.norm_1(), 0.0);
    EXPECT_DOUBLE_EQ(empty_csc.norm_inf(), 0.0);
    EXPECT_DOUBLE_EQ(empty_csc.norm_frobenius(), 0.0);
}
