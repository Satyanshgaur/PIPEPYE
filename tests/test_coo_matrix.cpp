#include <gtest/gtest.h>
#include <pipepye/sparse/coo_matrix.hpp>
#include <cmath>

using namespace pipepye::sparse;
using pipepye::index_t;
using pipepye::scalar_t;

TEST(COOMatrixTest, ConstructionAndProperties) {
    COOMatrix empty;
    EXPECT_EQ(empty.num_rows(), 0);
    EXPECT_EQ(empty.num_cols(), 0);
    EXPECT_EQ(empty.num_nonzeros(), 0);
    EXPECT_TRUE(empty.is_empty());
    EXPECT_TRUE(empty.is_square());
    EXPECT_DOUBLE_EQ(empty.density(), 0.0);
    EXPECT_EQ(empty.storage_format(), StorageFormat::COO);

    COOMatrix mat(4, 5, 10);
    EXPECT_EQ(mat.num_rows(), 4);
    EXPECT_EQ(mat.num_cols(), 5);
    EXPECT_EQ(mat.num_nonzeros(), 0);
    EXPECT_TRUE(mat.is_empty());
    EXPECT_FALSE(mat.is_square());

    EXPECT_THROW(COOMatrix(-1, 5), std::invalid_argument);
    EXPECT_THROW(COOMatrix(4, -2), std::invalid_argument);
}

TEST(COOMatrixTest, EntryAdditionAndBoundsChecking) {
    COOMatrix mat(3, 3);
    mat.add_entry(0, 0, 1.0);
    mat.add_entry(1, 2, 3.5);
    mat.add_entry(2, 1, -4.0);

    EXPECT_EQ(mat.num_nonzeros(), 3);
    EXPECT_FALSE(mat.is_empty());
    EXPECT_NEAR(mat.density(), 3.0 / 9.0, 1e-9);

    // Bounds checking
    EXPECT_THROW(mat.add_entry(-1, 0, 1.0), std::out_of_range);
    EXPECT_THROW(mat.add_entry(3, 1, 1.0), std::out_of_range);
    EXPECT_THROW(mat.add_entry(0, 3, 1.0), std::out_of_range);
    EXPECT_THROW(mat.add_entry(0, -1, 1.0), std::out_of_range);
}

TEST(COOMatrixTest, UncheckedAdditionAutoExpands) {
    COOMatrix mat(2, 2);
    mat.add_entry_unchecked(3, 4, 10.0);

    EXPECT_EQ(mat.num_rows(), 4);
    EXPECT_EQ(mat.num_cols(), 5);
    EXPECT_EQ(mat.num_nonzeros(), 1);
}

TEST(COOMatrixTest, ParallelSpanConstruction) {
    std::vector<index_t> rows = {0, 1, 2};
    std::vector<index_t> cols = {2, 0, 1};
    std::vector<scalar_t> vals = {5.0, 6.0, 7.0};

    COOMatrix mat(3, 3, rows, cols, vals);
    EXPECT_EQ(mat.num_nonzeros(), 3);

    std::vector<index_t> short_cols = {2, 0};
    EXPECT_THROW(COOMatrix(3, 3, rows, short_cols, vals), std::invalid_argument);
}

TEST(COOMatrixTest, SortingRowAndColMajor) {
    COOMatrix mat(3, 3);
    mat.add_entry(2, 0, 1.0);
    mat.add_entry(0, 2, 2.0);
    mat.add_entry(1, 1, 3.0);
    mat.add_entry(0, 0, 4.0);
    mat.add_entry(2, 2, 5.0);

    // Row-Major: (0,0), (0,2), (1,1), (2,0), (2,2)
    mat.sort(StorageOrder::RowMajor);
    EXPECT_TRUE(mat.is_sorted());
    EXPECT_EQ(mat.storage_order(), StorageOrder::RowMajor);

    auto trips = mat.triplets();
    EXPECT_EQ(trips[0].row, 0); EXPECT_EQ(trips[0].col, 0); EXPECT_DOUBLE_EQ(trips[0].val, 4.0);
    EXPECT_EQ(trips[1].row, 0); EXPECT_EQ(trips[1].col, 2); EXPECT_DOUBLE_EQ(trips[1].val, 2.0);
    EXPECT_EQ(trips[2].row, 1); EXPECT_EQ(trips[2].col, 1); EXPECT_DOUBLE_EQ(trips[2].val, 3.0);
    EXPECT_EQ(trips[3].row, 2); EXPECT_EQ(trips[3].col, 0); EXPECT_DOUBLE_EQ(trips[3].val, 1.0);
    EXPECT_EQ(trips[4].row, 2); EXPECT_EQ(trips[4].col, 2); EXPECT_DOUBLE_EQ(trips[4].val, 5.0);

    // Col-Major: (0,0), (2,0), (1,1), (0,2), (2,2)
    mat.sort(StorageOrder::ColMajor);
    EXPECT_EQ(mat.storage_order(), StorageOrder::ColMajor);
    trips = mat.triplets();
    EXPECT_EQ(trips[0].row, 0); EXPECT_EQ(trips[0].col, 0);
    EXPECT_EQ(trips[1].row, 2); EXPECT_EQ(trips[1].col, 0);
    EXPECT_EQ(trips[2].row, 1); EXPECT_EQ(trips[2].col, 1);
    EXPECT_EQ(trips[3].row, 0); EXPECT_EQ(trips[3].col, 2);
    EXPECT_EQ(trips[4].row, 2); EXPECT_EQ(trips[4].col, 2);
}

TEST(COOMatrixTest, SumDuplicates) {
    COOMatrix mat(2, 2);
    mat.add_entry(0, 1, 2.0);
    mat.add_entry(1, 0, 5.0);
    mat.add_entry(0, 1, 3.5); // duplicate at (0, 1) -> 5.5
    mat.add_entry(0, 1, -1.5); // duplicate at (0, 1) -> 4.0

    EXPECT_EQ(mat.num_nonzeros(), 4);
    mat.sum_duplicates();
    EXPECT_EQ(mat.num_nonzeros(), 2);
    EXPECT_FALSE(mat.has_duplicates());

    auto trips = mat.triplets();
    EXPECT_EQ(trips[0].row, 0);
    EXPECT_EQ(trips[0].col, 1);
    EXPECT_DOUBLE_EQ(trips[0].val, 4.0);

    EXPECT_EQ(trips[1].row, 1);
    EXPECT_EQ(trips[1].col, 0);
    EXPECT_DOUBLE_EQ(trips[1].val, 5.0);
}

TEST(COOMatrixTest, DropZeros) {
    COOMatrix mat(3, 3);
    mat.add_entry(0, 0, 1.0);
    mat.add_entry(1, 1, 1e-16);
    mat.add_entry(2, 2, -2.0);
    mat.add_entry(0, 1, 0.0);

    mat.drop_zeros(1e-15);
    EXPECT_EQ(mat.num_nonzeros(), 2);

    auto trips = mat.triplets();
    EXPECT_DOUBLE_EQ(trips[0].val, 1.0);
    EXPECT_DOUBLE_EQ(trips[1].val, -2.0);
}

TEST(COOMatrixTest, DenseConversion) {
    COOMatrix mat(2, 3);
    mat.add_entry(0, 0, 1.0);
    mat.add_entry(0, 2, 2.0);
    mat.add_entry(1, 1, 3.0);
    mat.add_entry(1, 1, 4.0); // Duplicate: 3.0 + 4.0 = 7.0

    auto dense = mat.to_dense();
    // Expected 2x3 matrix:
    // [1.0, 0.0, 2.0]
    // [0.0, 7.0, 0.0]
    ASSERT_EQ(dense.size(), 6);
    EXPECT_DOUBLE_EQ(dense[0], 1.0);
    EXPECT_DOUBLE_EQ(dense[1], 0.0);
    EXPECT_DOUBLE_EQ(dense[2], 2.0);
    EXPECT_DOUBLE_EQ(dense[3], 0.0);
    EXPECT_DOUBLE_EQ(dense[4], 7.0);
    EXPECT_DOUBLE_EQ(dense[5], 0.0);
}

TEST(COOMatrixTest, SpMVForwardAndTranspose) {
    // 2 x 3 matrix A:
    // [ 1.0,  0.0, -2.0 ]
    // [ 3.0,  4.0,  0.0 ]
    COOMatrix A(2, 3);
    A.add_entry(0, 0, 1.0);
    A.add_entry(0, 2, -2.0);
    A.add_entry(1, 0, 3.0);
    A.add_entry(1, 1, 4.0);

    // x = [2.0, 1.0, -1.0]^T
    // A * x = [ 1*(2) + 0*(1) + (-2)*(-1), 3*(2) + 4*(1) + 0*(-1) ] = [ 4.0, 10.0 ]
    Vector x = {2.0, 1.0, -1.0};
    Vector y(2, 0.0);

    // Test y = 1.0 * A * x + 0.0 * y
    A.spmv(1.0, x.view(), 0.0, y.view());
    EXPECT_DOUBLE_EQ(y[0], 4.0);
    EXPECT_DOUBLE_EQ(y[1], 10.0);

    // Test with alpha = 2.0 and beta = 1.5
    // y = 2.0 * [4, 10] + 1.5 * [4, 10] = 3.5 * [4, 10] = [14.0, 35.0]
    A.spmv(2.0, x.view(), 1.5, y.view());
    EXPECT_DOUBLE_EQ(y[0], 14.0);
    EXPECT_DOUBLE_EQ(y[1], 35.0);

    // Test Transpose: A^T is 3 x 2
    // A^T = [ 1.0,  3.0 ]
    //       [ 0.0,  4.0 ]
    //       [-2.0,  0.0 ]
    // x_t = [3.0, 2.0]^T
    // A^T * x_t = [ 1*(3) + 3*(2), 0*(3) + 4*(2), -2*(3) + 0*(2) ] = [ 9.0, 8.0, -6.0 ]
    Vector x_t = {3.0, 2.0};
    Vector y_t(3, 0.0);
    A.spmv_transpose(1.0, x_t.view(), 0.0, y_t.view());
    EXPECT_DOUBLE_EQ(y_t[0], 9.0);
    EXPECT_DOUBLE_EQ(y_t[1], 8.0);
    EXPECT_DOUBLE_EQ(y_t[2], -6.0);

    // Dimension mismatch checks
    Vector bad_x = {1.0, 2.0};
    EXPECT_THROW(A.spmv(1.0, bad_x.view(), 0.0, y.view()), std::invalid_argument);
    EXPECT_THROW(A.spmv_transpose(1.0, x.view(), 0.0, y_t.view()), std::invalid_argument);
}

TEST(COOMatrixTest, MathematicalNorms) {
    // 2 x 3 matrix A:
    // [ 1.0, -3.0,  0.0 ]
    // [ 4.0,  2.0, -5.0 ]
    COOMatrix A(2, 3);
    A.add_entry(0, 0, 1.0);
    A.add_entry(0, 1, -3.0);
    A.add_entry(1, 0, 4.0);
    A.add_entry(1, 1, 2.0);
    A.add_entry(1, 2, -5.0);

    // Inf-norm: max row sum: row 0: |1| + |-3| = 4; row 1: |4| + |2| + |-5| = 11 -> 11.0
    EXPECT_DOUBLE_EQ(A.norm_inf(), 11.0);

    // 1-norm: max col sum: col 0: 1+4=5; col 1: 3+2=5; col 2: 5 -> 5.0
    EXPECT_DOUBLE_EQ(A.norm_1(), 5.0);

    // Frobenius norm: sqrt(1^2 + 9 + 16 + 4 + 25) = sqrt(55)
    EXPECT_DOUBLE_EQ(A.norm_frobenius(), std::sqrt(55.0));
}

TEST(COOMatrixTest, ConvertToCSRAndCSC) {
    // 3 x 3 matrix
    // Row 0: col 1: 2.0, col 2: 3.0
    // Row 1: col 0: 4.0
    // Row 2: col 1: 1.0, col 2: 5.0
    COOMatrix coo(3, 3);
    // Add out-of-order and with a duplicate
    coo.add_entry(2, 2, 2.0);
    coo.add_entry(0, 1, 2.0);
    coo.add_entry(1, 0, 4.0);
    coo.add_entry(2, 1, 1.0);
    coo.add_entry(0, 2, 3.0);
    coo.add_entry(2, 2, 3.0); // duplicate at (2, 2) -> 5.0

    CSRMatrix csr = coo.to_csr();
    EXPECT_EQ(csr.num_rows(), 3);
    EXPECT_EQ(csr.num_cols(), 3);
    EXPECT_EQ(csr.num_nonzeros(), 5);

    auto row_ptr = csr.row_ptr();
    ASSERT_EQ(row_ptr.size(), 4);
    EXPECT_EQ(row_ptr[0], 0);
    EXPECT_EQ(row_ptr[1], 2); // row 0 has 2 nonzeros
    EXPECT_EQ(row_ptr[2], 3); // row 1 has 1 nonzero
    EXPECT_EQ(row_ptr[3], 5); // row 2 has 2 nonzeros

    auto col_ind = csr.col_ind();
    auto values = csr.values();
    EXPECT_EQ(col_ind[0], 1); EXPECT_DOUBLE_EQ(values[0], 2.0);
    EXPECT_EQ(col_ind[1], 2); EXPECT_DOUBLE_EQ(values[1], 3.0);
    EXPECT_EQ(col_ind[2], 0); EXPECT_DOUBLE_EQ(values[2], 4.0);
    EXPECT_EQ(col_ind[3], 1); EXPECT_DOUBLE_EQ(values[3], 1.0);
    EXPECT_EQ(col_ind[4], 2); EXPECT_DOUBLE_EQ(values[4], 5.0);

    // Check RowView
    auto r0 = csr.row(0);
    EXPECT_EQ(r0.size(), 2);
    EXPECT_EQ(r0.col_indices[0], 1);
    EXPECT_EQ(r0.col_indices[1], 2);
    EXPECT_EQ(csr.row_nnz(0), 2);

    // Convert to CSC
    CSCMatrix csc = coo.to_csc();
    EXPECT_EQ(csc.num_rows(), 3);
    EXPECT_EQ(csc.num_cols(), 3);
    EXPECT_EQ(csc.num_nonzeros(), 5);

    auto col_ptr = csc.col_ptr();
    ASSERT_EQ(col_ptr.size(), 4);
    EXPECT_EQ(col_ptr[0], 0);
    EXPECT_EQ(col_ptr[1], 1); // col 0 has (1,0) -> 1 nz
    EXPECT_EQ(col_ptr[2], 3); // col 1 has (0,1), (2,1) -> 2 nz
    EXPECT_EQ(col_ptr[3], 5); // col 2 has (0,2), (2,2) -> 2 nz

    auto row_ind = csc.row_ind();
    auto csc_values = csc.values();
    EXPECT_EQ(row_ind[0], 1); EXPECT_DOUBLE_EQ(csc_values[0], 4.0);
    EXPECT_EQ(row_ind[1], 0); EXPECT_DOUBLE_EQ(csc_values[1], 2.0);
    EXPECT_EQ(row_ind[2], 2); EXPECT_DOUBLE_EQ(csc_values[2], 1.0);
    EXPECT_EQ(row_ind[3], 0); EXPECT_DOUBLE_EQ(csc_values[3], 3.0);
    EXPECT_EQ(row_ind[4], 2); EXPECT_DOUBLE_EQ(csc_values[4], 5.0);

    // Verify SpMV consistency across COO, CSR, CSC
    Vector x = {1.5, -2.0, 3.5};
    Vector y_coo(3, 0.0);
    Vector y_csr(3, 0.0);
    Vector y_csc(3, 0.0);

    coo.spmv(1.0, x.view(), 0.0, y_coo.view());
    csr.spmv(1.0, x.view(), 0.0, y_csr.view());
    csc.spmv(1.0, x.view(), 0.0, y_csc.view());

    for (index_t i = 0; i < 3; ++i) {
        EXPECT_DOUBLE_EQ(y_coo[i], y_csr[i]);
        EXPECT_DOUBLE_EQ(y_csr[i], y_csc[i]);
    }

    // Verify SpMV Transpose consistency
    Vector xt = {2.0, -1.0, 4.0};
    Vector yt_coo(3, 0.0);
    Vector yt_csr(3, 0.0);
    Vector yt_csc(3, 0.0);

    coo.spmv_transpose(1.0, xt.view(), 0.0, yt_coo.view());
    csr.spmv_transpose(1.0, xt.view(), 0.0, yt_csr.view());
    csc.spmv_transpose(1.0, xt.view(), 0.0, yt_csc.view());

    for (index_t j = 0; j < 3; ++j) {
        EXPECT_DOUBLE_EQ(yt_coo[j], yt_csr[j]);
        EXPECT_DOUBLE_EQ(yt_csr[j], yt_csc[j]);
    }
}
