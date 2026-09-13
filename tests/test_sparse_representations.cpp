#include <gtest/gtest.h>
#include <pipepye/sparse/coo_matrix.hpp>
#include <pipepye/sparse/csr_matrix.hpp>
#include <pipepye/sparse/csc_matrix.hpp>
#include <cmath>
#include <random>

using namespace pipepye::sparse;
using pipepye::index_t;
using pipepye::scalar_t;

// =============================================================================
// Dimensions and NNZ Counts
// =============================================================================

TEST(SparseRepresentationsTest, DimensionVariantsAndNNZ) {
    // 1. Square Matrix
    {
        COOMatrix coo(4, 4);
        coo.add_entry(0, 0, 1.0);
        coo.add_entry(1, 1, 2.0);
        coo.add_entry(2, 2, 3.0);
        coo.add_entry(3, 3, 4.0);

        CSRMatrix csr = coo.to_csr();
        CSCMatrix csc = coo.to_csc();

        EXPECT_EQ(coo.num_rows(), 4); EXPECT_EQ(coo.num_cols(), 4); EXPECT_EQ(coo.num_nonzeros(), 4);
        EXPECT_EQ(csr.num_rows(), 4); EXPECT_EQ(csr.num_cols(), 4); EXPECT_EQ(csr.num_nonzeros(), 4);
        EXPECT_EQ(csc.num_rows(), 4); EXPECT_EQ(csc.num_cols(), 4); EXPECT_EQ(csc.num_nonzeros(), 4);
        EXPECT_TRUE(csr.is_square());
        EXPECT_TRUE(csc.is_square());
        EXPECT_TRUE(csr.equals(csc));
    }

    // 2. Rectangular Tall (6 x 2)
    {
        COOMatrix coo(6, 2);
        coo.add_entry(0, 0, 10.0);
        coo.add_entry(2, 1, 20.0);
        coo.add_entry(5, 0, 30.0);

        CSRMatrix csr = coo.to_csr();
        CSCMatrix csc = coo.to_csc();

        EXPECT_EQ(csr.num_rows(), 6); EXPECT_EQ(csr.num_cols(), 2); EXPECT_EQ(csr.num_nonzeros(), 3);
        EXPECT_EQ(csc.num_rows(), 6); EXPECT_EQ(csc.num_cols(), 2); EXPECT_EQ(csc.num_nonzeros(), 3);
        EXPECT_FALSE(csr.is_square());
        EXPECT_TRUE(csr.equals(csc));
        EXPECT_TRUE(coo.equals(csr));
    }

    // 3. Rectangular Wide (2 x 6)
    {
        COOMatrix coo(2, 6);
        coo.add_entry(0, 1, -1.0);
        coo.add_entry(0, 4, -2.0);
        coo.add_entry(1, 3, -3.0);
        coo.add_entry(1, 5, -4.0);

        CSRMatrix csr = coo.to_csr();
        CSCMatrix csc = coo.to_csc();

        EXPECT_EQ(csr.num_rows(), 2); EXPECT_EQ(csr.num_cols(), 6); EXPECT_EQ(csr.num_nonzeros(), 4);
        EXPECT_EQ(csc.num_rows(), 2); EXPECT_EQ(csc.num_cols(), 6); EXPECT_EQ(csc.num_nonzeros(), 4);
        EXPECT_TRUE(csr.equals(csc));
    }

    // 4. Single Row (1 x 5)
    {
        COOMatrix coo(1, 5);
        coo.add_entry(0, 0, 1.5);
        coo.add_entry(0, 3, 2.5);

        CSRMatrix csr = coo.to_csr();
        CSCMatrix csc = coo.to_csc();

        EXPECT_EQ(csr.num_rows(), 1); EXPECT_EQ(csr.num_cols(), 5); EXPECT_EQ(csr.num_nonzeros(), 2);
        EXPECT_EQ(csc.num_rows(), 1); EXPECT_EQ(csc.num_cols(), 5); EXPECT_EQ(csc.num_nonzeros(), 2);
        EXPECT_TRUE(csr.equals(csc));
    }

    // 5. Single Column (5 x 1)
    {
        COOMatrix coo(5, 1);
        coo.add_entry(1, 0, 7.0);
        coo.add_entry(4, 0, 9.0);

        CSRMatrix csr = coo.to_csr();
        CSCMatrix csc = coo.to_csc();

        EXPECT_EQ(csr.num_rows(), 5); EXPECT_EQ(csr.num_cols(), 1); EXPECT_EQ(csr.num_nonzeros(), 2);
        EXPECT_EQ(csc.num_rows(), 5); EXPECT_EQ(csc.num_cols(), 1); EXPECT_EQ(csc.num_nonzeros(), 2);
        EXPECT_TRUE(csr.equals(csc));
    }
}

// =============================================================================
// Empty Rows, Columns, and Matrices
// =============================================================================

TEST(SparseRepresentationsTest, CompletelyEmptyMatrix) {
    COOMatrix coo(5, 4); // 0 nonzeros
    EXPECT_TRUE(coo.is_empty());
    EXPECT_EQ(coo.num_nonzeros(), 0);

    CSRMatrix csr = coo.to_csr();
    CSCMatrix csc = coo.to_csc();

    EXPECT_TRUE(csr.is_empty());
    EXPECT_EQ(csr.num_nonzeros(), 0);
    EXPECT_EQ(csr.row_ptr().size(), 6);
    for (index_t p : csr.row_ptr()) {
        EXPECT_EQ(p, 0);
    }

    EXPECT_TRUE(csc.is_empty());
    EXPECT_EQ(csc.num_nonzeros(), 0);
    EXPECT_EQ(csc.col_ptr().size(), 5);
    for (index_t p : csc.col_ptr()) {
        EXPECT_EQ(p, 0);
    }

    EXPECT_TRUE(csr.equals(csc));

    // SpMV on empty matrix returns zero
    Vector x(4, 3.0);
    Vector y(5, 1.0);
    csr.spmv(1.0, x.view(), 0.0, y.view());
    for (index_t i = 0; i < 5; ++i) {
        EXPECT_DOUBLE_EQ(y[i], 0.0);
    }
}

TEST(SparseRepresentationsTest, EmptyRowsAndColumnsPatterns) {
    // Matrix size: 6 x 6
    // Row 0: EMPTY (leading empty row)
    // Row 1: col 2 = 5.0
    // Row 2: EMPTY (interior empty row)
    // Row 3: EMPTY (consecutive empty row)
    // Row 4: col 2 = 6.0, col 4 = 7.0
    // Row 5: EMPTY (trailing empty row)
    //
    // Col 0: EMPTY (leading empty col)
    // Col 1: EMPTY
    // Col 2: row 1, 4
    // Col 3: EMPTY (interior empty col)
    // Col 4: row 4
    // Col 5: EMPTY (trailing empty col)
    COOMatrix coo(6, 6);
    coo.add_entry(1, 2, 5.0);
    coo.add_entry(4, 2, 6.0);
    coo.add_entry(4, 4, 7.0);

    CSRMatrix csr = coo.to_csr();
    CSCMatrix csc = coo.to_csc();

    // Verify CSR row pointers
    auto rptr = csr.row_ptr();
    ASSERT_EQ(rptr.size(), 7);
    EXPECT_EQ(rptr[0], 0);
    EXPECT_EQ(rptr[1], 0); // row 0 is empty (0 nnz)
    EXPECT_EQ(rptr[2], 1); // row 1 has 1 nnz
    EXPECT_EQ(rptr[3], 1); // row 2 is empty
    EXPECT_EQ(rptr[4], 1); // row 3 is empty
    EXPECT_EQ(rptr[5], 3); // row 4 has 2 nnz
    EXPECT_EQ(rptr[6], 3); // row 5 is empty

    EXPECT_EQ(csr.row_nnz(0), 0);
    EXPECT_EQ(csr.row_nnz(1), 1);
    EXPECT_EQ(csr.row_nnz(2), 0);
    EXPECT_EQ(csr.row_nnz(3), 0);
    EXPECT_EQ(csr.row_nnz(4), 2);
    EXPECT_EQ(csr.row_nnz(5), 0);
    EXPECT_TRUE(csr.row(0).empty());
    EXPECT_TRUE(csr.row(2).empty());
    EXPECT_FALSE(csr.row(1).empty());

    // Verify CSC col pointers
    auto cptr = csc.col_ptr();
    ASSERT_EQ(cptr.size(), 7);
    EXPECT_EQ(cptr[0], 0);
    EXPECT_EQ(cptr[1], 0); // col 0 is empty
    EXPECT_EQ(cptr[2], 0); // col 1 is empty
    EXPECT_EQ(cptr[3], 2); // col 2 has 2 nnz
    EXPECT_EQ(cptr[4], 2); // col 3 is empty
    EXPECT_EQ(cptr[5], 3); // col 4 has 1 nnz
    EXPECT_EQ(cptr[6], 3); // col 5 is empty

    EXPECT_EQ(csc.col_nnz(0), 0);
    EXPECT_EQ(csc.col_nnz(1), 0);
    EXPECT_EQ(csc.col_nnz(2), 2);
    EXPECT_EQ(csc.col_nnz(3), 0);
    EXPECT_EQ(csc.col_nnz(4), 1);
    EXPECT_EQ(csc.col_nnz(5), 0);
    EXPECT_TRUE(csc.col(0).empty());
    EXPECT_TRUE(csc.col(5).empty());

    // Verify full equivalence
    EXPECT_TRUE(coo.equals(csr));
    EXPECT_TRUE(csr.equals(csc));
    EXPECT_TRUE(csc.equals(coo));

    // Verify coeff lookup on empty and non-empty slots
    EXPECT_DOUBLE_EQ(csr.coeff(0, 0), 0.0);
    EXPECT_DOUBLE_EQ(csr.coeff(1, 2), 5.0);
    EXPECT_DOUBLE_EQ(csr.coeff(4, 2), 6.0);
    EXPECT_DOUBLE_EQ(csr.coeff(4, 4), 7.0);
    EXPECT_DOUBLE_EQ(csr.coeff(5, 5), 0.0);

    EXPECT_DOUBLE_EQ(csc.coeff(0, 0), 0.0);
    EXPECT_DOUBLE_EQ(csc.coeff(1, 2), 5.0);
    EXPECT_DOUBLE_EQ(csc.coeff(4, 2), 6.0);
    EXPECT_DOUBLE_EQ(csc.coeff(4, 4), 7.0);
    EXPECT_DOUBLE_EQ(csc.coeff(5, 5), 0.0);
}

// =============================================================================
// Duplicate Entries and Cancellation
// =============================================================================

TEST(SparseRepresentationsTest, DuplicateEntriesSumming) {
    COOMatrix coo(3, 3);
    // Add multiple duplicates for coordinate (1, 1)
    coo.add_entry(1, 1, 2.0);
    coo.add_entry(0, 2, 4.0);
    coo.add_entry(1, 1, 3.5);
    coo.add_entry(2, 0, 1.0);
    coo.add_entry(1, 1, -1.5); // 2.0 + 3.5 - 1.5 = 4.0
    // Add duplicates for coordinate (0, 2)
    coo.add_entry(0, 2, -1.0); // 4.0 - 1.0 = 3.0

    EXPECT_EQ(coo.num_nonzeros(), 6);

    CSRMatrix csr = coo.to_csr();
    CSCMatrix csc = coo.to_csc();

    // Unique nonzeros: (0, 2), (1, 1), (2, 0) -> exactly 3
    EXPECT_EQ(csr.num_nonzeros(), 3);
    EXPECT_EQ(csc.num_nonzeros(), 3);

    EXPECT_DOUBLE_EQ(csr.coeff(1, 1), 4.0);
    EXPECT_DOUBLE_EQ(csr.coeff(0, 2), 3.0);
    EXPECT_DOUBLE_EQ(csr.coeff(2, 0), 1.0);

    EXPECT_DOUBLE_EQ(csc.coeff(1, 1), 4.0);
    EXPECT_DOUBLE_EQ(csc.coeff(0, 2), 3.0);
    EXPECT_DOUBLE_EQ(csc.coeff(2, 0), 1.0);

    EXPECT_TRUE(csr.equals(csc));
}

TEST(SparseRepresentationsTest, DuplicateCancellationToZero) {
    COOMatrix coo(2, 2);
    coo.add_entry(0, 1, 5.0);
    coo.add_entry(0, 1, -5.0); // sums to 0.0
    coo.add_entry(1, 0, 3.0);

    CSRMatrix csr = coo.to_csr();
    CSCMatrix csc = coo.to_csc();

    // The entry at (0, 1) summed to 0.0
    EXPECT_DOUBLE_EQ(csr.coeff(0, 1), 0.0);
    EXPECT_DOUBLE_EQ(csr.coeff(1, 0), 3.0);
    EXPECT_DOUBLE_EQ(csc.coeff(0, 1), 0.0);
    EXPECT_DOUBLE_EQ(csc.coeff(1, 0), 3.0);
    EXPECT_TRUE(csr.equals(csc));
}

// =============================================================================
// Unsorted Input and Strict Sorting Verification
// =============================================================================

TEST(SparseRepresentationsTest, UnsortedInputStrictAscendingOrder) {
    // Create an unsorted set of triplets in reverse order
    COOMatrix coo(4, 4);
    coo.add_entry(3, 3, 1.0);
    coo.add_entry(3, 1, 2.0);
    coo.add_entry(2, 3, 3.0);
    coo.add_entry(2, 0, 4.0);
    coo.add_entry(1, 2, 5.0);
    coo.add_entry(0, 3, 6.0);
    coo.add_entry(0, 1, 7.0);

    // Convert to CSR and verify row indices are strictly ascending within each row
    CSRMatrix csr = coo.to_csr();
    csr.validate(); // Will throw if col_idx not strictly ascending

    for (index_t i = 0; i < csr.num_rows(); ++i) {
        auto r = csr.row(i);
        for (index_t k = 1; k < r.size(); ++k) {
            EXPECT_LT(r.col_indices[k - 1], r.col_indices[k])
                << "Row " << i << " column indices not strictly ascending!";
        }
    }

    // Convert to CSC and verify row indices are strictly ascending within each column
    CSCMatrix csc = coo.to_csc();
    csc.validate(); // Will throw if row_idx not strictly ascending

    for (index_t j = 0; j < csc.num_cols(); ++j) {
        auto c = csc.col(j);
        for (index_t k = 1; k < c.size(); ++k) {
            EXPECT_LT(c.row_indices[k - 1], c.row_indices[k])
                << "Column " << j << " row indices not strictly ascending!";
        }
    }

    EXPECT_TRUE(csr.equals(csc));
}

// =============================================================================
// Negative Values and Mathematical Operations
// =============================================================================

TEST(SparseRepresentationsTest, NegativeValuesAndNorms) {
    COOMatrix coo(3, 3);
    coo.add_entry(0, 0, -10.0);
    coo.add_entry(0, 2, 2.0);
    coo.add_entry(1, 1, -5.0);
    coo.add_entry(2, 0, -3.0);
    coo.add_entry(2, 2, -4.0);

    CSRMatrix csr = coo.to_csr();
    CSCMatrix csc = coo.to_csc();

    // Infinity norm = max row sum:
    // row 0: |-10| + |2| = 12
    // row 1: |-5| = 5
    // row 2: |-3| + |-4| = 7 -> max = 12.0
    EXPECT_DOUBLE_EQ(coo.norm_inf(), 12.0);
    EXPECT_DOUBLE_EQ(csr.norm_inf(), 12.0);
    EXPECT_DOUBLE_EQ(csc.norm_inf(), 12.0);

    // 1-norm = max col sum:
    // col 0: |-10| + |-3| = 13
    // col 1: |-5| = 5
    // col 2: |2| + |-4| = 6 -> max = 13.0
    EXPECT_DOUBLE_EQ(coo.norm_1(), 13.0);
    EXPECT_DOUBLE_EQ(csr.norm_1(), 13.0);
    EXPECT_DOUBLE_EQ(csc.norm_1(), 13.0);

    // Frobenius norm = sqrt(100 + 4 + 25 + 9 + 16) = sqrt(154)
    scalar_t expected_frobenius = std::sqrt(154.0);
    EXPECT_DOUBLE_EQ(coo.norm_frobenius(), expected_frobenius);
    EXPECT_DOUBLE_EQ(csr.norm_frobenius(), expected_frobenius);
    EXPECT_DOUBLE_EQ(csc.norm_frobenius(), expected_frobenius);

    // SpMV with negative vector
    Vector x = {-1.0, 2.0, -3.0};
    // y[0] = -10*(-1) + 2*(-3) = 10 - 6 = 4
    // y[1] = -5*(2) = -10
    // y[2] = -3*(-1) + (-4)*(-3) = 3 + 12 = 15
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
    EXPECT_DOUBLE_EQ(y_csr[0], 4.0);
    EXPECT_DOUBLE_EQ(y_csr[1], -10.0);
    EXPECT_DOUBLE_EQ(y_csr[2], 15.0);
}

// =============================================================================
// Complete Conversion Graph Verification (All 6 Paths)
// =============================================================================

TEST(SparseRepresentationsTest, CompleteConversionRoundtrips) {
    // Build an arbitrary reference matrix with duplicates, negatives, unsorted
    COOMatrix original_coo(4, 5);
    original_coo.add_entry(3, 4, -8.0);
    original_coo.add_entry(0, 1, 3.2);
    original_coo.add_entry(2, 2, -4.5);
    original_coo.add_entry(1, 0, 1.1);
    original_coo.add_entry(0, 1, -1.2); // Duplicate at (0, 1) -> 2.0
    original_coo.add_entry(3, 2, 7.8);
    original_coo.add_entry(1, 3, -6.3);

    // 1. COO -> CSR
    CSRMatrix csr1 = original_coo.to_csr();
    csr1.validate();

    // 2. COO -> CSC
    CSCMatrix csc1 = original_coo.to_csc();
    csc1.validate();

    // Verify mathematical identity between initial conversions
    EXPECT_TRUE(csr1.equals(csc1));
    EXPECT_TRUE(original_coo.equals(csr1));
    EXPECT_TRUE(original_coo.equals(csc1));

    // 3. CSR -> CSC (Direct linear-time)
    CSCMatrix csc_from_csr = csr1.to_csc();
    csc_from_csr.validate();
    EXPECT_TRUE(csc_from_csr.equals(csr1));
    EXPECT_TRUE(csc_from_csr.equals(csc1));

    // 4. CSC -> CSR (Direct linear-time)
    CSRMatrix csr_from_csc = csc1.to_csr();
    csr_from_csc.validate();
    EXPECT_TRUE(csr_from_csc.equals(csc1));
    EXPECT_TRUE(csr_from_csc.equals(csr1));

    // 5. CSR -> COO
    COOMatrix coo_from_csr = csr1.to_coo();
    EXPECT_TRUE(coo_from_csr.equals(csr1));

    // 6. CSC -> COO
    COOMatrix coo_from_csc = csc1.to_coo();
    EXPECT_TRUE(coo_from_csc.equals(csc1));

    // Check coefficient-by-coefficient consistency across ALL coordinates (i, j)
    for (index_t i = 0; i < 4; ++i) {
        for (index_t j = 0; j < 5; ++j) {
            scalar_t v_coo = original_coo.coeff(i, j);
            scalar_t v_csr = csr1.coeff(i, j);
            scalar_t v_csc = csc1.coeff(i, j);
            scalar_t v_csr2 = csr_from_csc.coeff(i, j);
            scalar_t v_csc2 = csc_from_csr.coeff(i, j);

            EXPECT_DOUBLE_EQ(v_coo, v_csr);
            EXPECT_DOUBLE_EQ(v_csr, v_csc);
            EXPECT_DOUBLE_EQ(v_csc, v_csr2);
            EXPECT_DOUBLE_EQ(v_csr2, v_csc2);
        }
    }
}
