#include <gtest/gtest.h>
#include <pipepye/sparse/matrix_generator.hpp>
#include <pipepye/sparse/csr_matrix.hpp>
#include <pipepye/sparse/csc_matrix.hpp>

using namespace pipepye::sparse;
using pipepye::index_t;
using pipepye::scalar_t;

TEST(MatrixGeneratorTest, RandomMatrixGenerationAndDeterminism) {
    COOMatrix coo1 = MatrixGenerator::generate_random(50, 40, 0.05, -5.0, 5.0, 12345);
    COOMatrix coo2 = MatrixGenerator::generate_random(50, 40, 0.05, -5.0, 5.0, 12345);

    EXPECT_EQ(coo1.num_rows(), 50);
    EXPECT_EQ(coo1.num_cols(), 40);
    EXPECT_GT(coo1.num_nonzeros(), 50);
    EXPECT_LT(coo1.num_nonzeros(), 200);

    // Exact deterministic reproducibility
    EXPECT_TRUE(coo1.equals(coo2));

    CSRMatrix csr = coo1.to_csr();
    csr.validate();
    EXPECT_TRUE(coo1.equals(csr));

    // Error checking
    EXPECT_THROW(MatrixGenerator::generate_random(-1, 40, 0.05), std::invalid_argument);
    EXPECT_THROW(MatrixGenerator::generate_random(50, 40, 0.0), std::invalid_argument);
    EXPECT_THROW(MatrixGenerator::generate_random(50, 40, 1.5), std::invalid_argument);
}

TEST(MatrixGeneratorTest, BandedMatrixGeneration) {
    // 20 x 20 matrix with lower bandwidth 1 (tridiagonal lower) and upper bandwidth 2
    COOMatrix coo = MatrixGenerator::generate_banded(20, 20, 1, 2, 1.0, 5.0, 999);
    EXPECT_EQ(coo.num_rows(), 20);
    EXPECT_EQ(coo.num_cols(), 20);

    CSRMatrix csr = coo.to_csr();
    csr.validate();

    // Verify all nonzeros lie strictly within the [-1, 2] diagonal band
    for (index_t i = 0; i < csr.num_rows(); ++i) {
        auto r = csr.row(i);
        for (index_t c : r.col_indices) {
            index_t diag_offset = c - i;
            EXPECT_GE(diag_offset, -1);
            EXPECT_LE(diag_offset, 2);
        }
    }

    EXPECT_THROW(MatrixGenerator::generate_banded(20, 20, -1, 2), std::invalid_argument);
}

TEST(MatrixGeneratorTest, BlockDiagonalMatrixGeneration) {
    // 4 blocks, each 5 x 5 (total 20 x 20) with no coupling
    COOMatrix coo_decoupled = MatrixGenerator::generate_block_diagonal(4, 5, 5, 0.5, 0.0, 1.0, 2.0, 42);
    EXPECT_EQ(coo_decoupled.num_rows(), 20);
    EXPECT_EQ(coo_decoupled.num_cols(), 20);

    CSRMatrix csr = coo_decoupled.to_csr();
    csr.validate();

    // Verify all nonzeros lie within their respective 5x5 diagonal blocks
    for (index_t i = 0; i < csr.num_rows(); ++i) {
        index_t my_block = i / 5;
        auto r = csr.row(i);
        for (index_t c : r.col_indices) {
            index_t col_block = c / 5;
            EXPECT_EQ(my_block, col_block);
        }
    }

    // Now test with off-diagonal coupling
    COOMatrix coo_coupled = MatrixGenerator::generate_block_diagonal(4, 5, 5, 0.5, 0.05, 1.0, 2.0, 42);
    EXPECT_GT(coo_coupled.num_nonzeros(), coo_decoupled.num_nonzeros());
}

TEST(MatrixGeneratorTest, StaircaseMatrixGeneration) {
    // 5 stages, each stage 4 rows and 3 cols -> total rows = 5*4 = 20, total cols = 6*3 = 18
    COOMatrix coo = MatrixGenerator::generate_staircase(5, 4, 3, 0.4, 1.0, 3.0, 777);
    EXPECT_EQ(coo.num_rows(), 20);
    EXPECT_EQ(coo.num_cols(), 18);

    CSRMatrix csr = coo.to_csr();
    csr.validate();

    // Verify staircase coupling: row i in stage s only couples with columns in stage s and s+1
    for (index_t i = 0; i < csr.num_rows(); ++i) {
        index_t s = i / 4;
        index_t min_col = s * 3;
        index_t max_col = (s + 2) * 3 - 1;

        auto r = csr.row(i);
        for (index_t c : r.col_indices) {
            EXPECT_GE(c, min_col);
            EXPECT_LE(c, max_col);
        }
    }
}

TEST(MatrixGeneratorTest, IrregularMatrixGeneration) {
    // 100 x 100 matrix with 500 total nonzeros
    // Hub rows: 5% of rows (5 rows) contain 50% of nonzeros (250 nonzeros) -> ~50 nonzeros per hub row
    // Remaining 95 rows share 250 nonzeros -> ~2.6 nonzeros per regular row
    COOMatrix coo = MatrixGenerator::generate_irregular(100, 100, 500, 0.05, 0.50, 1.0, 2.0, 888);
    EXPECT_EQ(coo.num_rows(), 100);
    EXPECT_EQ(coo.num_cols(), 100);
    EXPECT_EQ(coo.num_nonzeros(), 500);

    CSRMatrix csr = coo.to_csr();
    csr.validate();

    // Check degree of hub rows vs regular rows
    size_t hub_nnz_count = 0;
    for (index_t i = 0; i < 5; ++i) {
        hub_nnz_count += csr.row_nnz(i);
        EXPECT_GT(csr.row_nnz(i), 15); // Each hub row is heavily populated
    }
    EXPECT_EQ(hub_nnz_count, 250);

    size_t regular_nnz_count = 0;
    for (index_t i = 5; i < 100; ++i) {
        regular_nnz_count += csr.row_nnz(i);
    }
    EXPECT_EQ(regular_nnz_count, 250);
}
