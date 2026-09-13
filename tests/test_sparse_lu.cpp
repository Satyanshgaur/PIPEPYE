#include <gtest/gtest.h>
#include <pipepye/factorization/sparse_lu.hpp>
#include <pipepye/factorization/dense_lu.hpp>
#include <cmath>
#include <vector>

using namespace pipepye;
using namespace pipepye::factorization;

class SparseLUTest : public ::testing::Test {
protected:
    static constexpr scalar_t kTol = 1e-9;
};

TEST_F(SparseLUTest, DiagonalMatrixFactorizeAndSolve) {
    // 4x4 diagonal matrix
    // col_ptr: [0, 1, 2, 3, 4]
    // row_ind: [0, 1, 2, 3]
    // values:  [2.0, -3.0, 5.0, 0.5]
    std::vector<index_t> col_ptr = {0, 1, 2, 3, 4};
    std::vector<index_t> row_ind = {0, 1, 2, 3};
    std::vector<scalar_t> values = {2.0, -3.0, 5.0, 0.5};

    SparseLU lu(4);
    Status s = lu.factorize(4, col_ptr, row_ind, values);
    ASSERT_TRUE(s.is_ok());
    EXPECT_TRUE(lu.is_factorized());

    std::vector<scalar_t> b = {4.0, 9.0, 15.0, 1.0};
    std::vector<scalar_t> x = lu.solve_ftran(b);

    ASSERT_EQ(x.size(), 4u);
    EXPECT_NEAR(x[0], 2.0, kTol);
    EXPECT_NEAR(x[1], -3.0, kTol);
    EXPECT_NEAR(x[2], 3.0, kTol);
    EXPECT_NEAR(x[3], 2.0, kTol);

    // BTRAN: diagonal is self-adjoint
    std::vector<scalar_t> y = lu.solve_btran(b);
    EXPECT_NEAR(y[0], 2.0, kTol);
    EXPECT_NEAR(y[1], -3.0, kTol);
    EXPECT_NEAR(y[2], 3.0, kTol);
    EXPECT_NEAR(y[3], 2.0, kTol);
}

TEST_F(SparseLUTest, TridiagonalMatrixFactorizeAndSolve) {
    // 3x3 tridiagonal:
    // [ 2, -1,  0 ]
    // [-1,  2, -1 ]
    // [ 0, -1,  2 ]
    // CSC format:
    // col 0: (0, 2.0), (1, -1.0)
    // col 1: (0, -1.0), (1, 2.0), (2, -1.0)
    // col 2: (1, -1.0), (2, 2.0)
    std::vector<index_t> col_ptr = {0, 2, 5, 7};
    std::vector<index_t> row_ind = {0, 1, 0, 1, 2, 1, 2};
    std::vector<scalar_t> values = {2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0};

    SparseLU lu(3);
    Status s = lu.factorize(3, col_ptr, row_ind, values);
    ASSERT_TRUE(s.is_ok());

    // True x = [1, 2, 3]^T
    // b = A * x = [ 2 - 2, -1 + 4 - 3, -2 + 6 ] = [0, 0, 4]^T
    std::vector<scalar_t> b = {0.0, 0.0, 4.0};
    std::vector<scalar_t> x = lu.solve_ftran(b);

    ASSERT_EQ(x.size(), 3u);
    EXPECT_NEAR(x[0], 1.0, kTol);
    EXPECT_NEAR(x[1], 2.0, kTol);
    EXPECT_NEAR(x[2], 3.0, kTol);

    // Symmetric => BTRAN solves A * y = b as well
    std::vector<scalar_t> y = lu.solve_btran(b);
    EXPECT_NEAR(y[0], 1.0, kTol);
    EXPECT_NEAR(y[1], 2.0, kTol);
    EXPECT_NEAR(y[2], 3.0, kTol);

    const auto& m = lu.metrics();
    EXPECT_GT(m.orig_nnz, 0u);
    EXPECT_GE(m.fill_in_ratio, 0.0);
}

TEST_F(SparseLUTest, ParityAgainstDenseLUOracle) {
    // 3x3 non-symmetric matrix
    // A = [ 1,  2, -1 ]
    //     [ 2,  1, -2 ]
    //     [-3,  1,  1 ]
    std::vector<scalar_t> dense_A = {
         1.0,  2.0, -1.0,
         2.0,  1.0, -2.0,
        -3.0,  1.0,  1.0
    };

    // In CSC format:
    // col 0: (0, 1), (1, 2), (2, -3)
    // col 1: (0, 2), (1, 1), (2, 1)
    // col 2: (0, -1), (1, -2), (2, 1)
    std::vector<index_t> col_ptr = {0, 3, 6, 9};
    std::vector<index_t> row_ind = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<scalar_t> values = {1.0, 2.0, -3.0, 2.0, 1.0, 1.0, -1.0, -2.0, 1.0};

    DenseLU dense_lu(3);
    ASSERT_TRUE(dense_lu.factorize(dense_A).is_ok());

    SparseLU sparse_lu(3);
    ASSERT_TRUE(sparse_lu.factorize(3, col_ptr, row_ind, values).is_ok());

    std::vector<scalar_t> b = {7.0, -3.0, 5.0};
    std::vector<scalar_t> x_dense = dense_lu.solve_ftran(b);
    std::vector<scalar_t> x_sparse = sparse_lu.solve_ftran(b);

    for (size_t i = 0; i < 3; ++i) {
        EXPECT_NEAR(x_sparse[i], x_dense[i], kTol);
    }

    std::vector<scalar_t> y_dense = dense_lu.solve_btran(b);
    std::vector<scalar_t> y_sparse = sparse_lu.solve_btran(b);

    for (size_t i = 0; i < 3; ++i) {
        EXPECT_NEAR(y_sparse[i], y_dense[i], kTol);
    }
}

TEST_F(SparseLUTest, SingularMatrixReturnsFailure) {
    // 2x2 singular
    std::vector<index_t> col_ptr = {0, 2, 4};
    std::vector<index_t> row_ind = {0, 1, 0, 1};
    std::vector<scalar_t> values = {1.0, 2.0, 2.0, 4.0};

    SparseLU lu(2);
    Status s = lu.factorize(2, col_ptr, row_ind, values);
    EXPECT_FALSE(s.is_ok());
    EXPECT_FALSE(lu.is_factorized());
}
