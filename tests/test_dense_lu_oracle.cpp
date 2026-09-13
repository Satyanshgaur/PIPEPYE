#include <gtest/gtest.h>
#include <pipepye/factorization/dense_lu.hpp>
#include <cmath>
#include <vector>

using namespace pipepye;
using namespace pipepye::factorization;

class DenseLUTest : public ::testing::Test {
protected:
    static constexpr scalar_t kTol = 1e-10;
};

TEST_F(DenseLUTest, FactorizeAndSolve2x2) {
    // A = [ 2,  1 ]
    //     [ 5,  7 ]
    // det(A) = 14 - 5 = 9 != 0
    std::vector<scalar_t> A = {
        2.0, 1.0,
        5.0, 7.0
    };

    DenseLU lu(2);
    Status s = lu.factorize(A);
    ASSERT_TRUE(s.is_ok());
    EXPECT_TRUE(lu.is_factorized());
    EXPECT_GT(lu.min_pivot(), 0.0);

    // FTRAN: A * x = b => b = [11, 28]^T => x = [7, -3]? No, 2*7 + 1*(-3) = 11, 5*7 + 7*(-3) = 14 != 28.
    // Let's pick known x = [3, 2]^T => b = [2*3 + 1*2, 5*3 + 7*2] = [8, 29]^T.
    std::vector<scalar_t> b = {8.0, 29.0};
    std::vector<scalar_t> x = lu.solve_ftran(b);

    ASSERT_EQ(x.size(), 2u);
    EXPECT_NEAR(x[0], 3.0, kTol);
    EXPECT_NEAR(x[1], 2.0, kTol);

    // BTRAN: A^T * y = c => A^T = [2, 5; 1, 7].
    // Let y = [1, 2]^T => c = [2*1 + 5*2, 1*1 + 7*2] = [12, 15]^T.
    std::vector<scalar_t> c = {12.0, 15.0};
    std::vector<scalar_t> y = lu.solve_btran(c);

    ASSERT_EQ(y.size(), 2u);
    EXPECT_NEAR(y[0], 1.0, kTol);
    EXPECT_NEAR(y[1], 2.0, kTol);
}

TEST_F(DenseLUTest, FactorizeAndSolve3x3) {
    // A = [ 1,  2, -1 ]
    //     [ 2,  1, -2 ]
    //     [-3,  1,  1 ]
    std::vector<scalar_t> A = {
         1.0,  2.0, -1.0,
         2.0,  1.0, -2.0,
        -3.0,  1.0,  1.0
    };

    DenseLU lu(3);
    Status s = lu.factorize(A);
    ASSERT_TRUE(s.is_ok());

    // True x = [1, 2, 3]^T
    // b = A * x = [ 1 + 4 - 3, 2 + 2 - 6, -3 + 2 + 3 ] = [2, -2, 2]^T
    std::vector<scalar_t> b = {2.0, -2.0, 2.0};
    std::vector<scalar_t> x = lu.solve_ftran(b);

    ASSERT_EQ(x.size(), 3u);
    EXPECT_NEAR(x[0], 1.0, kTol);
    EXPECT_NEAR(x[1], 2.0, kTol);
    EXPECT_NEAR(x[2], 3.0, kTol);

    // BTRAN verification: A^T * y = c
    // True y = [2, -1, 1]^T
    // c = A^T * y = [ 1*2 + 2*(-1) + (-3)*1, 2*2 + 1*(-1) + 1*1, -1*2 + (-2)*(-1) + 1*1 ]
    //   = [ 2 - 2 - 3, 4 - 1 + 1, -2 + 2 + 1 ] = [-3, 4, 1]^T
    std::vector<scalar_t> c = {-3.0, 4.0, 1.0};
    std::vector<scalar_t> y = lu.solve_btran(c);

    ASSERT_EQ(y.size(), 3u);
    EXPECT_NEAR(y[0], 2.0, kTol);
    EXPECT_NEAR(y[1], -1.0, kTol);
    EXPECT_NEAR(y[2], 1.0, kTol);
}

TEST_F(DenseLUTest, SingularMatrixDetection) {
    // Linearly dependent rows: row 2 is 2 * row 1
    std::vector<scalar_t> A = {
        1.0, 2.0,
        2.0, 4.0
    };

    DenseLU lu(2);
    Status s = lu.factorize(A);
    EXPECT_FALSE(s.is_ok());
    EXPECT_FALSE(lu.is_factorized());
}

TEST_F(DenseLUTest, IdentityMatrix) {
    std::vector<scalar_t> A = {
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0
    };

    DenseLU lu(3);
    Status s = lu.factorize(A);
    ASSERT_TRUE(s.is_ok());

    std::vector<scalar_t> b = {4.0, 5.0, 6.0};
    std::vector<scalar_t> x = lu.solve_ftran(b);
    EXPECT_NEAR(x[0], 4.0, kTol);
    EXPECT_NEAR(x[1], 5.0, kTol);
    EXPECT_NEAR(x[2], 6.0, kTol);
}
