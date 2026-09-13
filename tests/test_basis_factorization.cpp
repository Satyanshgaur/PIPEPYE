#include <gtest/gtest.h>
#include <pipepye/factorization/basis_factorization.hpp>
#include <cmath>
#include <vector>

using namespace pipepye;
using namespace pipepye::factorization;

class BasisFactorizationTest : public ::testing::Test {
protected:
    static constexpr scalar_t kTol = 1e-9;
};

TEST_F(BasisFactorizationTest, SlackInitialBasis) {
    // m = 3, n = 2
    // Matrix A: 3x2
    // [ 1, 2 ]
    // [ 0, 3 ]
    // [ 4, 0 ]
    std::vector<index_t> col_ptr = {0, 2, 4};
    std::vector<index_t> row_ind = {0, 2, 0, 1};
    std::vector<scalar_t> values = {1.0, 4.0, 2.0, 3.0};
    sparse::CSCMatrix A(3, 2, col_ptr, row_ind, values);

    // Initial slack basis: basic_vars = [2, 3, 4] (slacks for rows 0, 1, 2)
    // Slack columns are -e_0, -e_1, -e_2 => B = -I_3
    std::vector<index_t> basic_vars = {2, 3, 4};

    BasisFactorization fact(3);
    Status s = fact.factorize(A, basic_vars, 2);
    ASSERT_TRUE(s.is_ok());
    EXPECT_TRUE(fact.is_factorized());
    EXPECT_EQ(fact.num_updates(), 0u);

    // B * x = b => -I * x = b => x = -b
    std::vector<scalar_t> b = {5.0, -7.0, 3.0};
    std::vector<scalar_t> x = fact.solve_ftran(b);
    EXPECT_NEAR(x[0], -5.0, kTol);
    EXPECT_NEAR(x[1], 7.0, kTol);
    EXPECT_NEAR(x[2], -3.0, kTol);

    // B^T * y = b => -I * y = b => y = -b
    std::vector<scalar_t> y = fact.solve_btran(b);
    EXPECT_NEAR(y[0], -5.0, kTol);
    EXPECT_NEAR(y[1], 7.0, kTol);
    EXPECT_NEAR(y[2], -3.0, kTol);
}

TEST_F(BasisFactorizationTest, PFIUpdateParityAgainstRefactorize) {
    // m = 2, n = 2
    // A = [ 2,  1 ]
    //     [ 1,  3 ]
    std::vector<index_t> col_ptr = {0, 2, 4};
    std::vector<index_t> row_ind = {0, 1, 0, 1};
    std::vector<scalar_t> values = {2.0, 1.0, 1.0, 3.0};
    sparse::CSCMatrix A(2, 2, col_ptr, row_ind, values);

    // Start with slacks: basic_vars = [2, 3] => B = -I
    std::vector<index_t> basic_vars = {2, 3};
    BasisFactorization fact(2);
    ASSERT_TRUE(fact.factorize(A, basic_vars, 2).is_ok());

    // Pivot 0: Replace slack basic_vars[0] with structural column 0 (A_0 = [2, 1]^T)
    // FTRAN of entering column A_0 through current B = -I:
    // v = B^{-1} A_0 = -I * [2, 1]^T = [-2, -1]^T
    std::vector<scalar_t> v = fact.solve_ftran({2.0, 1.0});
    EXPECT_NEAR(v[0], -2.0, kTol);
    EXPECT_NEAR(v[1], -1.0, kTol);

    // Update with pivot_row = 0
    Status us = fact.update(0, v);
    ASSERT_TRUE(us.is_ok());
    EXPECT_EQ(fact.num_updates(), 1u);

    // The new basis matrix is B_new = [ A_0, -e_1 ] = [ 2,  0 ]
    //                                                [ 1, -1 ]
    // Let's test FTRAN on b = [6, 5]^T:
    // B_new * x = b => 2 * x_0 = 6 => x_0 = 3.
    // 1 * 3 - x_1 = 5 => x_1 = -2.
    std::vector<scalar_t> b = {6.0, 5.0};
    std::vector<scalar_t> x_pfi = fact.solve_ftran(b);
    EXPECT_NEAR(x_pfi[0], 3.0, kTol);
    EXPECT_NEAR(x_pfi[1], -2.0, kTol);

    // Compare with direct fresh factorization of new basis basic_vars = [0, 3]
    BasisFactorization ref_fact(2);
    std::vector<index_t> new_basic_vars = {0, 3};
    ASSERT_TRUE(ref_fact.factorize(A, new_basic_vars, 2).is_ok());
    std::vector<scalar_t> x_ref = ref_fact.solve_ftran(b);
    EXPECT_NEAR(x_pfi[0], x_ref[0], kTol);
    EXPECT_NEAR(x_pfi[1], x_ref[1], kTol);

    // Test BTRAN: B_new^T * y = c
    // B_new^T = [ 2,  1 ]
    //           [ 0, -1 ]
    // Let c = [5, -2]^T => -y_1 = -2 => y_1 = 2.
    // 2 * y_0 + 1 * 2 = 5 => y_0 = 1.5.
    std::vector<scalar_t> c = {5.0, -2.0};
    std::vector<scalar_t> y_pfi = fact.solve_btran(c);
    EXPECT_NEAR(y_pfi[0], 1.5, kTol);
    EXPECT_NEAR(y_pfi[1], 2.0, kTol);

    std::vector<scalar_t> y_ref = ref_fact.solve_btran(c);
    EXPECT_NEAR(y_pfi[0], y_ref[0], kTol);
    EXPECT_NEAR(y_pfi[1], y_ref[1], kTol);
}

TEST_F(BasisFactorizationTest, RefactorizePolicyThreshold) {
    BasisFactorization fact(2);
    EXPECT_FALSE(fact.needs_refactorize(5));

    // Fake 5 updates
    std::vector<scalar_t> col = {1.0, 0.5};
    for (int i = 0; i < 5; ++i) {
        ASSERT_TRUE(fact.update(0, col).is_ok());
    }
    EXPECT_TRUE(fact.needs_refactorize(5));
    fact.clear_updates();
    EXPECT_EQ(fact.num_updates(), 0u);
    EXPECT_FALSE(fact.needs_refactorize(5));
}
