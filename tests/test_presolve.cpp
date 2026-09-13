#include <gtest/gtest.h>
#include <pipepye/presolve/presolve_types.hpp>
#include <pipepye/presolve/presolve_pass.hpp>
#include <pipepye/presolve/presolve_pass_manager.hpp>
#include <pipepye/presolve/postsolve.hpp>
#include <pipepye/model/mps_parser.hpp>
#include <filesystem>
#include <vector>
#include <string>
#include <cmath>

using namespace pipepye;
using namespace pipepye::model;
using namespace pipepye::presolve;

namespace {

std::string find_netlib_file(const std::string& filename) {
    std::vector<std::string> search_dirs = {
        "tests/data/mps/netlib",
        "../tests/data/mps/netlib",
        "../../tests/data/mps/netlib",
        "/home/satyansh/pipepye/tests/data/mps/netlib"
    };
    for (const auto& dir : search_dirs) {
        std::filesystem::path p = std::filesystem::path(dir) / filename;
        if (std::filesystem::exists(p)) {
            return p.string();
        }
    }
    return "";
}

LinearProgram create_simple_lp(index_t num_rows, index_t num_cols,
                              const std::vector<scalar_t>& c,
                              const std::vector<scalar_t>& col_lb,
                              const std::vector<scalar_t>& col_ub,
                              const std::vector<scalar_t>& row_lb,
                              const std::vector<scalar_t>& row_ub,
                              const std::vector<std::tuple<index_t, index_t, scalar_t>>& entries) {
    LinearProgram lp;
    lp.name = "simple_test_lp";
    lp.obj_offset = 0.0;
    lp.is_maximization = false;

    for (index_t j = 0; j < num_cols; ++j) {
        std::string name = "x" + std::to_string(j + 1);
        lp.col_names.push_back(name);
        lp.col_name_to_idx[name] = j;
    }
    for (index_t i = 0; i < num_rows; ++i) {
        std::string name = "c" + std::to_string(i + 1);
        lp.row_names.push_back(name);
        lp.row_name_to_idx[name] = i;
        lp.row_senses.push_back(RowSense::LessEqual);
    }

    lp.c = c;
    lp.col_lower = col_lb;
    lp.col_upper = col_ub;
    lp.var_types.assign(num_cols, VariableType::Continuous);
    lp.row_lower = row_lb;
    lp.row_upper = row_ub;

    sparse::COOMatrix coo(num_rows, num_cols);
    for (const auto& [i, j, v] : entries) {
        coo.add_entry(i, j, v);
    }
    lp.A_coo = coo;

    auto csr = coo.to_csr();
    lp.csr_row_ptr.assign(csr.row_ptr().begin(), csr.row_ptr().end());
    lp.csr_col_ind.assign(csr.col_ind().begin(), csr.col_ind().end());
    lp.csr_values.assign(csr.values().begin(), csr.values().end());

    auto csc = coo.to_csc();
    lp.csc_col_ptr.assign(csc.col_ptr().begin(), csc.col_ptr().end());
    lp.csc_row_ind.assign(csc.row_ind().begin(), csc.row_ind().end());
    lp.csc_values.assign(csc.values().begin(), csc.values().end());

    return lp;
}

} // namespace

// =============================================================================
// SECTION 1: Empty Rows & Columns
// =============================================================================

TEST(PresolveTest, EmptyRowRedundantIsRemoved) {
    // 2 rows, 2 cols. Row 1 is empty: 0 <= 0 <= 10 (satisfied vacuously)
    auto lp = create_simple_lp(
        2, 2,
        {1.0, 2.0},
        {0.0, 0.0}, {10.0, 10.0},
        {0.0, 0.0}, {10.0, 10.0},
        {{0, 0, 1.0}, {0, 1, 1.0}} // Row 1 has no entries
    );

    PresolvePassManager ppm;
    auto res = ppm.run(lp);
    ASSERT_TRUE(res.is_ok());

    auto presolved = res.value();
    EXPECT_EQ(presolved.status, PresolveStatus::Reduced);
    EXPECT_EQ(presolved.lp.num_rows(), 1);
    EXPECT_EQ(presolved.lp.num_cols(), 2);
    EXPECT_EQ(presolved.stats.pass_history[0].rows_removed, 1);
}

TEST(PresolveTest, EmptyRowInfeasibleDetected) {
    // Row 1 is empty: 5 <= 0 <= 10 (impossible!)
    auto lp = create_simple_lp(
        2, 2,
        {1.0, 2.0},
        {0.0, 0.0}, {10.0, 10.0},
        {0.0, 5.0}, {10.0, 10.0},
        {{0, 0, 1.0}, {0, 1, 1.0}} // Row 1 has no entries
    );

    PresolvePassManager ppm;
    auto res = ppm.run(lp);
    ASSERT_TRUE(res.is_ok());

    auto presolved = res.value();
    EXPECT_TRUE(presolved.is_infeasible());
    EXPECT_EQ(presolved.status, PresolveStatus::Infeasible);
}

TEST(PresolveTest, EmptyColumnPositiveCostFixedToLowerBound) {
    // 3 variables: x1, x2, x3. Row 0: x1 + x3 <= 10.
    // Col 1 (x2) has no constraints. Min 3*x2, with 2 <= x2 <= 10.
    // Optimal x2 = 2, obj offset increases by 3*2 = 6.
    // x1 and x3 remain in the active problem.
    auto lp = create_simple_lp(
        1, 3,
        {1.0, 3.0, 2.0},
        {0.0, 2.0, 0.0}, {10.0, 10.0, 10.0},
        {0.0}, {10.0},
        {{0, 0, 1.0}, {0, 2, 1.0}} // Col 1 not in matrix
    );

    PresolvePassManager ppm;
    auto res = ppm.run(lp);
    ASSERT_TRUE(res.is_ok());

    auto presolved = res.value();
    EXPECT_EQ(presolved.status, PresolveStatus::Reduced);
    EXPECT_EQ(presolved.lp.num_cols(), 2);
    EXPECT_NEAR(presolved.lp.obj_offset, 6.0, 1e-9);
}

TEST(PresolveTest, EmptyColumnNegativeCostFixedToUpperBound) {
    // Col 1 (x2) has no constraints. Min -4*x2, with 0 <= x2 <= 7.
    // Optimal x2 = 7, obj offset increases by -4*7 = -28.
    auto lp = create_simple_lp(
        1, 3,
        {2.0, -4.0, 1.0},
        {0.0, 0.0, 0.0}, {10.0, 7.0, 10.0},
        {0.0}, {10.0},
        {{0, 0, 1.0}, {0, 2, 1.0}}
    );

    PresolvePassManager ppm;
    auto res = ppm.run(lp);
    ASSERT_TRUE(res.is_ok());

    auto presolved = res.value();
    EXPECT_EQ(presolved.status, PresolveStatus::Reduced);
    EXPECT_EQ(presolved.lp.num_cols(), 2);
    EXPECT_NEAR(presolved.lp.obj_offset, -28.0, 1e-9);
}

TEST(PresolveTest, EmptyColumnUnboundedDetected) {
    // Col 1 has cost -5.0 and upper bound +infinity -> unbounded!
    auto lp = create_simple_lp(
        1, 2,
        {1.0, -5.0},
        {0.0, 0.0}, {10.0, Infinity},
        {0.0}, {10.0},
        {{0, 0, 1.0}}
    );

    PresolvePassManager ppm;
    auto res = ppm.run(lp);
    ASSERT_TRUE(res.is_ok());

    auto presolved = res.value();
    EXPECT_TRUE(presolved.is_unbounded());
}

// =============================================================================
// SECTION 2: Fixed-Variable Elimination
// =============================================================================

TEST(PresolveTest, FixedVariableSubstitution) {
    // min x1 + 2*x2 + x3
    // s.t. 2*x1 + 3*x2 + x3 <= 20
    //      0 <= x1 <= 10
    //      4 <= x2 <= 4 (fixed variable)
    //      0 <= x3 <= 10
    // Substitution: x2 = 4
    // Row 0 upper bound: 20 - 3*4 = 8
    // Objective offset: 2*4 = 8
    auto lp = create_simple_lp(
        1, 3,
        {1.0, 2.0, 1.0},
        {0.0, 4.0, 0.0}, {10.0, 4.0, 10.0},
        {-Infinity}, {20.0},
        {{0, 0, 2.0}, {0, 1, 3.0}, {0, 2, 1.0}}
    );

    PresolvePassManager ppm;
    auto res = ppm.run(lp);
    ASSERT_TRUE(res.is_ok());

    auto presolved = res.value();
    EXPECT_EQ(presolved.status, PresolveStatus::Reduced);
    EXPECT_EQ(presolved.lp.num_cols(), 2);
    EXPECT_NEAR(presolved.lp.row_upper[0], 8.0, 1e-9);
    EXPECT_NEAR(presolved.lp.obj_offset, 8.0, 1e-9);

    // Test Postsolve
    PrimalDualSolution pre_sol;
    pre_sol.x = {2.0, 1.0}; // x1 = 2, x3 = 1
    pre_sol.y = {0.0};
    pre_sol.s = {1.0, 1.0};

    auto post_res = presolved.postsolve_mgr.postsolve(lp, pre_sol);
    ASSERT_TRUE(post_res.is_ok());

    auto full_sol = post_res.value();
    EXPECT_TRUE(full_sol.is_feasible);
    EXPECT_NEAR(full_sol.x[0], 2.0, 1e-9);
    EXPECT_NEAR(full_sol.x[1], 4.0, 1e-9); // Restored x2 = 4
    EXPECT_NEAR(full_sol.x[2], 1.0, 1e-9);
    EXPECT_NEAR(full_sol.objective_value, 1.0 * 2.0 + 2.0 * 4.0 + 1.0 * 1.0, 1e-9);
}

// =============================================================================
// SECTION 3: Singleton Row & Column Reductions
// =============================================================================

TEST(PresolveTest, SingletonRowTightensUpperBound) {
    // Row 0: 2*x1 <= 8  =>  x1 <= 4 (singleton row)
    // Row 1: x1 + x2 <= 10 (coupling row keeping x1, x2 active)
    // Original x1 in [0, 10] => tightened to [0, 4]
    // Row 0 is eliminated.
    auto lp = create_simple_lp(
        2, 2,
        {1.0, 1.0},
        {0.0, 0.0}, {10.0, 10.0},
        {-Infinity, -Infinity}, {8.0, 10.0},
        {{0, 0, 2.0}, {1, 0, 1.0}, {1, 1, 1.0}}
    );

    PresolvePassManager ppm;
    auto res = ppm.run(lp);
    ASSERT_TRUE(res.is_ok());

    auto presolved = res.value();
    EXPECT_EQ(presolved.status, PresolveStatus::Reduced);
    EXPECT_NEAR(presolved.lp.col_upper[0], 4.0, 1e-9);
    EXPECT_EQ(presolved.lp.num_rows(), 1); // Row 1 remains
    EXPECT_EQ(presolved.lp.num_cols(), 2);
}

TEST(PresolveTest, SingletonRowNegativeCoeffTightensLowerBound) {
    // Row 0: -3*x1 <= -12  =>  x1 >= 4
    // Row 1: x1 + x2 <= 10
    // Original x1 in [0, 10] => tightened to [4, 10]
    auto lp = create_simple_lp(
        2, 2,
        {1.0, 1.0},
        {0.0, 0.0}, {10.0, 10.0},
        {-Infinity, -Infinity}, {-12.0, 10.0},
        {{0, 0, -3.0}, {1, 0, 1.0}, {1, 1, 1.0}}
    );

    PresolvePassManager ppm;
    auto res = ppm.run(lp);
    ASSERT_TRUE(res.is_ok());

    auto presolved = res.value();
    EXPECT_EQ(presolved.status, PresolveStatus::Reduced);
    EXPECT_NEAR(presolved.lp.col_lower[0], 4.0, 1e-9);
    EXPECT_EQ(presolved.lp.num_rows(), 1);
}

TEST(PresolveTest, SingletonRowInfeasibleConflict) {
    // Row 0: 2*x1 >= 14  =>  x1 >= 7
    // Original x1 in [0, 5] => [7, 5] (empty interval!)
    auto lp = create_simple_lp(
        1, 1,
        {1.0},
        {0.0}, {5.0},
        {14.0}, {Infinity},
        {{0, 0, 2.0}}
    );

    PresolvePassManager ppm;
    auto res = ppm.run(lp);
    ASSERT_TRUE(res.is_ok());
    EXPECT_TRUE(res.value().is_infeasible());
}

TEST(PresolveTest, SingletonColumnSubstitutionInEquality) {
    // min 2*x1 + 4*x2 + 3*x3
    // s.t. x1 + 2*x2 + x3 = 10  (x2 appears only here)
    //      0 <= x1 <= 10, 0 <= x2 <= 10, 0 <= x3 <= 10
    // x2 = (10 - x1 - x3) / 2
    // Objective: 2*x1 + 4*(10 - x1 - x3)/2 + 3*x3 = 20 - x3
    // x2 eliminated, x1 and x3 remain in the active problem.
    auto lp = create_simple_lp(
        1, 3,
        {2.0, 4.0, 3.0},
        {0.0, 0.0, 0.0}, {10.0, 10.0, 10.0},
        {10.0}, {10.0},
        {{0, 0, 1.0}, {0, 1, 2.0}, {0, 2, 1.0}}
    );

    PresolvePassManager ppm;
    auto res = ppm.run(lp);
    ASSERT_TRUE(res.is_ok());

    auto presolved = res.value();
    EXPECT_NEAR(presolved.lp.obj_offset, 20.0, 1e-9);
    EXPECT_EQ(presolved.lp.num_cols(), 2); // x2 eliminated, x1 and x3 remain
}

// =============================================================================
// SECTION 4: Implied Bound Tightening & Forcing/Redundancy
// =============================================================================

TEST(PresolveTest, ImpliedBoundTighteningFromRow) {
    // 2*x1 + 4*x2 <= 20
    // with x1 in [0, 100], x2 in [2, 10]
    // Min activity of x2 is 4*2 = 8.
    // Implied: 2*x1 <= 20 - 8 = 12 => x1 <= 6.
    auto lp = create_simple_lp(
        1, 2,
        {1.0, 1.0},
        {0.0, 2.0}, {100.0, 10.0},
        {-Infinity}, {20.0},
        {{0, 0, 2.0}, {0, 1, 4.0}}
    );

    PresolvePassManager ppm;
    auto res = ppm.run(lp);
    ASSERT_TRUE(res.is_ok());

    auto presolved = res.value();
    EXPECT_NEAR(presolved.lp.col_upper[0], 6.0, 1e-7);
}

TEST(PresolveTest, RedundantConstraintEliminated) {
    // x1 + x2 <= 10
    // with 0 <= x1 <= 2, 0 <= x2 <= 3
    // Max activity is 2 + 3 = 5 <= 10 => constraint is redundant!
    auto lp = create_simple_lp(
        1, 2,
        {1.0, 1.0},
        {0.0, 0.0}, {2.0, 3.0},
        {-Infinity}, {10.0},
        {{0, 0, 1.0}, {0, 1, 1.0}}
    );

    PresolvePassManager ppm;
    auto res = ppm.run(lp);
    ASSERT_TRUE(res.is_ok());

    auto presolved = res.value();
    EXPECT_EQ(presolved.lp.num_rows(), 0); // Row removed as redundant
}

TEST(PresolveTest, ForcingConstraintFixesAllVariables) {
    // x1 + x2 <= 5
    // with x1 in [2, 10], x2 in [3, 10]
    // Min activity is 2 + 3 = 5 == upper bound 5.
    // Forces x1 = 2, x2 = 3!
    auto lp = create_simple_lp(
        1, 2,
        {1.0, 2.0},
        {2.0, 3.0}, {10.0, 10.0},
        {-Infinity}, {5.0},
        {{0, 0, 1.0}, {0, 1, 1.0}}
    );

    PresolvePassManager ppm;
    auto res = ppm.run(lp);
    ASSERT_TRUE(res.is_ok());

    auto presolved = res.value();
    EXPECT_EQ(presolved.status, PresolveStatus::OptimalSolved);
    EXPECT_EQ(presolved.lp.num_cols(), 0);
    EXPECT_NEAR(presolved.lp.obj_offset, 1.0 * 2.0 + 2.0 * 3.0, 1e-9);
}

TEST(PresolveTest, InfeasibleActivityDetected) {
    // x1 + x2 >= 20
    // with x1 in [0, 5], x2 in [0, 5]
    // Max activity is 10 < 20 => Infeasible!
    auto lp = create_simple_lp(
        1, 2,
        {1.0, 1.0},
        {0.0, 0.0}, {5.0, 5.0},
        {20.0}, {Infinity},
        {{0, 0, 1.0}, {0, 1, 1.0}}
    );

    PresolvePassManager ppm;
    auto res = ppm.run(lp);
    ASSERT_TRUE(res.is_ok());
    EXPECT_TRUE(res.value().is_infeasible());
}

// =============================================================================
// SECTION 5: Multi-Pass Cascades & Full Solution Reconstruction
// =============================================================================

TEST(PresolveTest, MultiPassReductionCascade) {
    // Model where:
    // x3 is fixed: 2 <= x3 <= 2
    // Row 0: x2 + x3 <= 5  =>  x2 <= 3 (turns into singleton row in pass 2)
    // Row 1: x1 + x2 <= 10 with x1 <= 2, x2 <= 3 (turns into redundant row in pass 3)
    auto lp = create_simple_lp(
        2, 3,
        {1.0, 1.0, 1.0},
        {0.0, 0.0, 2.0}, {2.0, 10.0, 2.0},
        {-Infinity, -Infinity}, {5.0, 10.0},
        {
            {0, 1, 1.0}, {0, 2, 1.0}, // Row 0: x2 + x3 <= 5
            {1, 0, 1.0}, {1, 1, 1.0}  // Row 1: x1 + x2 <= 10
        }
    );

    PresolvePassManager ppm;
    auto res = ppm.run(lp);
    ASSERT_TRUE(res.is_ok());

    auto presolved = res.value();
    EXPECT_TRUE(presolved.is_reduced());
    EXPECT_GE(presolved.stats.total_passes_executed, 2);
    // x3 was eliminated, Row 0 tightened x2 then was removed, Row 1 was redundant and removed
    // Then unconstrained x1, x2 with positive costs were fixed to lower bounds, achieving OptimalSolved!
    EXPECT_EQ(presolved.status, PresolveStatus::OptimalSolved);
    EXPECT_EQ(presolved.lp.num_cols(), 0);
}

TEST(PresolveTest, EndToEndPostsolveReconstruction) {
    // Complete 4-variable LP:
    // min 2*x1 + 3*x2 + 4*x3 + 5*x4
    // s.t. x1 + x2 <= 10
    //      x3 = 4 (fixed)
    //      x4 <= 5 (singleton row on x4)
    // Bounds: x1 in [0, 8], x2 in [0, 8], x3 in [4, 4], x4 in [0, 10]
    auto lp = create_simple_lp(
        3, 4,
        {2.0, 3.0, 4.0, 5.0},
        {0.0, 0.0, 4.0, 0.0}, {8.0, 8.0, 4.0, 10.0},
        {-Infinity, 4.0, -Infinity}, {10.0, 4.0, 5.0},
        {
            {0, 0, 1.0}, {0, 1, 1.0}, // Row 0
            {1, 2, 1.0},              // Row 1: x3 = 4
            {2, 3, 1.0}               // Row 2: x4 <= 5
        }
    );

    PresolvePassManager ppm;
    auto res = ppm.run(lp);
    ASSERT_TRUE(res.is_ok());

    auto presolved = res.value();
    EXPECT_TRUE(presolved.is_reduced());

    // Provide feasible solution to presolved model
    // Presolved model contains: x1, x2, and unconstrained/tightened x4
    PrimalDualSolution pre_sol;
    pre_sol.x.resize(presolved.lp.num_cols());
    for (size_t k = 0; k < pre_sol.x.size(); ++k) {
        pre_sol.x[k] = presolved.lp.col_lower[k];
    }
    pre_sol.y.assign(presolved.lp.num_rows(), 0.0);
    pre_sol.s.assign(presolved.lp.num_cols(), 0.0);

    auto post_res = presolved.postsolve_mgr.postsolve(lp, pre_sol);
    ASSERT_TRUE(post_res.is_ok());

    auto full_sol = post_res.value();
    EXPECT_TRUE(full_sol.is_feasible);
    EXPECT_EQ(full_sol.x.size(), 4);
    EXPECT_NEAR(full_sol.x[2], 4.0, 1e-9); // Fixed x3 restored
    EXPECT_LE(full_sol.x[3], 5.0 + 1e-7);  // x4 bound obeyed

    // Check objective parity
    scalar_t manual_obj = 2.0 * full_sol.x[0] + 3.0 * full_sol.x[1] +
                          4.0 * full_sol.x[2] + 5.0 * full_sol.x[3];
    EXPECT_NEAR(full_sol.objective_value, manual_obj, 1e-9);
}

// =============================================================================
// SECTION 6: Real-World Netlib LP Presolve Evaluation
// =============================================================================

TEST(PresolveTest, NetlibAFIRO) {
    std::string path = find_netlib_file("afiro.mps");
    if (path.empty()) GTEST_SKIP() << "Netlib afiro.mps not found";

    LinearProgram lp;
    ASSERT_TRUE(MPSParser::parse_file(path, lp).is_ok());
    EXPECT_EQ(lp.num_rows(), 27);
    EXPECT_EQ(lp.num_cols(), 32);

    PresolvePassManager ppm;
    auto res = ppm.run(lp);
    ASSERT_TRUE(res.is_ok());

    auto presolved = res.value();
    EXPECT_EQ(presolved.status, PresolveStatus::Reduced);
    EXPECT_LT(presolved.lp.num_rows(), 27);
    EXPECT_GT(presolved.stats.total_passes_executed, 0);
    EXPECT_GE(presolved.stats.row_reduction_pct(), 0.0);
}

TEST(PresolveTest, NetlibBEACONFD) {
    std::string path = find_netlib_file("beaconfd.mps");
    if (path.empty()) GTEST_SKIP() << "Netlib beaconfd.mps not found";

    LinearProgram lp;
    ASSERT_TRUE(MPSParser::parse_file(path, lp).is_ok());
    EXPECT_EQ(lp.num_rows(), 173);
    EXPECT_EQ(lp.num_cols(), 262);

    PresolvePassManager ppm;
    auto res = ppm.run(lp);
    ASSERT_TRUE(res.is_ok());

    auto presolved = res.value();
    EXPECT_EQ(presolved.status, PresolveStatus::Reduced);
    EXPECT_LT(presolved.lp.num_rows(), 173);
    EXPECT_LT(presolved.lp.num_cols(), 262);
    EXPECT_LT(presolved.lp.num_nonzeros(), lp.num_nonzeros());
}
