#include <gtest/gtest.h>
#include <pipepye/presolve/presolve_pass_manager.hpp>
#include <pipepye/presolve/presolve_oracle.hpp>
#include <pipepye/presolve/transformation_log.hpp>
#include <pipepye/model/lp_model.hpp>
#include <cmath>

using namespace pipepye;
using namespace pipepye::model;
using namespace pipepye::presolve;

namespace {

LinearProgram create_test_lp(
    index_t num_rows, index_t num_cols,
    const std::vector<scalar_t>& c,
    const std::vector<scalar_t>& col_lb, const std::vector<scalar_t>& col_ub,
    const std::vector<scalar_t>& row_lb, const std::vector<scalar_t>& row_ub,
    const std::vector<sparse::TripletF64>& triplets,
    scalar_t obj_offset = 0.0) {

    LinearProgram lp;
    lp.name = "edge_case_lp";
    lp.c = c;
    lp.col_lower = col_lb;
    lp.col_upper = col_ub;
    lp.row_lower = row_lb;
    lp.row_upper = row_ub;
    lp.obj_offset = obj_offset;

    for (index_t j = 0; j < num_cols; ++j) {
        lp.col_names.push_back("x" + std::to_string(j));
        lp.col_name_to_idx["x" + std::to_string(j)] = j;
        lp.var_types.push_back(VariableType::Continuous);
    }
    for (index_t i = 0; i < num_rows; ++i) {
        lp.row_names.push_back("c" + std::to_string(i));
        lp.row_name_to_idx["c" + std::to_string(i)] = i;
        lp.row_senses.push_back(RowSense::Ranged);
    }

    sparse::COOMatrix coo(num_rows, num_cols);
    for (const auto& tr : triplets) {
        coo.add_entry(tr.row, tr.col, tr.val);
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

constexpr scalar_t Inf = 1e20;

} // anonymous namespace

// =============================================================================
// Edge Case 1: Zero Coefficients in Matrix Nonzeros
// =============================================================================
TEST(PresolveEdgeCasesTest, ExplicitZeroCoefficientsIgnoredProperly) {
    // Model with explicit 0.0 coefficients added into triplet list.
    // Row 0: 0.0*x0 + 2.0*x1 <= 10.
    // x0 should have degree 0 in matrix!
    auto lp = create_test_lp(
        1, 2,
        {1.0, 2.0},
        {0.0, 0.0}, {10.0, 10.0},
        {-Inf}, {10.0},
        {
            {0, 0, 0.0}, // Explicit zero!
            {0, 1, 2.0}
        }
    );

    PresolvePassManager ppm;
    auto res = ppm.run(lp);
    ASSERT_TRUE(res.is_ok());
    auto presolved = res.value();

    // x0 is empty col with cost 1.0 > 0 -> fixed to lower bound 0.0
    // Row 0 becomes singleton on x1: 2.0*x1 <= 10 -> x1 in [0, 5]
    EXPECT_TRUE(presolved.is_reduced());
}

// =============================================================================
// Edge Case 2: Transformation Log Trace Mapping (original x17 -> fixed/eliminated -> reconstructed x17 = 4.2)
// =============================================================================
TEST(PresolveEdgeCasesTest, TransformationLogTraceVerification) {
    // Construct an 18-variable model where variable 17 has l_17 == u_17 == 4.2
    const int N = 18;
    std::vector<scalar_t> c(N, 1.0);
    std::vector<scalar_t> lb(N, 0.0);
    std::vector<scalar_t> ub(N, 10.0);
    lb[17] = 4.2;
    ub[17] = 4.2; // Fixed variable 17

    // Constraint: sum_{j=0}^{17} x_j <= 100
    std::vector<sparse::TripletF64> triplets;
    for (int j = 0; j < N; ++j) {
        triplets.push_back({0, j, 1.0});
    }

    auto lp = create_test_lp(1, N, c, lb, ub, {-Inf}, {100.0}, triplets);

    PresolvePassManager ppm;
    auto res = ppm.run(lp);
    ASSERT_TRUE(res.is_ok());
    auto presolved = res.value();

    EXPECT_EQ(presolved.lp.num_cols(), 17); // x17 was eliminated
    EXPECT_NEAR(presolved.lp.row_upper[0], 100.0 - 4.2, 1e-9);

    // Formulate a dummy presolved solution
    PrimalDualSolution pre_sol;
    pre_sol.x.assign(presolved.lp.num_cols(), 1.0);
    pre_sol.y.assign(presolved.lp.num_rows(), 0.0);
    pre_sol.s.assign(presolved.lp.num_cols(), 0.0);

    auto post_res = presolved.postsolve_mgr.postsolve(lp, pre_sol);
    ASSERT_TRUE(post_res.is_ok());
    auto full_sol = post_res.value();

    // Verify reconstructed value
    EXPECT_NEAR(full_sol.x[17], 4.2, 1e-9);

    // Query transformation log for variable 17
    std::string trace17 = presolved.postsolve_mgr.log().format_variable_trace(17);
    EXPECT_NE(trace17.find("original x17"), std::string::npos);
    EXPECT_NE(trace17.find("fixed/eliminated"), std::string::npos);
    EXPECT_NE(trace17.find("reconstructed x17 = 4.2"), std::string::npos);

    // Full log verification
    std::string full_log = presolved.postsolve_mgr.log().format_full_log();
    EXPECT_NE(full_log.find("original x17"), std::string::npos);
}

// =============================================================================
// Edge Case 3: Chained Fixed-Variable Substitutions
// =============================================================================
TEST(PresolveEdgeCasesTest, ChainedFixedVariableSubstitutions) {
    // x0 in [3, 3] (fixed)
    // Row 0: x0 + x1 = 7  => x1 = 4 (turns into fixed in pass 2)
    // Row 1: x1 + x2 = 10 => x2 = 6 (turns into fixed in pass 3)
    auto lp = create_test_lp(
        2, 3,
        {1.0, 2.0, 3.0},
        {3.0, 0.0, 0.0}, {3.0, 10.0, 10.0},
        {7.0, 10.0}, {7.0, 10.0},
        {
            {0, 0, 1.0}, {0, 1, 1.0},
            {1, 1, 1.0}, {1, 2, 1.0}
        }
    );

    PresolvePassManager ppm;
    auto res = ppm.run(lp);
    ASSERT_TRUE(res.is_ok());
    auto presolved = res.value();

    EXPECT_EQ(presolved.status, PresolveStatus::OptimalSolved);
    EXPECT_EQ(presolved.lp.num_cols(), 0);

    PrimalDualSolution pre_sol;
    auto post_res = presolved.postsolve_mgr.postsolve(lp, pre_sol);
    ASSERT_TRUE(post_res.is_ok());
    auto sol = post_res.value();
    EXPECT_TRUE(sol.is_feasible);
    EXPECT_NEAR(sol.x[0], 3.0, 1e-9);
    EXPECT_NEAR(sol.x[1], 4.0, 1e-9);
    EXPECT_NEAR(sol.x[2], 6.0, 1e-9);
}

// =============================================================================
// Edge Case 4: Presolve Correctness Oracle Feasibility & Objective Parity
// =============================================================================
TEST(PresolveEdgeCasesTest, PresolveOracleVerifiesFeasibilityAndObjective) {
    // 3-variable LP with redundant and singleton rows
    // min 2*x0 + 3*x1 + 4*x2
    // Row 0: x0 + x1 <= 10
    // Row 1: x2 <= 5 (singleton row on x2)
    // Row 2: x0 + x1 <= 20 (redundant since Row 0 is <= 10)
    auto lp = create_test_lp(
        3, 3,
        {2.0, 3.0, 4.0},
        {0.0, 0.0, 0.0}, {8.0, 8.0, 8.0},
        {-Inf, -Inf, -Inf}, {10.0, 5.0, 20.0},
        {
            {0, 0, 1.0}, {0, 1, 1.0},
            {1, 2, 1.0},
            {2, 0, 1.0}, {2, 1, 1.0}
        }
    );

    PresolvePassManager ppm;
    auto res = ppm.run(lp);
    ASSERT_TRUE(res.is_ok());
    auto presolved = res.value();

    EXPECT_TRUE(presolved.is_reduced());

    PresolveOracle oracle;
    auto oracle_res = oracle.verify_feasibility_and_objective(lp, presolved);
    EXPECT_TRUE(oracle_res.passed) << oracle_res.format_report();
    EXPECT_GT(oracle_res.feasible_samples_found, 0);
    EXPECT_EQ(oracle_res.bound_violations, 0);
    EXPECT_EQ(oracle_res.constraint_violations, 0);
    EXPECT_EQ(oracle_res.objective_mismatches, 0);
}

// =============================================================================
// Edge Case 5: Small LP Optimality Verification via Oracle
// =============================================================================
TEST(PresolveEdgeCasesTest, SmallLPOptimalityVerification) {
    // 2-variable LP
    // min x0 + 2*x1
    // s.t. x0 + x1 >= 4
    //      x0 in [0, 5], x1 in [0, 5]
    auto lp = create_test_lp(
        1, 2,
        {1.0, 2.0},
        {0.0, 0.0}, {5.0, 5.0},
        {4.0}, {Inf},
        {{0, 0, 1.0}, {0, 1, 1.0}}
    );

    PresolvePassManager ppm;
    auto res = ppm.run(lp);
    ASSERT_TRUE(res.is_ok());
    auto presolved = res.value();

    PresolveOracle oracle;
    auto oracle_res = oracle.verify_small_lp_optimality(lp, presolved);
    EXPECT_TRUE(oracle_res.passed) << oracle_res.format_report();
}

// =============================================================================
// Edge Case 6: Presolve Oracle Infeasibility Detection Consistency
// =============================================================================
TEST(PresolveEdgeCasesTest, PresolveOracleInfeasibilityConsistency) {
    // Infeasible: x0 + x1 >= 50 with x0 in [0, 5], x1 in [0, 5]
    auto lp = create_test_lp(
        1, 2,
        {1.0, 1.0},
        {0.0, 0.0}, {5.0, 5.0},
        {50.0}, {Inf},
        {{0, 0, 1.0}, {0, 1, 1.0}}
    );

    PresolvePassManager ppm;
    auto res = ppm.run(lp);
    ASSERT_TRUE(res.is_ok());
    auto presolved = res.value();
    EXPECT_TRUE(presolved.is_infeasible());

    PresolveOracle oracle;
    auto oracle_res = oracle.verify_infeasibility_consistency(lp, presolved);
    EXPECT_TRUE(oracle_res.passed) << oracle_res.format_report();
}

// =============================================================================
// Edge Case 7: Quantitative Model Reduction Measurements
// =============================================================================
TEST(PresolveEdgeCasesTest, QuantitativeModelReductionMetrics) {
    // Model with 5 rows and 5 columns
    // - Row 0: empty with 0 in bounds (redundant)
    // - Row 1: singleton 2*x1 <= 6
    // - Row 2: x2 fixed in [3, 3]
    // - Row 3: redundant x3 + x4 <= 100 with x3, x4 in [0, 10]
    // - Row 4: coupling row x1 + x4 <= 20
    auto lp = create_test_lp(
        5, 5,
        {1.0, 2.0, 3.0, 4.0, 5.0},
        {0.0, 0.0, 3.0, 0.0, 0.0}, {10.0, 10.0, 3.0, 10.0, 10.0},
        {-Inf, -Inf, -Inf, -Inf, -Inf}, {10.0, 6.0, 20.0, 100.0, 20.0},
        {
            // Row 0 has NO nonzeros (empty)
            {1, 1, 2.0},              // Row 1: 2*x1 <= 6
            {2, 2, 1.0},              // Row 2: x2
            {3, 3, 1.0}, {3, 4, 1.0}, // Row 3: x3 + x4 <= 100
            {4, 1, 1.0}, {4, 4, 1.0}  // Row 4: x1 + x4 <= 20
        }
    );

    PresolvePassManager ppm;
    auto res = ppm.run(lp);
    ASSERT_TRUE(res.is_ok());
    auto presolved = res.value();

    const auto& stats = presolved.stats;
    EXPECT_EQ(stats.initial_rows, 5);
    EXPECT_EQ(stats.initial_cols, 5);
    EXPECT_LT(stats.final_rows, 5);
    EXPECT_GT(stats.eliminated_rows(), 0);
    EXPECT_GE(stats.total_bounds_tightened(), 1);
    EXPECT_GE(stats.total_variables_fixed(), 1);
    EXPECT_GT(stats.row_reduction_pct(), 0.0);

    // Verify format_summary output contains key fields
    std::string summary = stats.format_summary();
    EXPECT_NE(summary.find("PIPEPYE PRESOLVE REDUCTION SUMMARY"), std::string::npos);
    EXPECT_NE(summary.find("eliminated:"), std::string::npos);
    EXPECT_NE(summary.find("Bounds Tightened:"), std::string::npos);
}
