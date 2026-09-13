#include <gtest/gtest.h>
#include <pipepye/milp/branch_and_bound.hpp>
#include <pipepye/workloads/case_c_refinery_scheduling.hpp>

using namespace pipepye;
using namespace pipepye::model;
using namespace pipepye::milp;

TEST(BranchAndBoundTest, KnapsackExactIntegerOptimum) {
    // Problem:
    // min -5 x1 - 6 x2 - 3 x3
    // s.t. 2 x1 + 3 x2 + 2 x3 <= 4
    //      x1, x2, x3 in {0, 1}
    LinearProgram lp;
    lp.name = "KNAPSACK";
    lp.is_maximization = false;
    lp.obj_name = "OBJ";

    lp.col_names = {"X1", "X2", "X3"};
    lp.col_name_to_idx["X1"] = 0;
    lp.col_name_to_idx["X2"] = 1;
    lp.col_name_to_idx["X3"] = 2;
    lp.c = {-5.0, -6.0, -3.0};
    lp.col_lower = {0.0, 0.0, 0.0};
    lp.col_upper = {1.0, 1.0, 1.0};
    lp.var_types = {VariableType::Binary, VariableType::Binary, VariableType::Binary};

    lp.row_names = {"CAP"};
    lp.row_name_to_idx["CAP"] = 0;
    lp.row_lower = {-Infinity};
    lp.row_upper = {4.0};
    lp.row_senses = {RowSense::LessEqual};

    lp.A_coo = sparse::COOMatrix(1, 3);
    lp.A_coo.add_entry(0, 0, 2.0);
    lp.A_coo.add_entry(0, 1, 3.0);
    lp.A_coo.add_entry(0, 2, 2.0);

    auto csc = lp.A_coo.to_csc();
    lp.csc_col_ptr = csc.col_ptr_vector();
    lp.csc_row_ind = csc.row_ind_vector();
    lp.csc_values = csc.values_vector();

    auto csr = lp.A_coo.to_csr();
    lp.csr_row_ptr = csr.row_ptr_vector();
    lp.csr_col_ind = csr.col_ind_vector();
    lp.csr_values = csr.values_vector();

    BranchAndBoundSolver solver;
    MILPResult res = solver.solve(lp);

    EXPECT_TRUE(res.is_optimal());
    EXPECT_NEAR(res.best_objective, -8.0, 1e-4); // x1=1, x3=1 -> 5 + 3 = 8
    EXPECT_NEAR(res.x[0], 1.0, 1e-4);
    EXPECT_NEAR(res.x[1], 0.0, 1e-4);
    EXPECT_NEAR(res.x[2], 1.0, 1e-4);
    EXPECT_GT(res.nodes_explored, 1);
}

TEST(BranchAndBoundTest, WarmStartPivotReductionAblation) {
    LinearProgram lp;
    lp.name = "MILP_WARM_TEST";
    lp.col_names = {"X1", "X2", "X3"};
    lp.col_name_to_idx["X1"] = 0;
    lp.col_name_to_idx["X2"] = 1;
    lp.col_name_to_idx["X3"] = 2;
    lp.c = {-4.0, -5.0, -2.0};
    lp.col_lower = {0.0, 0.0, 0.0};
    lp.col_upper = {1.0, 1.0, 1.0};
    lp.var_types = {VariableType::Binary, VariableType::Binary, VariableType::Binary};

    lp.row_names = {"C1"};
    lp.row_name_to_idx["C1"] = 0;
    lp.row_lower = {-Infinity};
    lp.row_upper = {3.0};
    lp.row_senses = {RowSense::LessEqual};

    lp.A_coo = sparse::COOMatrix(1, 3);
    lp.A_coo.add_entry(0, 0, 2.0);
    lp.A_coo.add_entry(0, 1, 2.0);
    lp.A_coo.add_entry(0, 2, 1.0);

    auto csc = lp.A_coo.to_csc();
    lp.csc_col_ptr = csc.col_ptr_vector();
    lp.csc_row_ind = csc.row_ind_vector();
    lp.csc_values = csc.values_vector();

    auto csr = lp.A_coo.to_csr();
    lp.csr_row_ptr = csr.row_ptr_vector();
    lp.csr_col_ind = csr.col_ind_vector();
    lp.csr_values = csr.values_vector();

    BranchAndBoundSolver solver;
    auto [warm_res, cold_res] = solver.solve_warm_vs_cold(lp);

    EXPECT_TRUE(warm_res.is_optimal());
    EXPECT_TRUE(cold_res.is_optimal());
    EXPECT_NEAR(warm_res.best_objective, cold_res.best_objective, 1e-4);
    // Warm start should require fewer or equal pivots per node
    EXPECT_LE(warm_res.warm_start_pivots, cold_res.cold_start_pivots);
}

TEST(BranchAndBoundTest, PureContinuousModelFallback) {
    LinearProgram lp;
    lp.name = "PURE_LP";
    lp.col_names = {"X1", "X2"};
    lp.c = {1.0, 2.0};
    lp.col_lower = {0.0, 0.0};
    lp.col_upper = {5.0, 5.0};
    lp.var_types = {VariableType::Continuous, VariableType::Continuous};

    lp.row_names = {"C1"};
    lp.row_lower = {3.0};
    lp.row_upper = {Infinity};
    lp.row_senses = {RowSense::GreaterEqual};

    lp.A_coo = sparse::COOMatrix(1, 2);
    lp.A_coo.add_entry(0, 0, 1.0);
    lp.A_coo.add_entry(0, 1, 1.0);

    auto csc = lp.A_coo.to_csc();
    lp.csc_col_ptr = csc.col_ptr_vector();
    lp.csc_row_ind = csc.row_ind_vector();
    lp.csc_values = csc.values_vector();

    auto csr = lp.A_coo.to_csr();
    lp.csr_row_ptr = csr.row_ptr_vector();
    lp.csr_col_ind = csr.col_ind_vector();
    lp.csr_values = csr.values_vector();

    BranchAndBoundSolver solver;
    MILPResult res = solver.solve(lp);

    EXPECT_TRUE(res.is_optimal());
    EXPECT_NEAR(res.best_objective, 3.0, 1e-4); // x1=3, x2=0
    EXPECT_EQ(res.nodes_explored, 1);
}
