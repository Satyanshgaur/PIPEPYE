#include <gtest/gtest.h>
#include <pipepye/milp/branch_and_bound.hpp>
#include <pipepye/workloads/case_c_refinery_scheduling.hpp>
#include <pipepye/workloads/case_d_unit_commitment.hpp>

using namespace pipepye;
using namespace pipepye::model;
using namespace pipepye::milp;

namespace {

LinearProgram make_small_ip() {
    // Problem:
    // min -x1 - x2
    // s.t. 2 x1 + x2 <= 5
    //     -4 x1 + 4 x2 <= 5
    //      x1, x2 >= 0, integer
    // LP relaxation: x1 = 1.25, x2 = 2.5, obj = -3.75
    // Integer optimum: x1 = 1, x2 = 2, obj = -3.0 (or x1=2, x2=1, obj = -3.0)
    LinearProgram lp;
    lp.name = "SMALL_IP";
    lp.is_maximization = false;
    lp.col_names = {"X1", "X2"};
    lp.col_name_to_idx["X1"] = 0;
    lp.col_name_to_idx["X2"] = 1;
    lp.c = {-1.0, -1.0};
    lp.col_lower = {0.0, 0.0};
    lp.col_upper = {10.0, 10.0};
    lp.var_types = {VariableType::Integer, VariableType::Integer};

    lp.row_names = {"C1", "C2"};
    lp.row_name_to_idx["C1"] = 0;
    lp.row_name_to_idx["C2"] = 1;
    lp.row_lower = {-Infinity, -Infinity};
    lp.row_upper = {5.0, 5.0};
    lp.row_senses = {RowSense::LessEqual, RowSense::LessEqual};

    lp.A_coo = sparse::COOMatrix(2, 2);
    lp.A_coo.add_entry(0, 0, 2.0);
    lp.A_coo.add_entry(0, 1, 1.0);
    lp.A_coo.add_entry(1, 0, -4.0);
    lp.A_coo.add_entry(1, 1, 4.0);

    auto csc = lp.A_coo.to_csc();
    lp.csc_col_ptr = csc.col_ptr_vector();
    lp.csc_row_ind = csc.row_ind_vector();
    lp.csc_values = csc.values_vector();

    auto csr = lp.A_coo.to_csr();
    lp.csr_row_ptr = csr.row_ptr_vector();
    lp.csr_col_ind = csr.col_ind_vector();
    lp.csr_values = csr.values_vector();

    return lp;
}

LinearProgram make_multivar_knapsack() {
    // min -7 x1 - 9 x2 - 5 x3 - 8 x4 - 4 x5
    // s.t. 3 x1 + 4 x2 + 2 x3 + 3 x4 + 2 x5 <= 8
    //      xi in {0, 1}
    LinearProgram lp;
    lp.name = "KNAPSACK_5";
    lp.is_maximization = false;
    lp.col_names = {"X1", "X2", "X3", "X4", "X5"};
    for (size_t i = 0; i < 5; ++i) {
        lp.col_name_to_idx[lp.col_names[i]] = static_cast<index_t>(i);
    }
    lp.c = {-7.0, -9.0, -5.0, -8.0, -4.0};
    lp.col_lower = {0.0, 0.0, 0.0, 0.0, 0.0};
    lp.col_upper = {1.0, 1.0, 1.0, 1.0, 1.0};
    lp.var_types = {VariableType::Binary, VariableType::Binary, VariableType::Binary,
                    VariableType::Binary, VariableType::Binary};

    lp.row_names = {"CAPACITY"};
    lp.row_name_to_idx["CAPACITY"] = 0;
    lp.row_lower = {-Infinity};
    lp.row_upper = {8.0};
    lp.row_senses = {RowSense::LessEqual};

    lp.A_coo = sparse::COOMatrix(1, 5);
    lp.A_coo.add_entry(0, 0, 3.0);
    lp.A_coo.add_entry(0, 1, 4.0);
    lp.A_coo.add_entry(0, 2, 2.0);
    lp.A_coo.add_entry(0, 3, 3.0);
    lp.A_coo.add_entry(0, 4, 2.0);

    auto csc = lp.A_coo.to_csc();
    lp.csc_col_ptr = csc.col_ptr_vector();
    lp.csc_row_ind = csc.row_ind_vector();
    lp.csc_values = csc.values_vector();

    auto csr = lp.A_coo.to_csr();
    lp.csr_row_ptr = csr.row_ptr_vector();
    lp.csr_col_ind = csr.col_ind_vector();
    lp.csr_values = csr.values_vector();

    return lp;
}

} // namespace

TEST(MILPPhase7Test, RootPresolveTightening) {
    LinearProgram lp;
    lp.name = "PRESOLVE_TEST";
    lp.col_names = {"X1", "X2", "X3"};
    lp.col_lower = {0.3, -1.8, 0.0};
    lp.col_upper = {4.7, 3.2, 5.0};
    lp.var_types = {VariableType::Integer, VariableType::Integer, VariableType::Continuous};

    // Singleton row: 2.0 * X1 <= 5.5 => X1 <= floor(5.5 / 2) = floor(2.75) = 2.0
    lp.row_names = {"R1"};
    lp.row_lower = {-Infinity};
    lp.row_upper = {5.5};
    lp.row_senses = {RowSense::LessEqual};

    lp.A_coo = sparse::COOMatrix(1, 3);
    lp.A_coo.add_entry(0, 0, 2.0);

    auto csc = lp.A_coo.to_csc();
    lp.csc_col_ptr = csc.col_ptr_vector();
    lp.csc_row_ind = csc.row_ind_vector();
    lp.csc_values = csc.values_vector();

    auto csr = lp.A_coo.to_csr();
    lp.csr_row_ptr = csr.row_ptr_vector();
    lp.csr_col_ind = csr.col_ind_vector();
    lp.csr_values = csr.values_vector();

    Status status = BranchAndBoundSolver::apply_root_presolve(lp);
    ASSERT_TRUE(status.is_ok());

    // X1 bounds tightened by integrality and singleton: [1.0, 2.0]
    EXPECT_NEAR(lp.col_lower[0], 1.0, 1e-6);
    EXPECT_NEAR(lp.col_upper[0], 2.0, 1e-6);

    // X2 bounds tightened by integrality: [-1.0, 3.0]
    EXPECT_NEAR(lp.col_lower[1], -1.0, 1e-6);
    EXPECT_NEAR(lp.col_upper[1], 3.0, 1e-6);

    // X3 remains continuous: [0.0, 5.0]
    EXPECT_NEAR(lp.col_lower[2], 0.0, 1e-6);
    EXPECT_NEAR(lp.col_upper[2], 5.0, 1e-6);
}

TEST(MILPPhase7Test, GomoryCutGenerationAndBoundImprovement) {
    LinearProgram lp = make_small_ip();

    // 1. Solve LP relaxation first
    simplex::DualSimplexSolver simplex(simplex::SimplexConfig::Fast());
    auto root_res = simplex.solve(lp, std::nullopt);
    ASSERT_TRUE(root_res.is_optimal());
    EXPECT_NEAR(root_res.objective_value, -3.75, 1e-4);

    // 2. Generate Gomory cuts
    std::vector<index_t> int_vars = {0, 1};

    int cuts = BranchAndBoundSolver::generate_gomory_cuts(lp, root_res, int_vars, 5);
    EXPECT_GT(cuts, 0);

    // 3. Solve with Gomory cuts
    auto cut_res = simplex.solve(lp, root_res.final_basis);
    ASSERT_TRUE(cut_res.is_optimal());

    // Gomory cut should strictly improve dual bound (increase objective towards integer optimum)
    EXPECT_GT(cut_res.objective_value, root_res.objective_value + 1e-4);
    EXPECT_LE(cut_res.objective_value, -3.0 + 1e-4); // Should not exceed IP optimum
}

TEST(MILPPhase7Test, SimpleRoundingHeuristic) {
    LinearProgram lp = make_multivar_knapsack();
    // Relaxed continuous solution has fractional values
    simplex::DualSimplexSolver simplex(simplex::SimplexConfig::Fast());
    auto root_res = simplex.solve(lp, std::nullopt);
    ASSERT_TRUE(root_res.is_optimal());

    std::vector<index_t> int_vars = {0, 1, 2, 3, 4};
    auto rounded_sol = BranchAndBoundSolver::run_simple_rounding(lp, root_res.x, int_vars);
    if (rounded_sol.has_value()) {
        // If rounded solution is feasible, check all integer variables are integral
        for (index_t j : int_vars) {
            scalar_t val = rounded_sol.value()[j];
            EXPECT_NEAR(val, std::round(val), 1e-5);
            EXPECT_GE(val, lp.col_lower[j] - 1e-5);
            EXPECT_LE(val, lp.col_upper[j] + 1e-5);
        }
    }
}

TEST(MILPPhase7Test, FractionalDivingHeuristic) {
    LinearProgram lp = make_multivar_knapsack();
    simplex::DualSimplexSolver simplex(simplex::SimplexConfig::Fast());
    auto root_res = simplex.solve(lp, std::nullopt);
    ASSERT_TRUE(root_res.is_optimal());

    std::vector<index_t> int_vars = {0, 1, 2, 3, 4};
    auto dive_sol = BranchAndBoundSolver::run_diving_heuristic(
        lp, root_res.x, int_vars, simplex, root_res.final_basis, 10);

    if (dive_sol.has_value()) {
        for (index_t j : int_vars) {
            scalar_t val = dive_sol.value()[j];
            EXPECT_NEAR(val, std::round(val), 1e-5);
        }
    }
}

TEST(MILPPhase7Test, NodeSelectionComparison) {
    LinearProgram lp = make_multivar_knapsack();

    // 1. BestBound
    MILPConfig cfg_bb;
    cfg_bb.node_selection = NodeSelectionStrategy::BestBound;
    BranchAndBoundSolver solver_bb(cfg_bb);
    MILPResult res_bb = solver_bb.solve(lp);

    // 2. DepthFirst
    MILPConfig cfg_dfs;
    cfg_dfs.node_selection = NodeSelectionStrategy::DepthFirst;
    BranchAndBoundSolver solver_dfs(cfg_dfs);
    MILPResult res_dfs = solver_dfs.solve(lp);

    // 3. BestEstimate
    MILPConfig cfg_be;
    cfg_be.node_selection = NodeSelectionStrategy::BestEstimate;
    BranchAndBoundSolver solver_be(cfg_be);
    MILPResult res_be = solver_be.solve(lp);

    ASSERT_TRUE(res_bb.is_optimal());
    ASSERT_TRUE(res_dfs.is_optimal());
    ASSERT_TRUE(res_be.is_optimal());

    // All node selection strategies must reach the same optimal objective value
    EXPECT_NEAR(res_bb.best_objective, res_dfs.best_objective, 1e-4);
    EXPECT_NEAR(res_bb.best_objective, res_be.best_objective, 1e-4);
}

TEST(MILPPhase7Test, PseudoCostBranchingVsMostFractional) {
    LinearProgram lp = make_multivar_knapsack();

    MILPConfig cfg_mf;
    cfg_mf.branching_strategy = BranchingStrategy::MostFractional;
    BranchAndBoundSolver solver_mf(cfg_mf);
    MILPResult res_mf = solver_mf.solve(lp);

    MILPConfig cfg_pc;
    cfg_pc.branching_strategy = BranchingStrategy::PseudoCost;
    BranchAndBoundSolver solver_pc(cfg_pc);
    MILPResult res_pc = solver_pc.solve(lp);

    ASSERT_TRUE(res_mf.is_optimal());
    ASSERT_TRUE(res_pc.is_optimal());
    EXPECT_NEAR(res_mf.best_objective, res_pc.best_objective, 1e-4);
}

TEST(MILPPhase7Test, MilestoneTrackingAndGapHistory) {
    LinearProgram lp = make_multivar_knapsack();

    MILPConfig cfg = MILPConfig::Advanced();
    BranchAndBoundSolver solver(cfg);
    MILPResult res = solver.solve(lp);

    ASSERT_TRUE(res.is_optimal());
    EXPECT_GT(res.nodes_explored, 0);
    EXPECT_GE(res.time_to_first_incumbent_ms, 0.0);
    EXPECT_GE(res.total_time_ms, 0.0);
    EXPECT_GT(res.nodes_per_second, 0.0);

    std::string summary = res.format_summary();
    EXPECT_NE(summary.find("Branch-and-Bound MILP Result"), std::string::npos);
    EXPECT_NE(summary.find("Best Incumbent"), std::string::npos);
    EXPECT_NE(summary.find("Nodes Explored"), std::string::npos);
}
