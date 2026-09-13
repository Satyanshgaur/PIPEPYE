#include <gtest/gtest.h>
#include <pipepye/workloads/case_a_crude_blending.hpp>
#include <pipepye/workloads/case_b_multi_period_planning.hpp>
#include <pipepye/workloads/case_c_refinery_scheduling.hpp>
#include <pipepye/workloads/case_d_unit_commitment.hpp>
#include <pipepye/simplex/dual_simplex.hpp>

using namespace pipepye;
using namespace pipepye::workloads;

TEST(WorkloadGeneratorsTest, CaseACrudeBlendingGenerationAndSolve) {
    auto [lp, meta] = CrudeBlendingGenerator::generate(CrudeBlendingParams::ForScale(InstanceScale::Toy));

    EXPECT_EQ(meta.problem_name, "Crude_Blending");
    EXPECT_EQ(meta.scale, InstanceScale::Toy);
    EXPECT_EQ(meta.num_continuous_vars, lp.num_cols());
    EXPECT_EQ(meta.num_binary_vars, 0);
    EXPECT_EQ(meta.expected_solver, "DualSimplex");
    EXPECT_EQ(meta.expected_backend, "CPU");

    // 3 crudes, 2 products, 2 qualities -> cols = 3*2 + 2 = 8, rows = 2 + 3 + 2*2*2 = 13
    EXPECT_EQ(lp.num_cols(), 8);
    EXPECT_EQ(lp.num_rows(), 13);
    EXPECT_GT(lp.num_nonzeros(), 0);

    // Solve with Dual Simplex
    simplex::DualSimplexSolver solver;
    auto res = solver.solve(lp);
    EXPECT_TRUE(res.is_optimal());
    EXPECT_LT(res.objective_value, 0.0); // Revenue exceeds costs -> negative min objective
    EXPECT_LT(res.max_primal_infeasibility, 1e-5);
}

TEST(WorkloadGeneratorsTest, CaseBMultiPeriodPlanningGenerationAndSolve) {
    auto [lp, meta] = MultiPeriodPlanningGenerator::generate(MultiPeriodPlanningParams::ForScale(InstanceScale::Toy));

    EXPECT_EQ(meta.problem_name, "Multi_Period_Planning");
    EXPECT_EQ(meta.scale, InstanceScale::Toy);
    EXPECT_GT(meta.staircase_score, 0.40); // Demonstrates block-angular staircase structure

    // T=5, P=4, M=2 -> cols = 5 * (2*4) = 40, rows = 5 * (4 + 2 + 1) = 35
    EXPECT_EQ(lp.num_cols(), 40);
    EXPECT_EQ(lp.num_rows(), 35);

    // Solve with Dual Simplex
    simplex::DualSimplexSolver solver;
    auto res = solver.solve(lp);
    EXPECT_TRUE(res.is_optimal());
    EXPECT_GT(res.objective_value, 0.0);
}

TEST(WorkloadGeneratorsTest, CaseCRefinerySchedulingGeneration) {
    auto [milp, meta] = RefinerySchedulingGenerator::generate(RefinerySchedulingParams::ForScale(InstanceScale::Toy));

    EXPECT_EQ(meta.problem_name, "Refinery_Scheduling");
    EXPECT_GT(meta.num_binary_vars, 0);
    EXPECT_EQ(meta.expected_solver, "BranchAndBound");
    EXPECT_EQ(meta.expected_backend, "CPU");

    // U=2, M=2, T=3
    // vars per t: 2*2*2 + 2 = 10 -> cols = 3 * 10 = 30
    EXPECT_EQ(milp.num_cols(), 30);
    EXPECT_EQ(meta.num_binary_vars, 12); // U * M * T = 2 * 2 * 3 = 12
    EXPECT_GT(milp.num_rows(), 0);
}

TEST(WorkloadGeneratorsTest, CaseDUnitCommitmentGeneration) {
    auto [milp, meta] = UnitCommitmentGenerator::generate(UnitCommitmentParams::ForScale(InstanceScale::Toy));

    EXPECT_EQ(meta.problem_name, "Unit_Commitment");
    EXPECT_GT(meta.num_binary_vars, 0);
    EXPECT_EQ(meta.expected_solver, "BranchAndBound");

    // G=3, T=4 -> cols = 4 * 2 * 3 = 24
    EXPECT_EQ(milp.num_cols(), 24);
    EXPECT_EQ(meta.num_binary_vars, 12); // G * T = 3 * 4 = 12
    EXPECT_GT(milp.num_rows(), 0);
}
