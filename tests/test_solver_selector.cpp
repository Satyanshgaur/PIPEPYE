#include <gtest/gtest.h>
#include <pipepye/analysis/solver_selector.hpp>
#include <pipepye/analysis/problem_analyzer.hpp>
#include <pipepye/workloads/case_a_crude_blending.hpp>
#include <pipepye/workloads/case_b_multi_period_planning.hpp>
#include <pipepye/workloads/case_c_refinery_scheduling.hpp>
#include <pipepye/workloads/case_d_unit_commitment.hpp>

using namespace pipepye;
using namespace pipepye::analysis;
using namespace pipepye::workloads;

TEST(SolverSelectorTest, CrudeBlendingSelectsCPUSimplex) {
    auto [lp, meta] = CrudeBlendingGenerator::generate(CrudeBlendingParams::ForScale(InstanceScale::Small));
    ProblemStats stats = ProblemAnalyzer::analyze(lp);

    auto rec = StructureAwareSelector::recommend(lp, stats);
    EXPECT_EQ(rec.solver, SolverCandidate::DualSimplex);
    EXPECT_EQ(rec.backend, BackendCandidate::CPU);
    EXPECT_FALSE(rec.rationale.empty());

    auto outcome = StructureAwareSelector::evaluate_outcome(rec, SolverCandidate::DualSimplex, BackendCandidate::CPU);
    EXPECT_EQ(outcome, PredictionOutcome::Confirmed);
}

TEST(SolverSelectorTest, LargeMultiPeriodSelectsGPUPDHG) {
    auto [lp, meta] = MultiPeriodPlanningGenerator::generate(MultiPeriodPlanningParams::ForScale(InstanceScale::Medium));
    ProblemStats stats = ProblemAnalyzer::analyze(lp);

    auto rec = StructureAwareSelector::recommend(lp, stats);
    if (stats.num_nonzeros >= 30000 && stats.staircase_score >= 0.45) {
        EXPECT_EQ(rec.solver, SolverCandidate::PDHG_GPU);
        EXPECT_EQ(rec.backend, BackendCandidate::GPU);
    }
}

TEST(SolverSelectorTest, RefinerySchedulingSelectsBranchAndBound) {
    auto [milp, meta] = RefinerySchedulingGenerator::generate(RefinerySchedulingParams::ForScale(InstanceScale::Small));
    ProblemStats stats = ProblemAnalyzer::analyze(milp);

    auto rec = StructureAwareSelector::recommend(milp, stats);
    EXPECT_EQ(rec.solver, SolverCandidate::BranchAndBound);
    EXPECT_EQ(rec.backend, BackendCandidate::CPU);
}

TEST(SolverSelectorTest, UnitCommitmentSelectsBranchAndBound) {
    auto [milp, meta] = UnitCommitmentGenerator::generate(UnitCommitmentParams::ForScale(InstanceScale::Small));
    ProblemStats stats = ProblemAnalyzer::analyze(milp);

    auto rec = StructureAwareSelector::recommend(milp, stats);
    EXPECT_EQ(rec.solver, SolverCandidate::BranchAndBound);
    EXPECT_EQ(rec.backend, BackendCandidate::CPU);
}

TEST(SolverSelectorTest, PredictionOutcomeClassification) {
    SolverRecommendation rec;
    rec.solver = SolverCandidate::DualSimplex;
    rec.backend = BackendCandidate::CPU;

    EXPECT_EQ(StructureAwareSelector::evaluate_outcome(rec, SolverCandidate::DualSimplex, BackendCandidate::CPU),
              PredictionOutcome::Confirmed);
    EXPECT_EQ(StructureAwareSelector::evaluate_outcome(rec, SolverCandidate::DualSimplex, BackendCandidate::GPU),
              PredictionOutcome::PartiallyConfirmed);
    EXPECT_EQ(StructureAwareSelector::evaluate_outcome(rec, SolverCandidate::PDHG_GPU, BackendCandidate::GPU),
              PredictionOutcome::Refuted);
}
