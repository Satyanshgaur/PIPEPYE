#include <pipepye/analysis/solver_selector.hpp>

namespace pipepye::analysis {

SolverRecommendation StructureAwareSelector::recommend(
    const model::LinearProgram& lp,
    const ProblemStats& stats) {

    SolverRecommendation rec;

    // 1. Integrality Check
    index_t int_vars = 0;
    for (index_t j = 0; j < lp.num_cols(); ++j) {
        if (j < static_cast<index_t>(lp.var_types.size())) {
            auto vt = lp.var_types[j];
            if (vt == model::VariableType::Binary || vt == model::VariableType::Integer) {
                int_vars++;
            }
        }
    }

    if (int_vars > 0) {
        rec.solver = SolverCandidate::BranchAndBound;
        rec.backend = BackendCandidate::CPU;
        rec.confidence = 0.95;
        rec.rationale = "Model contains " + std::to_string(int_vars) +
                        " discrete/binary variables requiring combinatorial branch-and-bound search. "
                        "Warm-started Dual Simplex provides crucial pivot efficiency during tree node exploration.";
        return rec;
    }

    // 2. Large Scale Block-Angular / Staircase Systems
    if (stats.num_nonzeros >= 30000 && stats.staircase_score >= 0.45) {
        rec.solver = SolverCandidate::PDHG_GPU;
        rec.backend = BackendCandidate::GPU;
        rec.confidence = 0.90;
        rec.rationale = "Large block-angular / staircase structure with " + std::to_string(stats.num_nonzeros) +
                        " NNZ and staircase score " + std::to_string(stats.staircase_score) +
                        " maps exceptionally well to GPU memory bandwidth and data-parallel SpMV.";
        return rec;
    }

    // 3. Very Large Scale General Sparse Systems
    if (stats.num_nonzeros >= 50000) {
        rec.solver = SolverCandidate::PDHG_GPU;
        rec.backend = BackendCandidate::GPU;
        rec.confidence = 0.85;
        rec.rationale = "Large-scale sparse LP (" + std::to_string(stats.num_nonzeros) +
                        " NNZ) crosses CPU/GPU crossover threshold, favoring GPU-accelerated first-order PDHG.";
        return rec;
    }

    // 4. Dense Coupling or Compact Dimensions
    if (stats.row_length_gini >= 0.35 || stats.density > 0.01 || stats.num_rows < 5000) {
        rec.solver = SolverCandidate::DualSimplex;
        rec.backend = BackendCandidate::CPU;
        rec.confidence = 0.90;
        rec.rationale = "Coupled constraint rows, high row imbalance, or compact dimensions (" +
                        std::to_string(stats.num_rows) + " rows, " + std::to_string(stats.num_cols) +
                        " cols) favor CPU Dual Simplex with sparse LU factorization over iterative GPU methods.";
        return rec;
    }

    // 5. Default Fallback
    rec.solver = SolverCandidate::DualSimplex;
    rec.backend = BackendCandidate::CPU;
    rec.confidence = 0.75;
    rec.rationale = "Moderate scale continuous LP operates efficiently within CPU cache boundaries, "
                    "favoring direct extreme-point simplex pivots.";
    return rec;
}

PredictionOutcome StructureAwareSelector::evaluate_outcome(
    const SolverRecommendation& rec,
    SolverCandidate actual_solver,
    BackendCandidate actual_backend) {

    if (rec.solver == actual_solver && rec.backend == actual_backend) {
        return PredictionOutcome::Confirmed;
    }
    if (rec.solver == actual_solver) {
        return PredictionOutcome::PartiallyConfirmed;
    }
    return PredictionOutcome::Refuted;
}

} // namespace pipepye::analysis
