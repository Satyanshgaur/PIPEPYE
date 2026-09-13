#pragma once

#include <pipepye/model/lp_model.hpp>
#include <pipepye/analysis/problem_stats.hpp>
#include <string>

namespace pipepye::analysis {

enum class SolverCandidate : uint8_t {
    DualSimplex = 0,
    PDHG_CPU,
    PDHG_GPU,
    BranchAndBound
};

[[nodiscard]] inline std::string to_string(SolverCandidate sc) {
    switch (sc) {
        case SolverCandidate::DualSimplex:    return "DualSimplex";
        case SolverCandidate::PDHG_CPU:       return "PDHG_CPU";
        case SolverCandidate::PDHG_GPU:       return "PDHG_GPU";
        case SolverCandidate::BranchAndBound: return "BranchAndBound";
    }
    return "Unknown";
}

enum class BackendCandidate : uint8_t {
    CPU = 0,
    GPU
};

[[nodiscard]] inline std::string to_string(BackendCandidate bc) {
    switch (bc) {
        case BackendCandidate::CPU: return "CPU";
        case BackendCandidate::GPU: return "GPU";
    }
    return "Unknown";
}

enum class PredictionOutcome : uint8_t {
    Confirmed = 0,
    PartiallyConfirmed,
    Refuted,
    Inconclusive
};

[[nodiscard]] inline std::string to_string(PredictionOutcome po) {
    switch (po) {
        case PredictionOutcome::Confirmed:          return "CONFIRMED";
        case PredictionOutcome::PartiallyConfirmed: return "PARTIALLY_CONFIRMED";
        case PredictionOutcome::Refuted:            return "REFUTED";
        case PredictionOutcome::Inconclusive:       return "INCONCLUSIVE";
    }
    return "Unknown";
}

struct SolverRecommendation {
    SolverCandidate solver{SolverCandidate::DualSimplex};
    BackendCandidate backend{BackendCandidate::CPU};
    std::string rationale;
    double confidence{1.0};

    [[nodiscard]] std::string format_summary() const {
        return "[" + to_string(solver) + " on " + to_string(backend) + "] " + rationale;
    }
};

/// @brief Structure-aware algorithm and hardware backend recommendation engine.
/// Maps LP/MILP topological features (dimensions, sparsity, block-angularity, integrality)
/// directly to the most computationally suited solver.
class StructureAwareSelector {
public:
    [[nodiscard]] static SolverRecommendation recommend(
        const model::LinearProgram& lp,
        const ProblemStats& stats);

    [[nodiscard]] static PredictionOutcome evaluate_outcome(
        const SolverRecommendation& rec,
        SolverCandidate actual_solver,
        BackendCandidate actual_backend);
};

} // namespace pipepye::analysis
