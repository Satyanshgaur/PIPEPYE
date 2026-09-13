#pragma once

#include <pipepye/model/lp_model.hpp>
#include <pipepye/pipeline/prepared_lp.hpp>
#include <pipepye/pipeline/model_pipeline.hpp>
#include <pipepye/solver/solver_types.hpp>
#include <pipepye/solver/solution_verifier.hpp>
#include <functional>
#include <memory>

namespace pipepye::solver {

/// @brief Unified PDHG Linear Programming Solver routing to CPU or CUDA backends.
/// Consumes PreparedLP or raw LinearProgram, producing SolverResult and verifiable original solutions.
class PDHGSolver {
public:
    /// @brief Solves a prepared linear program directly on the selected backend (CPU or CUDA).
    /// @param prepared_lp The prepared, dimension-reduced and equilibrated model.
    /// @param config Solver runtime options (backend, tolerances, iterations, step-sizes, restart).
    [[nodiscard]] static SolverResult solve(
        const pipeline::PreparedLP& prepared_lp,
        const SolverConfig& config = SolverConfig::BaselineCPU());

    /// @brief Solves a standard LinearProgram directly without prior pipeline preparation.
    /// @param lp The linear program to solve.
    /// @param config Solver runtime options.
    [[nodiscard]] static SolverResult solve(
        const model::LinearProgram& lp,
        const SolverConfig& config = SolverConfig::BaselineCPU());

    /// @brief End-to-end solve orchestrating ModelPipeline preparation, PDHG solve,
    /// 1-step solution recovery back to the original model, and independent verification.
    /// @param orig_lp The raw original linear program.
    /// @param pipe_config Pipeline options (presolve passes, scaling algorithm, ablation mode).
    /// @param solver_config Solver options (backend, tolerances, iterations).
    /// @param verification_tol Absolute feasibility tolerance for independent verification.
    /// @return Pair of reconstructed original SolverResult and independent VerificationResult.
    [[nodiscard]] static std::pair<SolverResult, VerificationResult> solve_end_to_end(
        const model::LinearProgram& orig_lp,
        const pipeline::PipelineConfig& pipe_config = pipeline::PipelineConfig::PresolveAndScaling(),
        const SolverConfig& solver_config = SolverConfig::BaselineCPU(),
        scalar_t verification_tol = 1e-4);

    using CudaSolverFactory = std::function<SolverResult(const model::LinearProgram&, const SolverConfig&)>;
    static void register_cuda_solver_factory(CudaSolverFactory factory);
};

} // namespace pipepye::solver
