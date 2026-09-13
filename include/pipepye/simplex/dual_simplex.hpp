#pragma once

#include <pipepye/core/types.hpp>
#include <pipepye/core/status.hpp>
#include <pipepye/model/lp_model.hpp>
#include <pipepye/pipeline/prepared_lp.hpp>
#include <pipepye/pipeline/model_pipeline.hpp>
#include <pipepye/solver/solution_verifier.hpp>
#include <pipepye/simplex/simplex_types.hpp>
#include <pipepye/factorization/basis_factorization.hpp>
#include <vector>
#include <memory>
#include <optional>

namespace pipepye::simplex {

/// @brief Sparse Dual Revised Simplex Solver.
/// Solves general bounded linear programs using sparse LU factorization, Product Form of Inverse (PFI),
/// Devex steepest-edge pricing, and bound-flipping ratio tests.
class DualSimplexSolver {
public:
    DualSimplexSolver() = default;
    explicit DualSimplexSolver(SimplexConfig config);

    /// @brief Solves a prepared linear program directly using Dual Revised Simplex.
    /// @param prepared_lp The prepared, dimension-reduced and equilibrated model.
    /// @param initial_basis Optional warm-start basis (e.g. from crossover).
    [[nodiscard]] SimplexResult solve(
        const pipeline::PreparedLP& prepared_lp,
        const std::optional<Basis>& initial_basis = std::nullopt);

    /// @brief Solves a standard LinearProgram directly without prior pipeline preparation.
    /// @param lp The linear program to solve.
    /// @param initial_basis Optional warm-start basis.
    [[nodiscard]] SimplexResult solve(
        const model::LinearProgram& lp,
        const std::optional<Basis>& initial_basis = std::nullopt);

    /// @brief End-to-end solve orchestrating ModelPipeline preparation, Dual Simplex solve,
    /// 1-step solution recovery back to the original model, and independent verification.
    /// @param orig_lp The raw original linear program.
    /// @param pipe_config Pipeline options (presolve, scaling, ablation).
    /// @param simplex_config Simplex solver options (pricing, ratio test, update method).
    /// @param verification_tol Absolute feasibility tolerance for independent verification.
    /// @return Pair of reconstructed original SimplexResult and independent VerificationResult.
    [[nodiscard]] static std::pair<SimplexResult, solver::VerificationResult> solve_end_to_end(
        const model::LinearProgram& orig_lp,
        const pipeline::PipelineConfig& pipe_config = pipeline::PipelineConfig::PresolveAndScaling(),
        const SimplexConfig& simplex_config = SimplexConfig::Fast(),
        scalar_t verification_tol = 1e-4);

    /// @brief Sets solver configuration.
    void set_config(const SimplexConfig& config) noexcept { config_ = config; }
    [[nodiscard]] const SimplexConfig& config() const noexcept { return config_; }

private:
    SimplexConfig config_;
};

} // namespace pipepye::simplex
