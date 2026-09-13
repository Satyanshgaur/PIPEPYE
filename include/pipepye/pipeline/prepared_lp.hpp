#pragma once

#include <memory>
#include <pipepye/core/types.hpp>
#include <pipepye/core/status.hpp>
#include <pipepye/model/lp_model.hpp>
#include <pipepye/presolve/presolve_types.hpp>
#include <pipepye/presolve/postsolve.hpp>
#include <pipepye/scaling/scaling_types.hpp>
#include <pipepye/analysis/problem_stats.hpp>

namespace pipepye::pipeline {

/// @brief Unified solution recovery mapping unscaling and postsolving back to original formulation.
class SolutionRecoveryMap {
public:
    SolutionRecoveryMap() = default;

    SolutionRecoveryMap(std::shared_ptr<presolve::PostsolveManager> postsolve_mgr,
                        scaling::ScaledModel scaling,
                        bool was_presolved,
                        bool was_scaled)
        : postsolve_mgr_(std::move(postsolve_mgr)),
          scaling_(std::move(scaling)),
          was_presolved_(was_presolved),
          was_scaled_(was_scaled) {}

    /// @brief Single-call recovery: takes solver output on transformed space, unscales, and reconstructs
    /// all eliminated/fixed variables and constraints back into the original LP space.
    [[nodiscard]] StatusOr<presolve::PrimalDualSolution> recover(
        const presolve::PrimalDualSolution& solver_solution,
        const model::LinearProgram& original_lp) const {

        presolve::PrimalDualSolution unscaled_sol = solver_solution;

        // 1. Unscale solver coordinates if scaling was applied
        if (was_scaled_) {
            unscaled_sol = scaling_.unscale_solution(solver_solution);
        }

        // 2. Unwind postsolve transformations if presolve was applied
        if (was_presolved_ && postsolve_mgr_) {
            return postsolve_mgr_->postsolve(original_lp, unscaled_sol);
        }

        // Neither presolve nor postsolve modified structure: compute final objective
        scalar_t obj = original_lp.obj_offset;
        for (index_t j = 0; j < original_lp.num_cols(); ++j) {
            if (j < static_cast<index_t>(unscaled_sol.x.size())) {
                obj += original_lp.c[j] * unscaled_sol.x[j];
            }
        }
        unscaled_sol.objective_value = obj;
        return unscaled_sol;
    }

    [[nodiscard]] bool was_presolved() const noexcept { return was_presolved_; }
    [[nodiscard]] bool was_scaled() const noexcept { return was_scaled_; }
    [[nodiscard]] const scaling::ScaledModel& scaling() const noexcept { return scaling_; }
    [[nodiscard]] const std::shared_ptr<presolve::PostsolveManager>& postsolve_mgr() const noexcept {
        return postsolve_mgr_;
    }

private:
    std::shared_ptr<presolve::PostsolveManager> postsolve_mgr_;
    scaling::ScaledModel scaling_;
    bool was_presolved_{false};
    bool was_scaled_{false};
};

/// @brief Unified transformed LP model container consumed directly by optimization solvers (PDHG / Simplex).
/// Downstream solvers consume this structure without needing to understand presolve or scaling internals.
struct PreparedLP {
    model::LinearProgram lp;                        ///< Final preconditioned, dimension-reduced canonical LP
    presolve::PresolveStats presolve_stats;         ///< Dimension reduction and pass execution telemetry
    scaling::ScaledModel scaling;                   ///< Diagonal scaling factors and pre/post diagnostics
    analysis::ProblemStats problem_stats;           ///< Sparsity geometry and recommended GPU/CPU engine
    SolutionRecoveryMap recovery_map;               ///< Reversible 1-step solution recovery pipeline

    bool is_solved_by_presolve{false};              ///< True if presolve eliminated all variables to optimality
    bool is_infeasible{false};                      ///< True if presolve proved the model has no feasible point
    bool is_unbounded{false};                       ///< True if presolve proved the problem is unbounded

    /// @brief Recovers the original solution in one call.
    [[nodiscard]] StatusOr<presolve::PrimalDualSolution> recover_solution(
        const presolve::PrimalDualSolution& solver_solution,
        const model::LinearProgram& original_lp) const {
        return recovery_map.recover(solver_solution, original_lp);
    }
};

} // namespace pipepye::pipeline
