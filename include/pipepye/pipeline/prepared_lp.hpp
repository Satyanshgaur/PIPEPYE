#pragma once

#include <memory>
#include <pipepye/core/types.hpp>
#include <pipepye/core/status.hpp>
#include <pipepye/model/lp_model.hpp>
#include <pipepye/presolve/presolve_types.hpp>
#include <pipepye/presolve/postsolve.hpp>
#include <pipepye/scaling/scaling_types.hpp>
#include <pipepye/analysis/problem_stats.hpp>
#include <pipepye/pipeline/pipeline_types.hpp>

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

    PipelineMode mode_applied{PipelineMode::PRESOLVE_AND_SCALING};
    double pipeline_elapsed_ms{0.0};                ///< Total execution time spent in prepare() in milliseconds

    bool is_solved_by_presolve{false};              ///< True if presolve eliminated all variables to optimality
    bool is_infeasible{false};                      ///< True if presolve proved the model has no feasible point
    bool is_unbounded{false};                       ///< True if presolve proved the problem is unbounded

    /// @brief Formats an executive multi-line summary of the preparation run and applied ablation mode.
    [[nodiscard]] std::string format_pipeline_summary() const {
        std::ostringstream ss;
        ss << "================================================================================\n";
        ss << "                   PIPEPYE MODEL PREPARATION SUMMARY                            \n";
        ss << "================================================================================\n";
        ss << "  Ablation Mode:        " << to_string(mode_applied) << "\n";
        ss << "  Total Prep Time:      " << std::fixed << std::setprecision(2) << pipeline_elapsed_ms << " ms\n";
        ss << "  Final Dimensions:     " << lp.num_rows() << " rows, " << lp.num_cols() << " cols, "
           << lp.num_nonzeros() << " nonzeros\n";
        ss << "  Presolve Status:      ";
        if (is_solved_by_presolve) ss << "OPTIMAL_SOLVED_BY_PRESOLVE\n";
        else if (is_infeasible)    ss << "INFEASIBLE\n";
        else if (is_unbounded)     ss << "UNBOUNDED\n";
        else if (recovery_map.was_presolved()) ss << "REDUCED\n";
        else ss << "SKIPPED_OR_UNCHANGED\n";

        if (recovery_map.was_presolved()) {
            ss << "  Presolve Reductions:  Rows: -" << presolve_stats.eliminated_rows()
               << " (" << std::setprecision(1) << presolve_stats.row_reduction_pct() << "%), "
               << "Cols: -" << presolve_stats.eliminated_cols()
               << " (" << std::setprecision(1) << presolve_stats.col_reduction_pct() << "%), "
               << "NNZ: -" << presolve_stats.eliminated_nonzeros()
               << " (" << std::setprecision(1) << presolve_stats.nnz_reduction_pct() << "%)\n";
            ss << "  Presolve Detail:      " << presolve_stats.total_variables_fixed() << " fixed vars, "
               << presolve_stats.total_bounds_tightened() << " tightened bounds, "
               << presolve_stats.total_passes_executed << " passes in "
               << std::setprecision(2) << presolve_stats.total_elapsed_ms << " ms\n";
        }

        if (recovery_map.was_scaled()) {
            ss << "  Matrix Scaling:       " << to_string(scaling.options_snapshot.method) << " ("
               << scaling.iterations_performed << " iterations in "
               << std::setprecision(2) << scaling.elapsed_ms << " ms)\n";
            ss << "  Dynamic Range:        " << std::scientific << std::setprecision(2)
               << scaling.diag_before.dynamic_range << " -> " << scaling.diag_after.dynamic_range << "\n";
        }

        ss << "  Recommended Engine:   " << to_string(problem_stats.recommended_engine) << "\n";
        ss << "  One-Line Banner:      " << problem_stats.format_one_line_summary() << "\n";
        ss << "================================================================================\n";
        return ss.str();
    }

    /// @brief Recovers the original solution in one call.
    [[nodiscard]] StatusOr<presolve::PrimalDualSolution> recover_solution(
        const presolve::PrimalDualSolution& solver_solution,
        const model::LinearProgram& original_lp) const {
        return recovery_map.recover(solver_solution, original_lp);
    }
};

} // namespace pipepye::pipeline
