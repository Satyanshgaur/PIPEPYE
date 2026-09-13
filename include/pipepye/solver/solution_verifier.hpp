#pragma once

#include <pipepye/model/lp_model.hpp>
#include <pipepye/solver/solver_types.hpp>
#include <pipepye/core/types.hpp>
#include <string>
#include <sstream>
#include <iomanip>

namespace pipepye::solver {

/// @brief Detailed result of independent solution verification on the original model.
struct VerificationResult {
    bool is_feasible{false};
    bool is_objective_consistent{false};
    bool is_dual_consistent{false};

    scalar_t max_bound_violation{0.0};
    scalar_t max_constraint_violation{0.0};
    scalar_t recomputed_objective{0.0};
    scalar_t objective_mismatch{0.0};
    scalar_t max_dual_stationarity_violation{0.0};

    index_t worst_bound_var{-1};
    index_t worst_constraint_row{-1};

    std::string details;

    [[nodiscard]] bool is_valid() const noexcept {
        return is_feasible && is_objective_consistent;
    }

    [[nodiscard]] std::string format_report() const {
        std::ostringstream ss;
        ss << "--- Independent Solution Verification Report ---\n"
           << "  Overall Result:        " << (is_valid() ? "PASSED (FEASIBLE)" : "FAILED (INFEASIBLE)") << "\n"
           << "  Max Bound Violation:   " << std::scientific << std::setprecision(2) << max_bound_violation;
        if (worst_bound_var >= 0) ss << " (var " << worst_bound_var << ")";
        ss << "\n  Max Row Violation:     " << std::scientific << std::setprecision(2) << max_constraint_violation;
        if (worst_constraint_row >= 0) ss << " (row " << worst_constraint_row << ")";
        ss << "\n  Recomputed Objective:  " << std::scientific << std::setprecision(6) << recomputed_objective
           << "\n  Objective Mismatch:    " << std::scientific << std::setprecision(2) << objective_mismatch
           << "\n  Dual Stationarity Err: " << std::scientific << std::setprecision(2) << max_dual_stationarity_violation
           << "\n------------------------------------------------";
        return ss.str();
    }
};

/// @brief Independent verification layer auditing solver outputs directly against model constraints.
class SolutionVerifier {
public:
    /// @brief Verifies a primal-dual solution against a LinearProgram (either raw or prepared).
    /// @param lp The linear program to verify against (must match dimension of x and y).
    /// @param x Primal variable values.
    /// @param y Dual multiplier values (can be empty if only primal verification is desired).
    /// @param claimed_objective Reported objective value by the solver.
    /// @param tolerance Absolute feasibility tolerance (default 1e-4).
    [[nodiscard]] static VerificationResult verify(
        const model::LinearProgram& lp,
        const std::vector<scalar_t>& x,
        const std::vector<scalar_t>& y = {},
        scalar_t claimed_objective = 0.0,
        scalar_t tolerance = 1e-4);

    /// @brief Overload taking SolverResult directly.
    [[nodiscard]] static VerificationResult verify(
        const model::LinearProgram& lp,
        const SolverResult& result,
        scalar_t tolerance = 1e-4) {
        return verify(lp, result.x, result.y, result.primal_objective, tolerance);
    }
};

} // namespace pipepye::solver
