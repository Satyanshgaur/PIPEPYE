#pragma once

#include <vector>
#include <string>
#include <pipepye/core/types.hpp>
#include <pipepye/core/status.hpp>
#include <pipepye/model/lp_model.hpp>
#include <pipepye/presolve/presolve_pass_manager.hpp>

namespace pipepye::presolve {

/// @brief Detailed results from presolve correctness oracle verification.
struct OracleVerificationResult {
    bool passed{true};
    int samples_checked{0};
    int feasible_samples_found{0};
    int bound_violations{0};
    int constraint_violations{0};
    int objective_mismatches{0};
    scalar_t max_objective_error{0.0};
    scalar_t max_constraint_violation{0.0};
    scalar_t max_bound_violation{0.0};
    std::vector<std::string> error_messages;

    [[nodiscard]] std::string format_report() const;
};

/// @brief Options controlling presolve oracle verification behavior.
struct PresolveOracleOptions {
    int grid_points_per_dim{5};      ///< Grid resolution per variable for small models.
    int random_samples{100};          ///< Number of random sampling trials.
    scalar_t tolerance{1e-6};         ///< Numerical tolerance for feasibility and objective checks.
    uint32_t seed{42};                ///< Deterministic seed for reproducible testing.
};

/// @brief Deliberately simple correctness oracle for verifying presolve and postsolve transformations.
class PresolveOracle {
public:
    using Options = PresolveOracleOptions;

    PresolveOracle() : options_() {}
    explicit PresolveOracle(Options options) : options_(options) {}

    /// @brief Verifies that feasible points in the presolved space map to feasible points in the original space
    /// with strictly identical objective values.
    [[nodiscard]] OracleVerificationResult verify_feasibility_and_objective(
        const model::LinearProgram& original_lp,
        const PresolvedModel& presolved);

    /// @brief For small models (<= 5 variables), runs exhaustive grid search on both original and presolved
    /// models and verifies that the minimum feasible objective matches.
    [[nodiscard]] OracleVerificationResult verify_small_lp_optimality(
        const model::LinearProgram& original_lp,
        const PresolvedModel& presolved);

    /// @brief Checks that infeasibility or unboundedness reported by presolve is consistent with the model.
    [[nodiscard]] OracleVerificationResult verify_infeasibility_consistency(
        const model::LinearProgram& original_lp,
        const PresolvedModel& presolved);

private:
    Options options_;
};

} // namespace pipepye::presolve
