#pragma once

#include <pipepye/core/types.hpp>
#include <pipepye/model/lp_model.hpp>
#include <pipepye/milp/milp_types.hpp>
#include <pipepye/simplex/dual_simplex.hpp>
#include <optional>

namespace pipepye::milp {

/// @brief Branch-and-Bound solver for Mixed-Integer Linear Programs (MILP).
/// Solves root relaxation and explores the branch-and-bound tree using warm-started
/// Dual Revised Simplex basis updates.
class BranchAndBoundSolver {
public:
    BranchAndBoundSolver() = default;
    explicit BranchAndBoundSolver(MILPConfig config);

    /// @brief Solves a Mixed-Integer Linear Program.
    /// If model contains no integer variables, solves continuous LP directly.
    [[nodiscard]] MILPResult solve(const model::LinearProgram& milp_model);

    /// @brief Solves the problem twice (with warm start vs cold start) to measure
    /// exact pivot reduction and time difference in tree exploration.
    [[nodiscard]] std::pair<MILPResult, MILPResult> solve_warm_vs_cold(const model::LinearProgram& milp_model);

    void set_config(const MILPConfig& config) noexcept { config_ = config; }
    [[nodiscard]] const MILPConfig& config() const noexcept { return config_; }

private:
    MILPConfig config_;
};

} // namespace pipepye::milp
