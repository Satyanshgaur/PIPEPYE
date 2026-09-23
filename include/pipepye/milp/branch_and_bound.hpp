#pragma once

#include <pipepye/core/types.hpp>
#include <pipepye/model/lp_model.hpp>
#include <pipepye/milp/milp_types.hpp>
#include <pipepye/simplex/dual_simplex.hpp>
#include <optional>
#include <vector>

namespace pipepye::milp {

/// @brief Structure tracking up/down objective degradation history for pseudocost branching.
struct PseudoCostTable {
    std::vector<scalar_t> up_cost;
    std::vector<scalar_t> down_cost;
    std::vector<int> up_count;
    std::vector<int> down_count;

    explicit PseudoCostTable(size_t n = 0)
        : up_cost(n, 1.0), down_cost(n, 1.0),
          up_count(n, 0), down_count(n, 0) {}

    void update_down(index_t j, scalar_t delta_obj, scalar_t frac) {
        if (j < static_cast<index_t>(down_cost.size()) && frac > 1e-6) {
            scalar_t unit_change = delta_obj / frac;
            down_cost[j] = (down_cost[j] * down_count[j] + unit_change) / (down_count[j] + 1);
            down_count[j]++;
        }
    }

    void update_up(index_t j, scalar_t delta_obj, scalar_t frac) {
        if (j < static_cast<index_t>(up_cost.size()) && frac > 1e-6) {
            scalar_t unit_change = delta_obj / frac;
            up_cost[j] = (up_cost[j] * up_count[j] + unit_change) / (up_count[j] + 1);
            up_count[j]++;
        }
    }

    [[nodiscard]] scalar_t score(index_t j, scalar_t frac_down, scalar_t frac_up) const {
        if (j >= static_cast<index_t>(up_cost.size())) return 0.0;
        scalar_t s_down = down_cost[j] * frac_down;
        scalar_t s_up = up_cost[j] * frac_up;
        return 0.8 * std::min(s_down, s_up) + 0.2 * std::max(s_down, s_up);
    }
};

/// @brief Branch-and-Bound solver for Mixed-Integer Linear Programs (MILP).
/// Implements full MILP pipeline: root presolve, Gomory cuts, primal heuristics (rounding & diving),
/// pseudocost branching, node selection (DepthFirst, BestBound, BestEstimate),
/// and warm-started Dual Revised Simplex re-optimizations.
class BranchAndBoundSolver {
public:
    BranchAndBoundSolver() = default;
    explicit BranchAndBoundSolver(MILPConfig config);

    /// @brief Solves a Mixed-Integer Linear Program.
    [[nodiscard]] MILPResult solve(const model::LinearProgram& milp_model);

    /// @brief Solves the problem twice (with warm start vs cold start) to measure
    /// exact pivot reduction and time difference in tree exploration.
    [[nodiscard]] std::pair<MILPResult, MILPResult> solve_warm_vs_cold(const model::LinearProgram& milp_model);

    void set_config(const MILPConfig& config) noexcept { config_ = config; }
    [[nodiscard]] const MILPConfig& config() const noexcept { return config_; }

    /// @brief Standalone helper: Integer-aware root presolve.
    static Status apply_root_presolve(model::LinearProgram& lp);

    /// @brief Standalone helper: Generate Gomory Mixed-Integer fractional cuts.
    static int generate_gomory_cuts(model::LinearProgram& lp,
                                   const simplex::SimplexResult& root_res,
                                   const std::vector<index_t>& int_vars,
                                   int max_cuts);

    /// @brief Standalone helper: Simple rounding primal heuristic.
    static std::optional<std::vector<scalar_t>> run_simple_rounding(
        const model::LinearProgram& lp,
        const std::vector<scalar_t>& x_lp,
        const std::vector<index_t>& int_vars);

    /// @brief Standalone helper: Fractional diving primal heuristic.
    static std::optional<std::vector<scalar_t>> run_diving_heuristic(
        const model::LinearProgram& lp,
        const std::vector<scalar_t>& x_lp,
        const std::vector<index_t>& int_vars,
        simplex::DualSimplexSolver& simplex,
        const std::optional<simplex::Basis>& basis,
        int max_depth = 25);

private:
    MILPConfig config_;
};

} // namespace pipepye::milp
