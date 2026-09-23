#pragma once

#include <pipepye/core/types.hpp>
#include <pipepye/core/status.hpp>
#include <pipepye/solver/solver_types.hpp>
#include <pipepye/simplex/simplex_types.hpp>
#include <vector>
#include <string>
#include <sstream>
#include <iomanip>

namespace pipepye::milp {

/// @brief Node selection strategy in the Branch-and-Bound tree search.
enum class NodeSelectionStrategy : uint8_t {
    BestBound = 0,    ///< Select node with smallest dual bound (minimizes total nodes)
    DepthFirst,       ///< Select deepest node in tree (LIFO - minimal memory, fast incumbent)
    BestEstimate      ///< Select node with best estimated integer objective using pseudocosts
};

[[nodiscard]] inline std::string to_string(NodeSelectionStrategy strategy) {
    switch (strategy) {
        case NodeSelectionStrategy::BestBound:    return "BEST_BOUND";
        case NodeSelectionStrategy::DepthFirst:   return "DEPTH_FIRST";
        case NodeSelectionStrategy::BestEstimate: return "BEST_ESTIMATE";
    }
    return "UNKNOWN";
}

/// @brief Variable selection strategy when branching on fractional integer variables.
enum class BranchingStrategy : uint8_t {
    MostFractional = 0,  ///< Variable closest to 0.5 (maximum fractionality)
    FirstFractional,     ///< First variable with non-integer value
    PseudoCost           ///< Variable maximizing expected bound increase via history
};

[[nodiscard]] inline std::string to_string(BranchingStrategy strategy) {
    switch (strategy) {
        case BranchingStrategy::MostFractional:  return "MOST_FRACTIONAL";
        case BranchingStrategy::FirstFractional: return "FIRST_FRACTIONAL";
        case BranchingStrategy::PseudoCost:      return "PSEUDO_COST";
    }
    return "UNKNOWN";
}

/// @brief Primal heuristics applied during tree search.
enum class HeuristicStrategy : uint8_t {
    None = 0,
    SimpleRounding,   ///< Round fractional values directly and test constraint feasibility
    Diving,           ///< Iteratively fix most integral variable and resolve LP
    All               ///< Apply rounding and diving heuristics
};

[[nodiscard]] inline std::string to_string(HeuristicStrategy strategy) {
    switch (strategy) {
        case HeuristicStrategy::None:           return "NONE";
        case HeuristicStrategy::SimpleRounding: return "SIMPLE_ROUNDING";
        case HeuristicStrategy::Diving:         return "DIVING";
        case HeuristicStrategy::All:            return "ALL";
    }
    return "UNKNOWN";
}

/// @brief Cutting plane generation strategy at root node.
enum class CuttingPlaneStrategy : uint8_t {
    None = 0,
    GomoryFractional  ///< Gomory Mixed-Integer (GMI) fractional cuts from simplex tableau
};

[[nodiscard]] inline std::string to_string(CuttingPlaneStrategy strategy) {
    switch (strategy) {
        case CuttingPlaneStrategy::None:             return "NONE";
        case CuttingPlaneStrategy::GomoryFractional: return "GOMORY_FRACTIONAL";
    }
    return "UNKNOWN";
}

enum class MILPTerminationStatus : uint8_t {
    Optimal = 0,
    Feasible,
    Infeasible,
    TimeLimit,
    NodeLimit,
    NumericalFailure
};

[[nodiscard]] inline std::string to_string(MILPTerminationStatus status) {
    switch (status) {
        case MILPTerminationStatus::Optimal:          return "OPTIMAL";
        case MILPTerminationStatus::Feasible:         return "FEASIBLE";
        case MILPTerminationStatus::Infeasible:       return "INFEASIBLE";
        case MILPTerminationStatus::TimeLimit:        return "TIME_LIMIT";
        case MILPTerminationStatus::NodeLimit:        return "NODE_LIMIT";
        case MILPTerminationStatus::NumericalFailure: return "NUMERICAL_FAILURE";
    }
    return "UNKNOWN";
}

struct MILPConfig {
    int max_nodes{2000};
    double time_limit_sec{60.0};
    scalar_t mip_gap_tol{1e-4};
    scalar_t integrality_tol{1e-5};
    bool use_warm_start{true};

    NodeSelectionStrategy node_selection{NodeSelectionStrategy::BestBound};
    BranchingStrategy branching_strategy{BranchingStrategy::MostFractional};
    HeuristicStrategy heuristic_strategy{HeuristicStrategy::None};
    CuttingPlaneStrategy cutting_strategy{CuttingPlaneStrategy::None};
    int max_root_cuts{0};
    bool enable_root_presolve{false};

    simplex::SimplexConfig simplex_config{simplex::SimplexConfig::Fast()};

    static MILPConfig Default() {
        return MILPConfig{};
    }

    static MILPConfig Advanced() {
        MILPConfig c;
        c.heuristic_strategy = HeuristicStrategy::All;
        c.cutting_strategy = CuttingPlaneStrategy::GomoryFractional;
        c.max_root_cuts = 25;
        c.enable_root_presolve = true;
        c.branching_strategy = BranchingStrategy::PseudoCost;
        return c;
    }

    static MILPConfig WithHeuristics() {
        MILPConfig c;
        c.heuristic_strategy = HeuristicStrategy::All;
        return c;
    }

    static MILPConfig WithCuts() {
        MILPConfig c;
        c.cutting_strategy = CuttingPlaneStrategy::GomoryFractional;
        c.max_root_cuts = 25;
        return c;
    }

    static MILPConfig DepthFirst() {
        MILPConfig c;
        c.node_selection = NodeSelectionStrategy::DepthFirst;
        return c;
    }

    static MILPConfig PseudoCostBranching() {
        MILPConfig c;
        c.branching_strategy = BranchingStrategy::PseudoCost;
        return c;
    }

    static MILPConfig ColdStartBaseline() {
        MILPConfig c;
        c.use_warm_start = false;
        c.heuristic_strategy = HeuristicStrategy::None;
        c.cutting_strategy = CuttingPlaneStrategy::None;
        c.enable_root_presolve = false;
        return c;
    }
};

struct GapMilestone {
    double time_ms{0.0};
    int node{0};
    scalar_t dual_bound{0.0};
    scalar_t incumbent_obj{0.0};
    scalar_t gap{1.0};
};

struct MILPResult {
    MILPTerminationStatus status{MILPTerminationStatus::NumericalFailure};

    scalar_t best_objective{model::Infinity};
    scalar_t best_bound{-model::Infinity};
    scalar_t mip_gap{1.0};
    std::vector<scalar_t> x;

    int nodes_created{0};
    int nodes_explored{0};
    int nodes_pruned_bound{0};
    int nodes_pruned_infeasible{0};
    int nodes_pruned_integral{0};
    int active_tree_size{0};
    int peak_tree_size{0};

    int total_lp_solves{0};
    int total_pivots{0};
    int warm_start_pivots{0};
    int cold_start_pivots{0};

    scalar_t root_objective{model::Infinity};
    scalar_t root_bound_after_cuts{model::Infinity};
    int root_cuts_added{0};
    int heuristic_solutions_found{0};

    double root_relaxation_time_ms{0.0};
    double cut_generation_time_ms{0.0};
    double heuristic_time_ms{0.0};
    double lp_relaxation_time_ms{0.0};
    double time_to_first_incumbent_ms{0.0};
    double time_to_gap_10pct_ms{0.0};
    double time_to_gap_1pct_ms{0.0};
    double time_to_gap_01pct_ms{0.0};
    double total_time_ms{0.0};
    double nodes_per_second{0.0};

    std::vector<GapMilestone> gap_history;

    [[nodiscard]] bool is_optimal() const noexcept {
        return status == MILPTerminationStatus::Optimal;
    }

    [[nodiscard]] bool is_feasible() const noexcept {
        return best_objective < model::Infinity / 2.0;
    }

    [[nodiscard]] double pivot_reduction_ratio() const noexcept {
        if (cold_start_pivots == 0) return 0.0;
        return 1.0 - (static_cast<double>(warm_start_pivots) / static_cast<double>(cold_start_pivots));
    }

    [[nodiscard]] std::string format_summary() const {
        std::ostringstream ss;
        ss << "--- Branch-and-Bound MILP Result ---\n"
           << "  Status:                 " << to_string(status) << "\n"
           << "  Best Incumbent:         " << std::scientific << std::setprecision(8) << best_objective << "\n"
           << "  Best Dual Bound:        " << std::scientific << std::setprecision(8) << best_bound << "\n"
           << "  MIP Relative Gap:       " << std::fixed << std::setprecision(4) << (mip_gap * 100.0) << " %\n"
           << "  Nodes Created:          " << nodes_created << "\n"
           << "  Nodes Explored:         " << nodes_explored << " ("
           << std::fixed << std::setprecision(1) << nodes_per_second << " nodes/sec)\n"
           << "  Nodes Pruned:           Bound=" << nodes_pruned_bound 
           << ", Infeas=" << nodes_pruned_infeasible 
           << ", Integer=" << nodes_pruned_integral << "\n"
           << "  Active / Peak Tree:     " << active_tree_size << " / " << peak_tree_size << "\n"
           << "  Total LP Solves:        " << total_lp_solves << "\n"
           << "  Total Pivots:           " << total_pivots << " (Warm: " << warm_start_pivots << ", Cold: " << cold_start_pivots << ")\n"
           << "  Root Objective:         " << std::scientific << std::setprecision(8) << root_objective;
        if (root_cuts_added > 0) {
            ss << " (Tightened after cuts: " << root_bound_after_cuts << ", " << root_cuts_added << " cuts)\n";
        } else {
            ss << "\n";
        }
        ss << "  Heuristic Incumbents:   " << heuristic_solutions_found << "\n"
           << "  Time to 1st Incumbent:  " << std::fixed << std::setprecision(2) << time_to_first_incumbent_ms << " ms\n"
           << "  Time to Gap <= 10%:     " << (time_to_gap_10pct_ms > 0 ? std::to_string(time_to_gap_10pct_ms) + " ms" : "N/A") << "\n"
           << "  Time to Gap <= 1%:      " << (time_to_gap_1pct_ms > 0 ? std::to_string(time_to_gap_1pct_ms) + " ms" : "N/A") << "\n"
           << "  Time to Gap <= 0.1%:    " << (time_to_gap_01pct_ms > 0 ? std::to_string(time_to_gap_01pct_ms) + " ms" : "N/A") << "\n"
           << "  Total Solve Time:       " << std::fixed << std::setprecision(2) << total_time_ms << " ms\n"
           << "------------------------------------";
        return ss.str();
    }
};

} // namespace pipepye::milp
