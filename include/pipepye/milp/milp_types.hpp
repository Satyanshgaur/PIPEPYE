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

enum class BranchingStrategy : uint8_t {
    MostFractional = 0,
    FirstFractional
};

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
    BranchingStrategy branching_strategy{BranchingStrategy::MostFractional};
    simplex::SimplexConfig simplex_config{simplex::SimplexConfig::Fast()};

    static MILPConfig Default() {
        return MILPConfig{};
    }

    static MILPConfig ColdStartBaseline() {
        MILPConfig c;
        c.use_warm_start = false;
        return c;
    }
};

struct MILPResult {
    MILPTerminationStatus status{MILPTerminationStatus::NumericalFailure};

    scalar_t best_objective{model::Infinity};
    scalar_t best_bound{-model::Infinity};
    scalar_t mip_gap{1.0};
    std::vector<scalar_t> x;

    int nodes_explored{0};
    int total_lp_solves{0};
    int total_pivots{0};
    int warm_start_pivots{0};
    int cold_start_pivots{0};

    scalar_t root_objective{model::Infinity};
    double root_relaxation_time_ms{0.0};
    double time_to_first_incumbent_ms{0.0};
    double total_time_ms{0.0};

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
           << "  Nodes Explored:         " << nodes_explored << "\n"
           << "  Total LP Solves:        " << total_lp_solves << "\n"
           << "  Total Pivots:           " << total_pivots << "\n"
           << "  Root Objective:         " << std::scientific << std::setprecision(8) << root_objective << "\n"
           << "  Root Solve Time:        " << std::fixed << std::setprecision(2) << root_relaxation_time_ms << " ms\n"
           << "  Time to 1st Incumbent:  " << std::fixed << std::setprecision(2) << time_to_first_incumbent_ms << " ms\n"
           << "  Total Solve Time:       " << std::fixed << std::setprecision(2) << total_time_ms << " ms\n"
           << "------------------------------------";
        return ss.str();
    }
};

} // namespace pipepye::milp
