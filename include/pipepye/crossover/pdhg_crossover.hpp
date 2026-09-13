#pragma once

#include <pipepye/core/types.hpp>
#include <pipepye/core/status.hpp>
#include <pipepye/model/lp_model.hpp>
#include <pipepye/pipeline/prepared_lp.hpp>
#include <pipepye/simplex/simplex_types.hpp>
#include <pipepye/simplex/dual_simplex.hpp>
#include <pipepye/solver/solver_types.hpp>
#include <pipepye/solver/solution_verifier.hpp>
#include <vector>
#include <string>
#include <sstream>
#include <iomanip>

namespace pipepye::crossover {

/// @brief Configuration for PDHG to Simplex basis crossover.
struct CrossoverConfig {
    scalar_t active_tolerance{1e-4};            ///< Bound proximity tolerance to classify nonbasic variables
    simplex::SimplexConfig simplex_config{simplex::SimplexConfig::Fast()};
    int max_crossover_pivots{10000};
    bool verbose{false};
};

/// @brief Detailed results from the PDHG -> Simplex Crossover pipeline.
struct CrossoverResult {
    solver::TerminationStatus status{solver::TerminationStatus::NUMERICAL_FAILURE};

    simplex::SimplexResult final_result;
    simplex::Basis initial_crashed_basis;
    simplex::Basis final_basis;

    int cleanup_pivots{0};
    int active_bounds_detected{0};
    int structural_basic_vars{0};
    int slack_basic_vars{0};

    double crash_time_ms{0.0};
    double simplex_cleanup_time_ms{0.0};
    double total_time_ms{0.0};

    scalar_t pdhg_objective{0.0};
    scalar_t final_simplex_objective{0.0};
    scalar_t objective_difference{0.0};

    [[nodiscard]] bool is_optimal() const noexcept {
        return status == solver::TerminationStatus::OPTIMAL;
    }

    [[nodiscard]] std::string format_summary() const {
        std::ostringstream ss;
        ss << "=== PDHG -> Simplex Crossover Result ===\n"
           << "  Status:                 " << to_string(status) << "\n"
           << "  PDHG Objective:         " << std::scientific << std::setprecision(8) << pdhg_objective << "\n"
           << "  Final Simplex Obj:      " << std::scientific << std::setprecision(8) << final_simplex_objective << "\n"
           << "  Objective Shift:        " << std::scientific << std::setprecision(2) << objective_difference << "\n"
           << "  Active Bounds Found:    " << active_bounds_detected << "\n"
           << "  Crashed Basis (Str/Slk):" << structural_basic_vars << " / " << slack_basic_vars << "\n"
           << "  Simplex Cleanup Pivots: " << cleanup_pivots << "\n"
           << "  Total Crossover Time:   " << std::fixed << std::setprecision(2) << total_time_ms << " ms\n"
           << "    - Crash & Repair:     " << crash_time_ms << " ms\n"
           << "    - Simplex Cleanup:    " << simplex_cleanup_time_ms << " ms\n"
           << "========================================";
        return ss.str();
    }
};

/// @brief Crossover engine bridging first-order PDHG solutions to vertex-accurate Simplex bases.
class PDHGCrossover {
public:
    /// @brief Executes basis crash and warm-started simplex clean-up on a LinearProgram.
    /// @param lp Target linear program
    /// @param pdhg_x Approximate primal structural solution from PDHG
    /// @param pdhg_y Approximate dual multipliers from PDHG
    /// @param config Crossover configuration parameters
    [[nodiscard]] static CrossoverResult run(
        const model::LinearProgram& lp,
        const std::vector<scalar_t>& pdhg_x,
        const std::vector<scalar_t>& pdhg_y,
        const CrossoverConfig& config = CrossoverConfig());

    /// @brief Overload consuming PreparedLP.
    [[nodiscard]] static CrossoverResult run(
        const pipeline::PreparedLP& prepared_lp,
        const std::vector<scalar_t>& pdhg_x,
        const std::vector<scalar_t>& pdhg_y,
        const CrossoverConfig& config = CrossoverConfig());
};

} // namespace pipepye::crossover
