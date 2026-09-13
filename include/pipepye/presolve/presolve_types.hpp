#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <chrono>
#include <pipepye/core/types.hpp>
#include <pipepye/model/lp_model.hpp>

namespace pipepye::presolve {

/// @brief Outcome status of the presolve execution pipeline.
enum class PresolveStatus : uint8_t {
    Unchanged = 0,    ///< Presolve ran but no reductions were possible.
    Reduced,          ///< Presolve successfully reduced problem dimensions or tightened bounds.
    OptimalSolved,    ///< Presolve solved the problem entirely (all variables fixed/eliminated).
    Infeasible,       ///< Presolve proved the problem has no feasible solution.
    Unbounded,        ///< Presolve proved the problem is dual infeasible / primal unbounded.
    EmptyModel        ///< Input model contains 0 rows or 0 columns.
};

[[nodiscard]] inline std::string to_string(PresolveStatus status) {
    switch (status) {
        case PresolveStatus::Unchanged:     return "Unchanged";
        case PresolveStatus::Reduced:       return "Reduced";
        case PresolveStatus::OptimalSolved: return "OptimalSolved";
        case PresolveStatus::Infeasible:    return "Infeasible";
        case PresolveStatus::Unbounded:     return "Unbounded";
        case PresolveStatus::EmptyModel:    return "EmptyModel";
    }
    return "Unknown";
}

/// @brief Configuration settings for the presolve pipeline.
struct PresolveOptions {
    int max_passes{20};                   ///< Maximum number of presolve reduction iterations.
    scalar_t tolerance{1e-10};            ///< Numerical tolerance for bound and equality checks.
    scalar_t infinity_threshold{1e20};    ///< Values above this magnitude are treated as infinity.

    // Pass Enable / Disable toggles
    bool enable_empty_row_col{true};      ///< Detect and remove empty rows and columns.
    bool enable_fixed_variables{true};    ///< Detect and eliminate fixed variables.
    bool enable_singletons{true};         ///< Detect and reduce singleton rows and columns.
    bool enable_forcing_redundancy{true}; ///< Detect forcing and redundant constraints.
    bool enable_bound_tightening{true};   ///< Compute implied activity bounds and tighten variable bounds.

    bool verbose{false};                  ///< Emit logging telemetry during presolve passes.
};

/// @brief Detailed statistics captured by a single presolve pass.
struct PassStats {
    std::string pass_name;
    index_t rows_removed{0};
    index_t cols_removed{0};
    size_t nonzeros_removed{0};
    index_t bounds_tightened{0};
    index_t variables_fixed{0};
    index_t redundant_rows{0};
    double elapsed_ms{0.0};
};

/// @brief Aggregate telemetry metrics for the entire presolve execution.
struct PresolveStats {
    index_t initial_rows{0};
    index_t initial_cols{0};
    size_t initial_nnz{0};

    index_t final_rows{0};
    index_t final_cols{0};
    size_t final_nnz{0};

    int total_passes_executed{0};
    double total_elapsed_ms{0.0};

    std::vector<PassStats> pass_history;

    [[nodiscard]] double row_reduction_pct() const noexcept {
        if (initial_rows == 0) return 0.0;
        return (1.0 - static_cast<double>(final_rows) / initial_rows) * 100.0;
    }

    [[nodiscard]] double col_reduction_pct() const noexcept {
        if (initial_cols == 0) return 0.0;
        return (1.0 - static_cast<double>(final_cols) / initial_cols) * 100.0;
    }

    [[nodiscard]] double nnz_reduction_pct() const noexcept {
        if (initial_nnz == 0) return 0.0;
        return (1.0 - static_cast<double>(final_nnz) / initial_nnz) * 100.0;
    }
};

/// @brief Primal-dual solution container for solver evaluation and postsolve mapping.
struct PrimalDualSolution {
    std::vector<scalar_t> x;             ///< Primal variable values (length: num_cols)
    std::vector<scalar_t> y;             ///< Dual constraint multipliers (length: num_rows)
    std::vector<scalar_t> s;             ///< Dual reduced costs (length: num_cols)
    scalar_t objective_value{0.0};       ///< Evaluated objective value (including offsets)
    bool is_feasible{true};              ///< Whether primal-dual solution satisfies all bounds
};

} // namespace pipepye::presolve
