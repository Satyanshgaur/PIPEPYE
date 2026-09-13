#pragma once

#include <pipepye/core/types.hpp>
#include <pipepye/model/lp_model.hpp>
#include <pipepye/solver/solver_types.hpp>
#include <vector>
#include <string>
#include <sstream>
#include <iomanip>

namespace pipepye::simplex {

/// @brief Variable status in revised simplex basis partition.
enum class VariableStatus : uint8_t {
    Basic = 0,     ///< Variable is currently in the basis B
    AtLower,       ///< Nonbasic variable fixed at its finite lower bound
    AtUpper,       ///< Nonbasic variable fixed at its finite upper bound
    Fixed,         ///< Nonbasic variable with identical lower and upper bounds (l == u)
    Free           ///< Nonbasic free variable (l = -inf, u = +inf, value = 0.0)
};

[[nodiscard]] inline std::string to_string(VariableStatus status) {
    switch (status) {
        case VariableStatus::Basic:   return "BASIC";
        case VariableStatus::AtLower: return "AT_LOWER";
        case VariableStatus::AtUpper: return "AT_UPPER";
        case VariableStatus::Fixed:   return "FIXED";
        case VariableStatus::Free:    return "FREE";
    }
    return "UNKNOWN";
}

/// @brief Dual simplex pricing rules.
enum class PricingStrategy : uint8_t {
    Dantzig = 0,         ///< Standard dual textbook pricing (most infeasible basic variable)
    Devex,               ///< Devex approximation of dual steepest edge
    DualSteepestEdge     ///< Exact dual steepest-edge pricing with updated edge weights
};

[[nodiscard]] inline std::string to_string(PricingStrategy strategy) {
    switch (strategy) {
        case PricingStrategy::Dantzig:          return "DANTZIG";
        case PricingStrategy::Devex:            return "DEVEX";
        case PricingStrategy::DualSteepestEdge: return "DUAL_STEEPEST_EDGE";
    }
    return "UNKNOWN";
}

/// @brief Dual ratio test strategy for entering variable selection.
enum class RatioTestStrategy : uint8_t {
    Standard = 0,        ///< Standard Harris dual ratio test (single pivot per iteration)
    BoundFlipping        ///< Multi-step bound flipping ratio test across intervening nonbasic bounds
};

[[nodiscard]] inline std::string to_string(RatioTestStrategy strategy) {
    switch (strategy) {
        case RatioTestStrategy::Standard:      return "STANDARD";
        case RatioTestStrategy::BoundFlipping: return "BOUND_FLIPPING";
    }
    return "UNKNOWN";
}

/// @brief Basis update representation.
enum class BasisUpdateMethod : uint8_t {
    RefactorizeAlways = 0,   ///< Recompute fresh sparse LU on every pivot (testing / oracle)
    ProductFormInverse,      ///< Product Form of Inverse (PFI) with eta vectors
    ForrestTomlin            ///< Forrest-Tomlin row-and-column update
};

[[nodiscard]] inline std::string to_string(BasisUpdateMethod method) {
    switch (method) {
        case BasisUpdateMethod::RefactorizeAlways: return "REFACTORIZE_ALWAYS";
        case BasisUpdateMethod::ProductFormInverse: return "PRODUCT_FORM_INVERSE";
        case BasisUpdateMethod::ForrestTomlin:      return "FORREST_TOMLIN";
    }
    return "UNKNOWN";
}

/// @brief Explicit Simplex Basis State partition.
/// In standard bounded LP: l_r <= A x <= u_r, we introduce slack variables s: A x - s = 0 with l_r <= s <= u_r.
/// Total variables = n (structural) + m (slack) = n + m.
/// Dimension of basis B is m x m.
struct Basis {
    index_t num_rows{0};                            ///< m: row dimension of basis
    index_t num_total_vars{0};                      ///< n + m: structurals + slacks

    std::vector<index_t> basic_vars;                ///< basic_vars[i] is variable index of i-th basic column (size m)
    std::vector<index_t> nonbasic_vars;             ///< List of nonbasic variable indices
    std::vector<VariableStatus> var_status;         ///< Status for all n + m variables (size n + m)
    std::vector<index_t> var_to_basic_idx;          ///< Maps variable index to row position in basis (-1 if nonbasic)

    Basis() = default;

    Basis(index_t m, index_t total_vars)
        : num_rows(m), num_total_vars(total_vars),
          basic_vars(m, -1),
          var_status(total_vars, VariableStatus::AtLower),
          var_to_basic_idx(total_vars, -1) {}

    /// @brief Checks structural basis validity invariants:
    /// 1. Number of basic variables == num_rows
    /// 2. Every basic variable has status BASIC and non-negative row index
    /// 3. Every nonbasic variable has status != BASIC
    [[nodiscard]] bool is_valid() const noexcept {
        if (static_cast<index_t>(basic_vars.size()) != num_rows) return false;
        if (static_cast<index_t>(var_status.size()) != num_total_vars) return false;

        index_t basic_count = 0;
        for (index_t i = 0; i < num_rows; ++i) {
            index_t v = basic_vars[i];
            if (v < 0 || v >= num_total_vars) return false;
            if (var_status[v] != VariableStatus::Basic) return false;
            if (var_to_basic_idx[v] != i) return false;
            basic_count++;
        }
        return basic_count == num_rows;
    }

    /// @brief Swaps leaving basic variable with entering nonbasic variable.
    /// @param row_idx Position in basis (0 <= row_idx < m)
    /// @param entering_var Variable entering the basis
    /// @param leaving_new_status New status of the leaving variable (AtLower, AtUpper, Fixed, or Free)
    void swap(index_t row_idx, index_t entering_var, VariableStatus leaving_new_status) {
        index_t leaving_var = basic_vars[row_idx];
        var_status[leaving_var] = leaving_new_status;
        var_to_basic_idx[leaving_var] = -1;

        basic_vars[row_idx] = entering_var;
        var_status[entering_var] = VariableStatus::Basic;
        var_to_basic_idx[entering_var] = row_idx;

        // Reconstruct nonbasic list
        nonbasic_vars.clear();
        for (index_t v = 0; v < num_total_vars; ++v) {
            if (var_status[v] != VariableStatus::Basic) {
                nonbasic_vars.push_back(v);
            }
        }
    }
};

/// @brief Runtime configuration for Dual Revised Simplex solver.
struct SimplexConfig {
    PricingStrategy pricing{PricingStrategy::Devex};
    RatioTestStrategy ratio_test{RatioTestStrategy::BoundFlipping};
    BasisUpdateMethod update_method{BasisUpdateMethod::ProductFormInverse};

    scalar_t primal_feasibility_tol{1e-7};      ///< Primal feasibility tolerance
    scalar_t dual_feasibility_tol{1e-7};        ///< Dual reduced cost tolerance
    scalar_t pivot_threshold{1e-8};             ///< Minimum absolute pivot magnitude allowed
    scalar_t markowitz_threshold{0.1};          ///< Pivot threshold for sparse LU factorization (0.01 - 0.5)

    int max_iterations{50000};                  ///< Maximum simplex pivots
    int max_updates_before_refactorize{60};     ///< Maximum eta updates before triggering full refactorization
    double time_limit_sec{300.0};               ///< Wall-clock limit in seconds

    bool check_numerical_stability{true};       ///< Check Bx = b and B^T y = c residuals
    bool verbose{false};

    static SimplexConfig Fast() {
        SimplexConfig c;
        c.pricing = PricingStrategy::Devex;
        c.ratio_test = RatioTestStrategy::BoundFlipping;
        c.update_method = BasisUpdateMethod::ProductFormInverse;
        c.max_updates_before_refactorize = 60;
        return c;
    }

    static SimplexConfig Robust() {
        SimplexConfig c;
        c.pricing = PricingStrategy::Devex;
        c.ratio_test = RatioTestStrategy::Standard;
        c.update_method = BasisUpdateMethod::ProductFormInverse;
        c.max_updates_before_refactorize = 30;
        c.pivot_threshold = 1e-7;
        c.markowitz_threshold = 0.2;
        return c;
    }

    static SimplexConfig ExactDense() {
        SimplexConfig c;
        c.pricing = PricingStrategy::Dantzig;
        c.ratio_test = RatioTestStrategy::Standard;
        c.update_method = BasisUpdateMethod::RefactorizeAlways;
        c.max_updates_before_refactorize = 0;
        return c;
    }
};

/// @brief Simplex solve execution telemetry.
struct SimplexTiming {
    double total_time_ms{0.0};
    double initial_basis_ms{0.0};
    double factorization_ms{0.0};
    double update_ms{0.0};
    double ftran_ms{0.0};
    double btran_ms{0.0};
    double pricing_ms{0.0};
    double ratio_test_ms{0.0};
};

/// @brief Complete result of Dual Revised Simplex solve.
struct SimplexResult {
    solver::TerminationStatus status{solver::TerminationStatus::NUMERICAL_FAILURE};

    std::vector<scalar_t> x;                    ///< Primal structural solution (size n)
    std::vector<scalar_t> y;                    ///< Dual multipliers on constraints (size m)
    std::vector<scalar_t> s;                    ///< Dual reduced costs (size n)

    scalar_t objective_value{0.0};
    scalar_t max_primal_infeasibility{0.0};
    scalar_t max_dual_infeasibility{0.0};

    int iterations{0};                          ///< Total pivots performed
    int factorizations{0};                      ///< Count of full basis refactorizations
    int updates{0};                             ///< Count of basis updates (PFI / eta)
    int bound_flips{0};                         ///< Count of nonbasic bound flips during ratio tests
    int ftran_count{0};                         ///< Total FTRAN solves
    int btran_count{0};                         ///< Total BTRAN solves

    Basis final_basis;                          ///< Optimal or terminating basis state
    SimplexTiming timing;                       ///< Component-level timing breakdown

    [[nodiscard]] bool is_optimal() const noexcept {
        return status == solver::TerminationStatus::OPTIMAL;
    }

    [[nodiscard]] std::string format_summary() const {
        std::ostringstream ss;
        ss << "--- Dual Revised Simplex Result ---\n"
           << "  Status:               " << to_string(status) << "\n"
           << "  Objective:            " << std::scientific << std::setprecision(8) << objective_value << "\n"
           << "  Pivots (Iterations):  " << iterations << " (" << bound_flips << " bound flips)\n"
           << "  Factorizations:       " << factorizations << " (plus " << updates << " updates)\n"
           << "  FTRAN / BTRAN Count:  " << ftran_count << " / " << btran_count << "\n"
           << "  Max Primal Infeas:    " << std::scientific << std::setprecision(2) << max_primal_infeasibility << "\n"
           << "  Max Dual Infeas:      " << std::scientific << std::setprecision(2) << max_dual_infeasibility << "\n"
           << "  Solve Time:           " << std::fixed << std::setprecision(2) << timing.total_time_ms << " ms\n"
           << "    - Factorization:    " << timing.factorization_ms << " ms\n"
           << "    - FTRAN / BTRAN:    " << (timing.ftran_ms + timing.btran_ms) << " ms\n"
           << "    - Pricing & Ratio:  " << (timing.pricing_ms + timing.ratio_test_ms) << " ms\n"
           << "-----------------------------------";
        return ss.str();
    }
};

} // namespace pipepye::simplex
