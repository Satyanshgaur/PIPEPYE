#pragma once

#include <string>
#include <vector>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <pipepye/core/types.hpp>
#include <pipepye/model/lp_model.hpp>
#include <pipepye/presolve/presolve_types.hpp>

namespace pipepye::scaling {

/// @brief Vector norm statistical summary.
struct NormStats {
    scalar_t min_norm{0.0};
    scalar_t max_norm{0.0};
    scalar_t mean_norm{0.0};
    scalar_t stddev_norm{0.0};
};

/// @brief Quantitative diagnostic telemetry measuring numerical characteristics before and after scaling.
struct ScalingDiagnostics {
    scalar_t min_abs_coeff{0.0};
    scalar_t max_abs_coeff{0.0};
    scalar_t dynamic_range{0.0};            ///< max |A_ij| / min |A_ij|
    scalar_t dynamic_range_orders{0.0};     ///< log10(max / min)

    NormStats row_l1_stats;
    NormStats row_l2_stats;
    NormStats row_linf_stats;

    NormStats col_l1_stats;
    NormStats col_l2_stats;
    NormStats col_linf_stats;

    scalar_t row_conditioning_proxy{0.0};   ///< max(row_l2) / min(row_l2)
    scalar_t col_conditioning_proxy{0.0};   ///< max(col_l2) / min(col_l2)
    scalar_t overall_conditioning_proxy{0.0};

    [[nodiscard]] std::string format_report(const std::string& title = "SCALING DIAGNOSTICS") const {
        std::ostringstream ss;
        ss << "=== " << title << " ===\n";
        ss << std::scientific << std::setprecision(3);
        ss << "  Coeff Min / Max:        " << min_abs_coeff << " / " << max_abs_coeff << "\n";
        ss << "  Dynamic Range:          " << dynamic_range << " (" << std::fixed << std::setprecision(1)
           << dynamic_range_orders << " orders of magnitude)\n";
        ss << std::scientific << std::setprecision(3);
        ss << "  Row Linf Min/Max/Mean:  " << row_linf_stats.min_norm << " / " << row_linf_stats.max_norm
           << " / " << row_linf_stats.mean_norm << " (stddev: " << row_linf_stats.stddev_norm << ")\n";
        ss << "  Col Linf Min/Max/Mean:  " << col_linf_stats.min_norm << " / " << col_linf_stats.max_norm
           << " / " << col_linf_stats.mean_norm << " (stddev: " << col_linf_stats.stddev_norm << ")\n";
        ss << "  Conditioning Proxies:   Row=" << row_conditioning_proxy
           << ", Col=" << col_conditioning_proxy
           << ", Overall=" << overall_conditioning_proxy << "\n";
        ss << "========================================\n";
        return ss.str();
    }
};

/// @brief Matrix scaling strategy algorithm.
enum class ScalingMethod : uint8_t {
    Ruiz = 0,               ///< Ruiz iterative infinity-norm equilibration
    PockChambolle,          ///< Pock-Chambolle diagonal preconditioning (alpha = 1.0)
    L2Equilibration,        ///< L2-norm iterative equilibration
    None                    ///< Identity scaling (no-op)
};

[[nodiscard]] inline std::string to_string(ScalingMethod method) {
    switch (method) {
        case ScalingMethod::Ruiz:            return "Ruiz";
        case ScalingMethod::PockChambolle:   return "PockChambolle";
        case ScalingMethod::L2Equilibration: return "L2Equilibration";
        case ScalingMethod::None:            return "None";
    }
    return "Unknown";
}

/// @brief Parameters controlling scaling iteration and tolerances.
struct ScalingOptions {
    ScalingMethod method{ScalingMethod::Ruiz};
    int max_iterations{20};
    scalar_t tolerance{1e-4};
    scalar_t alpha{1.0};           ///< Parameter for Pock-Chambolle scaling (default 1.0)
    bool verbose{false};
};

/// @brief Result container storing the scaled linear program and reversible scaling diagonals.
struct ScaledModel {
    model::LinearProgram lp;
    std::vector<scalar_t> row_scale_R;       ///< R_i: row multipliers (length: m)
    std::vector<scalar_t> col_scale_C;       ///< C_j: col multipliers (length: n)
    std::vector<scalar_t> inv_row_scale_R;   ///< 1 / R_i
    std::vector<scalar_t> inv_col_scale_C;   ///< 1 / C_j
    ScalingDiagnostics diag_before;
    ScalingDiagnostics diag_after;
    int iterations_performed{0};
    double elapsed_ms{0.0};

    /// @brief Recovers original unscaled primal-dual solution from scaled solution.
    /// x = C * x_scaled, y = R * y_scaled, s = C^{-1} * s_scaled.
    [[nodiscard]] presolve::PrimalDualSolution unscale_solution(
        const presolve::PrimalDualSolution& scaled_sol) const;

    /// @brief Scales an unscaled primal-dual solution into the scaled coordinate space.
    /// x_scaled = C^{-1} * x, y_scaled = R^{-1} * y, s_scaled = C * s.
    [[nodiscard]] presolve::PrimalDualSolution scale_solution(
        const presolve::PrimalDualSolution& unscaled_sol) const;

    /// @brief Exact inverse model recovery: transforms scaled LP back to original LP.
    [[nodiscard]] model::LinearProgram unscale_model() const;
};

} // namespace pipepye::scaling
