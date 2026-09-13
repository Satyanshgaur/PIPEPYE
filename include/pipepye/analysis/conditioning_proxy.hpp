#pragma once

#include <string>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <pipepye/core/types.hpp>
#include <pipepye/model/lp_model.hpp>

namespace pipepye::analysis {

/// @brief Practical numerical conditioning proxies and cheap estimators.
/// @note IMPORTANT: These metrics are explicit condition PROXIES and cheap indicators,
/// NOT exact matrix condition numbers kappa_2(A). Exact condition numbers require
/// full SVD factorization which is computationally prohibitive on large sparse systems.
struct ConditioningProxy {
    scalar_t magnitude_range_proxy{0.0};          ///< max |A_ij| / min |A_ij| across nonzeros
    scalar_t log10_magnitude_range{0.0};          ///< log10 of magnitude ratio (orders of magnitude)
    scalar_t row_norm_ratio_proxy{0.0};           ///< max_i ||A_{i,*}||_2 / min_i ||A_{i,*}||_2
    scalar_t col_norm_ratio_proxy{0.0};           ///< max_j ||A_{*,j}||_2 / min_j ||A_{*,j}||_2
    scalar_t norm_imbalance_proxy{0.0};           ///< max(row_norm_ratio_proxy, col_norm_ratio_proxy)
    scalar_t spectral_norm_estimate{0.0};         ///< Cheap power-iteration estimate of ||A||_2
    scalar_t spectral_conditioning_proxy{0.0};    ///< spectral_norm_estimate / min(row/col norm)

    [[nodiscard]] std::string format_summary() const {
        std::ostringstream ss;
        ss << std::scientific << std::setprecision(2);
        ss << "Magnitude Range Proxy: " << magnitude_range_proxy << " ("
           << std::fixed << std::setprecision(1) << log10_magnitude_range << " orders)\n";
        ss << std::scientific << std::setprecision(2);
        ss << "Row Norm Ratio Proxy:  " << row_norm_ratio_proxy << "\n";
        ss << "Col Norm Ratio Proxy:  " << col_norm_ratio_proxy << "\n";
        ss << "Norm Imbalance Proxy:  " << norm_imbalance_proxy << "\n";
        ss << "Spectral Norm Est:     " << spectral_norm_estimate << "\n";
        ss << "Spectral Cond Proxy:   " << spectral_conditioning_proxy << "\n";
        return ss.str();
    }
};

/// @brief Cheap estimator computing practical conditioning proxies without expensive factorizations.
class ConditioningEstimator {
public:
    ConditioningEstimator() = default;

    /// @brief Computes conditioning proxies using sparse coefficient traversal and 5 power iterations.
    [[nodiscard]] static ConditioningProxy estimate(const model::LinearProgram& lp, int power_iterations = 5);
};

} // namespace pipepye::analysis
