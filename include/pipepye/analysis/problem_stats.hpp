#pragma once

#include <string>
#include <vector>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <pipepye/core/types.hpp>
#include <pipepye/analysis/conditioning_proxy.hpp>

namespace pipepye::analysis {

/// @brief Recommended computational kernel engine based on problem structure and empirical crossover.
enum class RecommendedEngine : uint8_t {
    CPU_SingleThread = 0,    ///< Problem NNZ < 15,000: CPU L1/L2 cache execution wins decisively.
    CPU_MultiThread,         ///< Moderate NNZ in crossover region on multi-core host.
    GPU_CSR_Basic,           ///< Large NNZ with uniform row lengths (low Gini coefficient).
    GPU_RowAdaptive,         ///< Large NNZ with moderate variance in row nonzeros.
    GPU_MergePath            ///< Large NNZ with severe irregular hub imbalance (high Gini coefficient).
};

[[nodiscard]] inline std::string to_string(RecommendedEngine engine) {
    switch (engine) {
        case RecommendedEngine::CPU_SingleThread: return "CPU_SingleThread";
        case RecommendedEngine::CPU_MultiThread:  return "CPU_MultiThread";
        case RecommendedEngine::GPU_CSR_Basic:    return "GPU_CSR_Basic";
        case RecommendedEngine::GPU_RowAdaptive:  return "GPU_RowAdaptive";
        case RecommendedEngine::GPU_MergePath:    return "GPU_MergePath";
    }
    return "Unknown";
}

/// @brief Comprehensive statistical profile characterizing LP problem geometry and sparsity structure.
struct ProblemStats {
    // 1. Fundamental Dimensions & Density
    index_t num_rows{0};
    index_t num_cols{0};
    size_t num_nonzeros{0};
    double density{0.0};                  ///< NNZ / (rows * cols)

    // 2. Matrix Coefficient Distributions
    scalar_t min_abs_coeff{0.0};
    scalar_t max_abs_coeff{0.0};
    scalar_t dynamic_range{0.0};          ///< max / min
    scalar_t dynamic_range_orders{0.0};   ///< log10(max / min)
    scalar_t mean_abs_coeff{0.0};
    scalar_t stddev_abs_coeff{0.0};

    // 3. Bound Characterization
    index_t num_free_vars{0};
    index_t num_bounded_below_vars{0};
    index_t num_bounded_above_vars{0};
    index_t num_boxed_vars{0};
    index_t num_fixed_vars{0};

    index_t num_equality_rows{0};
    index_t num_inequality_lower_rows{0};
    index_t num_inequality_upper_rows{0};
    index_t num_ranged_rows{0};
    index_t num_free_rows{0};

    // 4. Objective Characteristics
    index_t num_zero_cost_vars{0};
    index_t num_nonzero_cost_vars{0};
    scalar_t min_abs_cost{0.0};
    scalar_t max_abs_cost{0.0};

    // 5. Degree Distributions (Row & Column NNZ)
    index_t row_nnz_min{0};
    index_t row_nnz_max{0};
    double row_nnz_avg{0.0};
    double row_nnz_variance{0.0};
    double row_nnz_stddev{0.0};
    double row_nnz_skewness{0.0};

    index_t col_nnz_min{0};
    index_t col_nnz_max{0};
    double col_nnz_avg{0.0};
    double col_nnz_variance{0.0};
    double col_nnz_stddev{0.0};
    double col_nnz_skewness{0.0};

    index_t num_empty_rows{0};
    index_t num_empty_cols{0};

    // 6. Advanced Sparsity Topology & Structural Metrics
    index_t half_bandwidth{0};            ///< max |i - j| for A_{i, j} != 0
    double normalized_bandwidth{0.0};     ///< half_bandwidth / max(m, n)
    size_t profile_size{0};               ///< Sum of active envelope lengths
    double row_length_imbalance{0.0};     ///< row_nnz_max / avg
    double row_length_gini{0.0};          ///< Gini coefficient in [0, 1]
    double staircase_score{0.0};          ///< Correlation between row index and column median
    int num_connected_components{0};      ///< Connected components in bipartite graph

    // 7. Numerical Conditioning Proxies
    ConditioningProxy conditioning_proxy;

    // 8. Memory Footprint Estimates
    double estimated_host_ram_mb{0.0};
    double estimated_gpu_vram_mb{0.0};

    // 9. Empirical Hardware Engine Recommendation
    RecommendedEngine recommended_engine{RecommendedEngine::CPU_SingleThread};
    std::string recommendation_reason;

    [[nodiscard]] std::string row_imbalance_rating() const {
        if (row_length_gini >= 0.50 || row_length_imbalance >= 8.0) return "extreme";
        if (row_length_gini >= 0.35 || row_length_imbalance >= 4.0) return "high";
        if (row_length_gini >= 0.20 || row_length_imbalance >= 2.0) return "moderate";
        return "low";
    }

    [[nodiscard]] std::string format_count(size_t val) const {
        std::ostringstream ss;
        if (val >= 1000000) {
            ss << std::fixed << std::setprecision(1) << (static_cast<double>(val) / 1e6) << "M";
        } else if (val >= 1000) {
            ss << std::fixed << std::setprecision(1) << (static_cast<double>(val) / 1e3) << "K";
        } else {
            ss << val;
        }
        return ss.str();
    }

    [[nodiscard]] std::string format_memory(double mb) const {
        std::ostringstream ss;
        if (mb >= 1024.0) {
            ss << std::fixed << std::setprecision(1) << (mb / 1024.0) << " GB";
        } else if (mb >= 1.0) {
            ss << std::fixed << std::setprecision(1) << mb << " MB";
        } else {
            ss << std::fixed << std::setprecision(0) << (mb * 1024.0) << " KB";
        }
        return ss.str();
    }

    [[nodiscard]] std::string format_coeff_range() const {
        if (min_abs_coeff <= 0.0 && max_abs_coeff <= 0.0) return "0";
        auto to_exp = [](scalar_t val) -> std::string {
            int exp = static_cast<int>(std::floor(std::log10(val > 0.0 ? val : 1.0)));
            return "10^" + std::to_string(exp);
        };
        return to_exp(min_abs_coeff) + "–" + to_exp(max_abs_coeff);
    }

    /// @brief Generates canonical single-line unified report banner for fast CLI inspection and dispatch.
    /// Example: "12.4M variables, 8.1M constraints, 0.003% density, coefficient range 10^-7–10^6, row imbalance high, estimated VRAM 3.2 GB"
    [[nodiscard]] std::string format_one_line_summary() const {
        std::ostringstream ss;
        ss << format_count(num_cols) << " variables, "
           << format_count(num_rows) << " constraints, "
           << std::fixed << std::setprecision(3) << (density * 100.0) << "% density, "
           << "coefficient range " << format_coeff_range() << ", "
           << "row imbalance " << row_imbalance_rating() << ", "
           << "estimated VRAM " << format_memory(estimated_gpu_vram_mb);
        return ss.str();
    }

    [[nodiscard]] std::string format_report() const {
        std::ostringstream ss;
        ss << "================================================================================\n";
        ss << "                    PIPEPYE PROBLEM CHARACTERIZATION REPORT                     \n";
        ss << "================================================================================\n";
        ss << "SUMMARY BANNER:\n";
        ss << "  " << format_one_line_summary() << "\n";
        ss << "--------------------------------------------------------------------------------\n";
        ss << "DIMENSIONS & SPARSITY:\n";
        ss << "  Rows: " << num_rows << ", Columns: " << num_cols << ", Nonzeros (NNZ): " << num_nonzeros << "\n";
        ss << "  Density: " << std::fixed << std::setprecision(5) << (density * 100.0) << "%\n";
        ss << "  Empty Rows: " << num_empty_rows << ", Empty Columns: " << num_empty_cols << "\n";
        ss << "--------------------------------------------------------------------------------\n";
        ss << "COEFFICIENT DYNAMICS & CONDITIONING PROXIES:\n";
        ss << std::scientific << std::setprecision(3);
        ss << "  Magnitude Min / Max:        " << min_abs_coeff << " / " << max_abs_coeff << "\n";
        ss << "  Dynamic Range (Proxy):      " << dynamic_range << " (" << std::fixed << std::setprecision(1)
           << dynamic_range_orders << " orders of magnitude)\n";
        ss << std::scientific << std::setprecision(3);
        ss << "  Mean / StdDev Coeff:        " << mean_abs_coeff << " / " << stddev_abs_coeff << "\n";
        ss << "  Row/Col Norm Ratio Proxies: Row=" << conditioning_proxy.row_norm_ratio_proxy
           << ", Col=" << conditioning_proxy.col_norm_ratio_proxy << "\n";
        ss << "  Spectral Cond Proxy:        " << conditioning_proxy.spectral_conditioning_proxy
           << " (spectral norm est: " << conditioning_proxy.spectral_norm_estimate << ")\n";
        ss << "--------------------------------------------------------------------------------\n";
        ss << "BOUNDS & VARIABLES:\n";
        ss << "  Variables:   " << num_boxed_vars << " Boxed, " << num_bounded_below_vars << " Bounded Below, "
           << num_bounded_above_vars << " Bounded Above, " << num_free_vars << " Free, " << num_fixed_vars << " Fixed\n";
        ss << "  Constraints: " << num_equality_rows << " Equality, " << num_ranged_rows << " Ranged, "
           << num_inequality_lower_rows << " Lower-bounded, " << num_inequality_upper_rows << " Upper-bounded\n";
        ss << "  Objective:   " << num_nonzero_cost_vars << " Non-zero, " << num_zero_cost_vars << " Zero coefficients\n";
        ss << "--------------------------------------------------------------------------------\n";
        ss << "SPARSITY DISTRIBUTION & TOPOLOGY:\n";
        ss << std::fixed << std::setprecision(2);
        ss << "  Row NNZ:  Min=" << row_nnz_min << ", Max=" << row_nnz_max << ", Avg=" << row_nnz_avg
           << ", StdDev=" << row_nnz_stddev << ", Skew=" << row_nnz_skewness << "\n";
        ss << "  Col NNZ:  Min=" << col_nnz_min << ", Max=" << col_nnz_max << ", Avg=" << col_nnz_avg
           << ", StdDev=" << col_nnz_stddev << ", Skew=" << col_nnz_skewness << "\n";
        ss << "  Half-Bandwidth:             " << half_bandwidth << " (Normalized: " << std::setprecision(4)
           << normalized_bandwidth << ")\n";
        ss << "  Row Imbalance Ratio:        " << std::setprecision(2) << row_length_imbalance << "x ("
           << row_imbalance_rating() << ")\n";
        ss << "  Row Gini Coefficient:       " << std::setprecision(4) << row_length_gini << " (0=Uniform, 1=Heavy Hubs)\n";
        ss << "  Staircase Score:            " << std::setprecision(3) << staircase_score << "\n";
        ss << "  Connected Components:       " << num_connected_components << "\n";
        ss << "--------------------------------------------------------------------------------\n";
        ss << "ESTIMATED MEMORY CONSUMPTION:\n";
        ss << "  Estimated Host RAM:         " << format_memory(estimated_host_ram_mb) << "\n";
        ss << "  Estimated GPU VRAM:         " << format_memory(estimated_gpu_vram_mb) << "\n";
        ss << "--------------------------------------------------------------------------------\n";
        ss << "RECOMMENDED COMPUTATIONAL ENGINE:\n";
        ss << "  Selection: " << to_string(recommended_engine) << "\n";
        ss << "  Rationale: " << recommendation_reason << "\n";
        ss << "================================================================================\n";
        return ss.str();
    }
};

} // namespace pipepye::analysis
