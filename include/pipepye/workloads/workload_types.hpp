#pragma once

#include <pipepye/core/types.hpp>
#include <pipepye/analysis/problem_stats.hpp>
#include <string>
#include <sstream>
#include <iomanip>

namespace pipepye::workloads {

enum class InstanceScale : uint8_t {
    Toy = 0,
    Small,
    Medium,
    Large,
    Stress
};

[[nodiscard]] inline std::string to_string(InstanceScale scale) {
    switch (scale) {
        case InstanceScale::Toy:    return "Toy";
        case InstanceScale::Small:  return "Small";
        case InstanceScale::Medium: return "Medium";
        case InstanceScale::Large:  return "Large";
        case InstanceScale::Stress: return "Stress";
    }
    return "Unknown";
}

/// @brief Metadata schema mandated by Phase 5 specification for industrial workloads.
struct WorkloadMetadata {
    // 1. Identification
    std::string problem_name;
    std::string instance_id;
    std::string generator_version{"1.0.0"};
    uint64_t random_seed{42};
    InstanceScale scale{InstanceScale::Small};

    // 2. Dimensions & Sparsity
    index_t num_rows{0};
    index_t num_cols{0};
    size_t num_nonzeros{0};
    double density{0.0};

    // 3. Variable Types
    index_t num_continuous_vars{0};
    index_t num_integer_vars{0};
    index_t num_binary_vars{0};

    // 4. Coefficient & Bound Ranges
    scalar_t min_abs_coeff{0.0};
    scalar_t max_abs_coeff{0.0};
    scalar_t min_rhs{0.0};
    scalar_t max_rhs{0.0};
    scalar_t min_bound{0.0};
    scalar_t max_bound{0.0};

    // 5. Structural Statistics (derived from ProblemStats)
    double row_nnz_avg{0.0};
    double row_nnz_stddev{0.0};
    double col_nnz_avg{0.0};
    double col_nnz_stddev{0.0};
    index_t half_bandwidth{0};
    double normalized_bandwidth{0.0};
    double staircase_score{0.0};
    double row_length_gini{0.0};
    int num_connected_components{0};
    double coupling_density{0.0};

    // 6. Pre-Registered Predictions
    std::string expected_solver;   // e.g. "DualSimplex", "PDHG_GPU", "BranchAndBound"
    std::string expected_backend;  // e.g. "CPU", "GPU"
    std::string prediction_rationale;

    // 7. Reference Solution (if known)
    bool has_reference_solution{false};
    scalar_t reference_objective{0.0};
    std::string reference_source;

    /// @brief Populates structural fields from computed ProblemStats.
    void populate_from_stats(const analysis::ProblemStats& stats) {
        num_rows = stats.num_rows;
        num_cols = stats.num_cols;
        num_nonzeros = stats.num_nonzeros;
        density = stats.density;

        min_abs_coeff = stats.min_abs_coeff;
        max_abs_coeff = stats.max_abs_coeff;

        row_nnz_avg = stats.row_nnz_avg;
        row_nnz_stddev = stats.row_nnz_stddev;
        col_nnz_avg = stats.col_nnz_avg;
        col_nnz_stddev = stats.col_nnz_stddev;
        half_bandwidth = stats.half_bandwidth;
        normalized_bandwidth = stats.normalized_bandwidth;
        staircase_score = stats.staircase_score;
        row_length_gini = stats.row_length_gini;
        num_connected_components = stats.num_connected_components;
        coupling_density = (num_rows > 0 && num_cols > 0) ?
            static_cast<double>(num_nonzeros) / (num_rows * num_cols) : 0.0;
    }

    [[nodiscard]] std::string to_json() const {
        std::ostringstream ss;
        ss << "{\n"
           << "  \"problem_name\": \"" << problem_name << "\",\n"
           << "  \"instance_id\": \"" << instance_id << "\",\n"
           << "  \"generator_version\": \"" << generator_version << "\",\n"
           << "  \"random_seed\": " << random_seed << ",\n"
           << "  \"scale\": \"" << to_string(scale) << "\",\n"
           << "  \"dimensions\": {\n"
           << "    \"rows\": " << num_rows << ",\n"
           << "    \"cols\": " << num_cols << ",\n"
           << "    \"nnz\": " << num_nonzeros << ",\n"
           << "    \"density\": " << std::scientific << std::setprecision(6) << density << "\n"
           << "  },\n"
           << "  \"variable_types\": {\n"
           << "    \"continuous\": " << num_continuous_vars << ",\n"
           << "    \"integer\": " << num_integer_vars << ",\n"
           << "    \"binary\": " << num_binary_vars << "\n"
           << "  },\n"
           << "  \"ranges\": {\n"
           << "    \"min_abs_coeff\": " << std::scientific << min_abs_coeff << ",\n"
           << "    \"max_abs_coeff\": " << std::scientific << max_abs_coeff << ",\n"
           << "    \"min_rhs\": " << std::fixed << std::setprecision(4) << min_rhs << ",\n"
           << "    \"max_rhs\": " << std::fixed << std::setprecision(4) << max_rhs << ",\n"
           << "    \"min_bound\": " << std::fixed << std::setprecision(4) << min_bound << ",\n"
           << "    \"max_bound\": " << std::fixed << std::setprecision(4) << max_bound << "\n"
           << "  },\n"
           << "  \"structural_statistics\": {\n"
           << "    \"row_nnz_avg\": " << std::fixed << std::setprecision(2) << row_nnz_avg << ",\n"
           << "    \"row_nnz_stddev\": " << std::fixed << std::setprecision(2) << row_nnz_stddev << ",\n"
           << "    \"col_nnz_avg\": " << std::fixed << std::setprecision(2) << col_nnz_avg << ",\n"
           << "    \"col_nnz_stddev\": " << std::fixed << std::setprecision(2) << col_nnz_stddev << ",\n"
           << "    \"half_bandwidth\": " << half_bandwidth << ",\n"
           << "    \"normalized_bandwidth\": " << std::fixed << std::setprecision(4) << normalized_bandwidth << ",\n"
           << "    \"staircase_score\": " << std::fixed << std::setprecision(4) << staircase_score << ",\n"
           << "    \"row_length_gini\": " << std::fixed << std::setprecision(4) << row_length_gini << ",\n"
           << "    \"num_connected_components\": " << num_connected_components << ",\n"
           << "    \"coupling_density\": " << std::scientific << coupling_density << "\n"
           << "  },\n"
           << "  \"pre_registered_prediction\": {\n"
           << "    \"expected_solver\": \"" << expected_solver << "\",\n"
           << "    \"expected_backend\": \"" << expected_backend << "\",\n"
           << "    \"rationale\": \"" << prediction_rationale << "\"\n"
           << "  },\n"
           << "  \"reference_solution\": {\n"
           << "    \"has_reference\": " << (has_reference_solution ? "true" : "false") << ",\n"
           << "    \"objective\": " << std::scientific << reference_objective << ",\n"
           << "    \"source\": \"" << reference_source << "\"\n"
           << "  }\n"
           << "}\n";
        return ss.str();
    }
};

} // namespace pipepye::workloads
