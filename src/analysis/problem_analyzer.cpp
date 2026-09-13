#include <pipepye/analysis/problem_analyzer.hpp>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <queue>
#include <iostream>

namespace pipepye::analysis {

ProblemStats ProblemAnalyzer::analyze(const model::LinearProgram& lp) {
    ProblemStats stats;
    index_t m = lp.num_rows();
    index_t n = lp.num_cols();
    size_t nnz = lp.num_nonzeros();

    stats.num_rows = m;
    stats.num_cols = n;
    stats.num_nonzeros = nnz;
    stats.density = (m > 0 && n > 0) ? (static_cast<double>(nnz) / (static_cast<double>(m) * n)) : 0.0;

    const scalar_t inf_thresh = 1e19;
    const scalar_t eps = 1e-9;

    // 1. Bound Statistics: Variables
    for (index_t j = 0; j < n; ++j) {
        scalar_t lb = lp.col_lower[j];
        scalar_t ub = lp.col_upper[j];
        bool has_lb = (lb > -inf_thresh);
        bool has_ub = (ub < inf_thresh);

        if (!has_lb && !has_ub) {
            ++stats.num_free_vars;
        } else if (has_lb && !has_ub) {
            ++stats.num_bounded_below_vars;
        } else if (!has_lb && has_ub) {
            ++stats.num_bounded_above_vars;
        } else if (std::abs(ub - lb) <= eps) {
            ++stats.num_fixed_vars;
        } else {
            ++stats.num_boxed_vars;
        }
    }

    // 2. Bound Statistics: Constraints
    for (index_t i = 0; i < m; ++i) {
        scalar_t lb = lp.row_lower[i];
        scalar_t ub = lp.row_upper[i];
        bool has_lb = (lb > -inf_thresh);
        bool has_ub = (ub < inf_thresh);

        if (std::abs(ub - lb) <= eps) {
            ++stats.num_equality_rows;
        } else if (!has_lb && !has_ub) {
            ++stats.num_free_rows;
        } else if (has_lb && !has_ub) {
            ++stats.num_inequality_lower_rows;
        } else if (!has_lb && has_ub) {
            ++stats.num_inequality_upper_rows;
        } else {
            ++stats.num_ranged_rows;
        }
    }

    // 3. Objective Statistics
    stats.min_abs_cost = 1e30;
    stats.max_abs_cost = 0.0;
    for (index_t j = 0; j < n; ++j) {
        scalar_t c = std::abs(lp.c[j]);
        if (c <= eps) {
            ++stats.num_zero_cost_vars;
        } else {
            ++stats.num_nonzero_cost_vars;
            stats.min_abs_cost = std::min(stats.min_abs_cost, c);
            stats.max_abs_cost = std::max(stats.max_abs_cost, c);
        }
    }
    if (stats.min_abs_cost == 1e30) stats.min_abs_cost = 0.0;

    // 4. Matrix Coefficients & Degrees
    std::vector<index_t> row_nnz(m, 0);
    std::vector<index_t> col_nnz(n, 0);
    std::vector<std::vector<index_t>> row_col_indices(m);

    stats.min_abs_coeff = 1e30;
    stats.max_abs_coeff = 0.0;
    scalar_t coeff_sum = 0.0;

    index_t max_band_diff = 0;
    size_t profile = 0;

    auto record_nonzero = [&](index_t i, index_t j, scalar_t v) {
        scalar_t abs_v = std::abs(v);
        if (abs_v > 0.0) {
            stats.min_abs_coeff = std::min(stats.min_abs_coeff, abs_v);
            stats.max_abs_coeff = std::max(stats.max_abs_coeff, abs_v);
            coeff_sum += abs_v;

            ++row_nnz[i];
            ++col_nnz[j];
            row_col_indices[i].push_back(j);

            max_band_diff = std::max(max_band_diff, std::abs(i - j));
        }
    };

    if (!lp.csr_row_ptr.empty()) {
        for (index_t i = 0; i < m; ++i) {
            index_t start = lp.csr_row_ptr[i];
            index_t end = lp.csr_row_ptr[i + 1];
            for (index_t p = start; p < end; ++p) {
                record_nonzero(i, lp.csr_col_ind[p], lp.csr_values[p]);
            }
        }
    } else {
        for (const auto& tr : lp.A_coo.triplets()) {
            record_nonzero(tr.row, tr.col, tr.val);
        }
    }

    stats.half_bandwidth = max_band_diff;
    stats.normalized_bandwidth = (std::max(m, n) > 0)
        ? (static_cast<double>(max_band_diff) / std::max(m, n)) : 0.0;

    for (index_t i = 0; i < m; ++i) {
        if (!row_col_indices[i].empty()) {
            index_t min_j = *std::min_element(row_col_indices[i].begin(), row_col_indices[i].end());
            if (i >= min_j) {
                profile += static_cast<size_t>(i - min_j);
            }
        }
    }
    stats.profile_size = profile;

    if (stats.min_abs_coeff == 1e30) stats.min_abs_coeff = 0.0;
    if (stats.min_abs_coeff > 0.0) {
        stats.dynamic_range = stats.max_abs_coeff / stats.min_abs_coeff;
        stats.dynamic_range_orders = std::log10(stats.dynamic_range);
    }
    if (nnz > 0) {
        stats.mean_abs_coeff = coeff_sum / nnz;
        scalar_t coeff_var = 0.0;
        if (!lp.csr_row_ptr.empty()) {
            for (index_t i = 0; i < m; ++i) {
                index_t start = lp.csr_row_ptr[i];
                index_t end = lp.csr_row_ptr[i + 1];
                for (index_t p = start; p < end; ++p) {
                    scalar_t d = std::abs(lp.csr_values[p]) - stats.mean_abs_coeff;
                    coeff_var += d * d;
                }
            }
        } else {
            for (const auto& tr : lp.A_coo.triplets()) {
                scalar_t d = std::abs(tr.val) - stats.mean_abs_coeff;
                coeff_var += d * d;
            }
        }
        stats.stddev_abs_coeff = std::sqrt(coeff_var / nnz);
    }

    // 5. Degree Distributions & Higher Statistical Moments
    if (m > 0) {
        stats.row_nnz_min = *std::min_element(row_nnz.begin(), row_nnz.end());
        stats.row_nnz_max = *std::max_element(row_nnz.begin(), row_nnz.end());
        stats.row_nnz_avg = static_cast<double>(nnz) / m;

        double sum_sq_diff = 0.0;
        double sum_cube_diff = 0.0;
        for (index_t i = 0; i < m; ++i) {
            if (row_nnz[i] == 0) ++stats.num_empty_rows;
            double diff = row_nnz[i] - stats.row_nnz_avg;
            sum_sq_diff += diff * diff;
            sum_cube_diff += diff * diff * diff;
        }
        stats.row_nnz_variance = sum_sq_diff / m;
        stats.row_nnz_stddev = std::sqrt(stats.row_nnz_variance);
        if (stats.row_nnz_stddev > 1e-8) {
            stats.row_nnz_skewness = (sum_cube_diff / m) / std::pow(stats.row_nnz_stddev, 3);
        }
    }

    if (n > 0) {
        stats.col_nnz_min = *std::min_element(col_nnz.begin(), col_nnz.end());
        stats.col_nnz_max = *std::max_element(col_nnz.begin(), col_nnz.end());
        stats.col_nnz_avg = static_cast<double>(nnz) / n;

        double sum_sq_diff = 0.0;
        double sum_cube_diff = 0.0;
        for (index_t j = 0; j < n; ++j) {
            if (col_nnz[j] == 0) ++stats.num_empty_cols;
            double diff = col_nnz[j] - stats.col_nnz_avg;
            sum_sq_diff += diff * diff;
            sum_cube_diff += diff * diff * diff;
        }
        stats.col_nnz_variance = sum_sq_diff / n;
        stats.col_nnz_stddev = std::sqrt(stats.col_nnz_variance);
        if (stats.col_nnz_stddev > 1e-8) {
            stats.col_nnz_skewness = (sum_cube_diff / n) / std::pow(stats.col_nnz_stddev, 3);
        }
    }

    // 6. Imbalance Ratio & Gini Coefficient
    stats.row_length_imbalance = (stats.row_nnz_avg > 0.0)
        ? (static_cast<double>(stats.row_nnz_max) / stats.row_nnz_avg) : 1.0;

    if (m > 0 && nnz > 0) {
        std::vector<index_t> sorted_rows = row_nnz;
        std::sort(sorted_rows.begin(), sorted_rows.end());
        double weighted_sum = 0.0;
        for (index_t k = 0; k < m; ++k) {
            weighted_sum += (k + 1) * sorted_rows[k];
        }
        stats.row_length_gini = (2.0 * weighted_sum) / (m * static_cast<double>(nnz)) - (static_cast<double>(m + 1) / m);
        stats.row_length_gini = std::max(0.0, std::min(1.0, stats.row_length_gini));
    }

    // 7. Staircase Score (Correlation between row index and median column index)
    std::vector<double> row_medians;
    std::vector<double> active_row_indices;
    for (index_t i = 0; i < m; ++i) {
        if (!row_col_indices[i].empty()) {
            std::sort(row_col_indices[i].begin(), row_col_indices[i].end());
            size_t sz = row_col_indices[i].size();
            double med = (sz % 2 == 1)
                ? row_col_indices[i][sz / 2]
                : 0.5 * (row_col_indices[i][sz / 2 - 1] + row_col_indices[i][sz / 2]);
            row_medians.push_back(med);
            active_row_indices.push_back(i);
        }
    }

    if (active_row_indices.size() >= 3) {
        double mean_r = 0.0, mean_c = 0.0;
        size_t k_count = active_row_indices.size();
        for (size_t k = 0; k < k_count; ++k) {
            mean_r += active_row_indices[k];
            mean_c += row_medians[k];
        }
        mean_r /= k_count;
        mean_c /= k_count;

        double cov = 0.0, var_r = 0.0, var_c = 0.0;
        for (size_t k = 0; k < k_count; ++k) {
            double dr = active_row_indices[k] - mean_r;
            double dc = row_medians[k] - mean_c;
            cov += dr * dc;
            var_r += dr * dr;
            var_c += dc * dc;
        }
        if (var_r > 1e-8 && var_c > 1e-8) {
            stats.staircase_score = cov / (std::sqrt(var_r) * std::sqrt(var_c));
        }
    }

    // 8. Connected Components (Bipartite Graph: m rows + n cols)
    index_t total_nodes = m + n;
    std::vector<int> parent(total_nodes);
    std::iota(parent.begin(), parent.end(), 0);

    auto find_root = [&](auto self, int u) -> int {
        return (parent[u] == u) ? u : (parent[u] = self(self, parent[u]));
    };

    auto union_nodes = [&](int u, int v) {
        int ru = find_root(find_root, u);
        int rv = find_root(find_root, v);
        if (ru != rv) parent[ru] = rv;
    };

    std::vector<bool> has_edges(total_nodes, false);
    for (index_t i = 0; i < m; ++i) {
        for (index_t j : row_col_indices[i]) {
            union_nodes(i, m + j);
            has_edges[i] = true;
            has_edges[m + j] = true;
        }
    }

    std::vector<bool> seen_roots(total_nodes, false);
    int components = 0;
    for (index_t u = 0; u < total_nodes; ++u) {
        if (has_edges[u]) {
            int r = find_root(find_root, u);
            if (!seen_roots[r]) {
                seen_roots[r] = true;
                ++components;
            }
        }
    }
    stats.num_connected_components = components;

    // 9. Kernel Engine Recommendation (derived from Phase 1 empirical findings)
    if (stats.num_nonzeros < 15000) {
        stats.recommended_engine = RecommendedEngine::CPU_SingleThread;
        stats.recommendation_reason =
            "NNZ (" + std::to_string(stats.num_nonzeros) +
            " < 15,000) is dominated by CPU L1/L2 cache (1.6-6.0 us); GPU kernel launch overhead exceeds computation.";
    } else if (stats.num_nonzeros <= 30000) {
        stats.recommended_engine = RecommendedEngine::CPU_MultiThread;
        stats.recommendation_reason =
            "NNZ (" + std::to_string(stats.num_nonzeros) +
            ") falls in the empirical crossover zone (15,000-30,000); multithreaded OpenMP provides low latency without PCIe transfers.";
    } else {
        // NNZ > 30,000: GPU dominant zone
        if (stats.row_length_gini >= 0.35 || stats.row_length_imbalance >= 4.0) {
            stats.recommended_engine = RecommendedEngine::GPU_MergePath;
            stats.recommendation_reason =
                "High row-length imbalance (Gini=" + std::to_string(stats.row_length_gini) +
                ", Imbalance=" + std::to_string(stats.row_length_imbalance) +
                "x); Merge-Path eliminates warp divergence by partitioning nonzeros evenly across threads.";
        } else if (stats.row_nnz_stddev > 1.5 * stats.row_nnz_avg) {
            stats.recommended_engine = RecommendedEngine::GPU_RowAdaptive;
            stats.recommendation_reason =
                "Moderate row degree variance; dynamic row-adaptive assignment provides balanced SM occupancy.";
        } else {
            stats.recommended_engine = RecommendedEngine::GPU_CSR_Basic;
            stats.recommendation_reason =
                "Uniform row distributions (Gini=" + std::to_string(stats.row_length_gini) +
                " < 0.35); standard 1-thread-per-row CSR SpMV achieves peak coalesced memory throughput.";
        }
    }

    return stats;
}

} // namespace pipepye::analysis
