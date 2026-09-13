#include <pipepye/analysis/conditioning_proxy.hpp>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <vector>

namespace pipepye::analysis {

ConditioningProxy ConditioningEstimator::estimate(const model::LinearProgram& lp, int power_iterations) {
    ConditioningProxy proxy;
    index_t m = lp.num_rows();
    index_t n = lp.num_cols();

    if (m == 0 || n == 0 || lp.num_nonzeros() == 0) {
        return proxy;
    }

    // 1. Traverse nonzeros for min/max magnitudes and row/col Euclidean norms
    scalar_t min_val = 1e30;
    scalar_t max_val = 0.0;

    std::vector<scalar_t> row_sq_sum(m, 0.0);
    std::vector<scalar_t> col_sq_sum(n, 0.0);

    if (!lp.csr_row_ptr.empty()) {
        for (index_t i = 0; i < m; ++i) {
            index_t start = lp.csr_row_ptr[i];
            index_t end = lp.csr_row_ptr[i + 1];
            for (index_t p = start; p < end; ++p) {
                scalar_t abs_v = std::abs(lp.csr_values[p]);
                if (abs_v > 0.0) {
                    min_val = std::min(min_val, abs_v);
                    max_val = std::max(max_val, abs_v);
                    row_sq_sum[i] += abs_v * abs_v;
                    col_sq_sum[lp.csr_col_ind[p]] += abs_v * abs_v;
                }
            }
        }
    } else {
        for (const auto& tr : lp.A_coo.triplets()) {
            scalar_t abs_v = std::abs(tr.val);
            if (abs_v > 0.0) {
                min_val = std::min(min_val, abs_v);
                max_val = std::max(max_val, abs_v);
                row_sq_sum[tr.row] += abs_v * abs_v;
                col_sq_sum[tr.col] += abs_v * abs_v;
            }
        }
    }

    if (min_val == 1e30) min_val = 0.0;
    if (min_val > 0.0) {
        proxy.magnitude_range_proxy = max_val / min_val;
        proxy.log10_magnitude_range = std::log10(proxy.magnitude_range_proxy);
    }

    // Row norm stats
    scalar_t min_row_l2 = 1e30;
    scalar_t max_row_l2 = 0.0;
    for (index_t i = 0; i < m; ++i) {
        scalar_t l2 = std::sqrt(row_sq_sum[i]);
        if (l2 > 0.0) min_row_l2 = std::min(min_row_l2, l2);
        max_row_l2 = std::max(max_row_l2, l2);
    }
    if (min_row_l2 == 1e30) min_row_l2 = 0.0;
    proxy.row_norm_ratio_proxy = (min_row_l2 > 1e-12) ? (max_row_l2 / min_row_l2) : max_row_l2;

    // Col norm stats
    scalar_t min_col_l2 = 1e30;
    scalar_t max_col_l2 = 0.0;
    for (index_t j = 0; j < n; ++j) {
        scalar_t l2 = std::sqrt(col_sq_sum[j]);
        if (l2 > 0.0) min_col_l2 = std::min(min_col_l2, l2);
        max_col_l2 = std::max(max_col_l2, l2);
    }
    if (min_col_l2 == 1e30) min_col_l2 = 0.0;
    proxy.col_norm_ratio_proxy = (min_col_l2 > 1e-12) ? (max_col_l2 / min_col_l2) : max_col_l2;

    proxy.norm_imbalance_proxy = std::max(proxy.row_norm_ratio_proxy, proxy.col_norm_ratio_proxy);

    // 2. Cheap Power-Iteration for Spectral Norm ||A||_2 Estimate
    // Executes 5 iterations of w = A v, u = A^T w on normalized vectors
    std::vector<scalar_t> v(n, 1.0 / std::sqrt(static_cast<scalar_t>(n)));
    std::vector<scalar_t> w(m, 0.0);
    std::vector<scalar_t> u(n, 0.0);

    scalar_t sigma_est = 0.0;

    for (int iter = 0; iter < power_iterations; ++iter) {
        // w = A * v
        if (!lp.csr_row_ptr.empty()) {
            for (index_t i = 0; i < m; ++i) {
                scalar_t sum = 0.0;
                index_t start = lp.csr_row_ptr[i];
                index_t end = lp.csr_row_ptr[i + 1];
                for (index_t p = start; p < end; ++p) {
                    sum += lp.csr_values[p] * v[lp.csr_col_ind[p]];
                }
                w[i] = sum;
            }
        } else {
            std::fill(w.begin(), w.end(), 0.0);
            for (const auto& tr : lp.A_coo.triplets()) {
                w[tr.row] += tr.val * v[tr.col];
            }
        }

        scalar_t w_norm = 0.0;
        for (index_t i = 0; i < m; ++i) w_norm += w[i] * w[i];
        w_norm = std::sqrt(w_norm);
        if (w_norm < 1e-15) break;

        // u = A^T * w
        if (!lp.csc_col_ptr.empty()) {
            for (index_t j = 0; j < n; ++j) {
                scalar_t sum = 0.0;
                index_t start = lp.csc_col_ptr[j];
                index_t end = lp.csc_col_ptr[j + 1];
                for (index_t p = start; p < end; ++p) {
                    sum += lp.csc_values[p] * w[lp.csc_row_ind[p]];
                }
                u[j] = sum;
            }
        } else {
            std::fill(u.begin(), u.end(), 0.0);
            for (const auto& tr : lp.A_coo.triplets()) {
                u[tr.col] += tr.val * w[tr.row];
            }
        }

        scalar_t u_norm = 0.0;
        for (index_t j = 0; j < n; ++j) u_norm += u[j] * u[j];
        u_norm = std::sqrt(u_norm);
        if (u_norm < 1e-15) break;

        sigma_est = std::sqrt(u_norm);
        for (index_t j = 0; j < n; ++j) v[j] = u[j] / u_norm;
    }

    proxy.spectral_norm_estimate = sigma_est;
    scalar_t min_pivot = std::min((min_row_l2 > 0.0 ? min_row_l2 : 1.0),
                                  (min_col_l2 > 0.0 ? min_col_l2 : 1.0));
    proxy.spectral_conditioning_proxy = (min_pivot > 1e-12) ? (sigma_est / min_pivot) : sigma_est;

    return proxy;
}

} // namespace pipepye::analysis
