#include <pipepye/scaling/equilibrator.hpp>
#include <pipepye/utils/timer.hpp>
#include <pipepye/sparse/coo_matrix.hpp>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <iostream>

namespace pipepye::scaling {

static NormStats compute_norm_summary(const std::vector<scalar_t>& norms) {
    NormStats s;
    if (norms.empty()) return s;

    s.min_norm = 1e30;
    s.max_norm = 0.0;
    scalar_t sum = 0.0;

    for (scalar_t val : norms) {
        s.min_norm = std::min(s.min_norm, val);
        s.max_norm = std::max(s.max_norm, val);
        sum += val;
    }
    s.mean_norm = sum / norms.size();

    scalar_t var = 0.0;
    for (scalar_t val : norms) {
        scalar_t diff = val - s.mean_norm;
        var += diff * diff;
    }
    s.stddev_norm = std::sqrt(var / norms.size());
    return s;
}

ScalingDiagnostics Equilibrator::compute_diagnostics(const model::LinearProgram& lp) {
    ScalingDiagnostics diag;
    index_t m = lp.num_rows();
    index_t n = lp.num_cols();

    if (m == 0 || n == 0) return diag;

    std::vector<scalar_t> row_l1(m, 0.0), row_l2(m, 0.0), row_linf(m, 0.0);
    std::vector<scalar_t> col_l1(n, 0.0), col_l2(n, 0.0), col_linf(n, 0.0);

    diag.min_abs_coeff = 1e30;
    diag.max_abs_coeff = 0.0;

    auto process_entry = [&](index_t i, index_t j, scalar_t v) {
        scalar_t abs_v = std::abs(v);
        if (abs_v > 0.0) {
            diag.min_abs_coeff = std::min(diag.min_abs_coeff, abs_v);
            diag.max_abs_coeff = std::max(diag.max_abs_coeff, abs_v);

            row_l1[i] += abs_v;
            row_l2[i] += abs_v * abs_v;
            row_linf[i] = std::max(row_linf[i], abs_v);

            col_l1[j] += abs_v;
            col_l2[j] += abs_v * abs_v;
            col_linf[j] = std::max(col_linf[j], abs_v);
        }
    };

    if (!lp.csr_row_ptr.empty()) {
        for (index_t i = 0; i < m; ++i) {
            index_t start = lp.csr_row_ptr[i];
            index_t end = lp.csr_row_ptr[i + 1];
            for (index_t p = start; p < end; ++p) {
                process_entry(i, lp.csr_col_ind[p], lp.csr_values[p]);
            }
        }
    } else {
        for (const auto& tr : lp.A_coo.triplets()) {
            process_entry(tr.row, tr.col, tr.val);
        }
    }

    for (index_t i = 0; i < m; ++i) row_l2[i] = std::sqrt(row_l2[i]);
    for (index_t j = 0; j < n; ++j) col_l2[j] = std::sqrt(col_l2[j]);

    if (diag.min_abs_coeff == 1e30) diag.min_abs_coeff = 0.0;

    if (diag.min_abs_coeff > 0.0) {
        diag.dynamic_range = diag.max_abs_coeff / diag.min_abs_coeff;
        diag.dynamic_range_orders = std::log10(diag.dynamic_range);
    } else {
        diag.dynamic_range = 0.0;
        diag.dynamic_range_orders = 0.0;
    }

    diag.row_l1_stats = compute_norm_summary(row_l1);
    diag.row_l2_stats = compute_norm_summary(row_l2);
    diag.row_linf_stats = compute_norm_summary(row_linf);

    diag.col_l1_stats = compute_norm_summary(col_l1);
    diag.col_l2_stats = compute_norm_summary(col_l2);
    diag.col_linf_stats = compute_norm_summary(col_linf);

    diag.row_conditioning_proxy = (diag.row_l2_stats.min_norm > 1e-12)
        ? (diag.row_l2_stats.max_norm / diag.row_l2_stats.min_norm) : diag.row_l2_stats.max_norm;
    diag.col_conditioning_proxy = (diag.col_l2_stats.min_norm > 1e-12)
        ? (diag.col_l2_stats.max_norm / diag.col_l2_stats.min_norm) : diag.col_l2_stats.max_norm;
    diag.overall_conditioning_proxy = diag.row_conditioning_proxy * diag.col_conditioning_proxy;

    return diag;
}

StatusOr<ScaledModel> Equilibrator::scale(const model::LinearProgram& lp) const {
    switch (options_.method) {
        case ScalingMethod::Ruiz:
            return scale_ruiz(lp);
        case ScalingMethod::PockChambolle:
            return scale_pock_chambolle(lp);
        case ScalingMethod::L2Equilibration:
            return scale_l2(lp);
        case ScalingMethod::None: {
            ScaledModel model;
            model.lp = lp;
            index_t m = lp.num_rows();
            index_t n = lp.num_cols();
            model.row_scale_R.assign(m, 1.0);
            model.col_scale_C.assign(n, 1.0);
            model.inv_row_scale_R.assign(m, 1.0);
            model.inv_col_scale_C.assign(n, 1.0);
            model.diag_before = compute_diagnostics(lp);
            model.diag_after = model.diag_before;
            return model;
        }
    }
    return scale_ruiz(lp);
}

StatusOr<ScaledModel> Equilibrator::scale_ruiz(const model::LinearProgram& lp) const {
    utils::CPUTimer timer;
    timer.start();

    ScaledModel result;
    result.diag_before = compute_diagnostics(lp);

    index_t m = lp.num_rows();
    index_t n = lp.num_cols();
    if (m == 0 || n == 0) {
        result.lp = lp;
        return result;
    }

    // Cumulative scaling vectors
    std::vector<scalar_t> R(m, 1.0);
    std::vector<scalar_t> C(n, 1.0);

    // Extract nonzeros into a flat working array of triplets
    struct WorkingTriplet {
        index_t row;
        index_t col;
        scalar_t val;
    };
    std::vector<WorkingTriplet> triplets;

    if (!lp.csr_row_ptr.empty()) {
        for (index_t i = 0; i < m; ++i) {
            index_t start = lp.csr_row_ptr[i];
            index_t end = lp.csr_row_ptr[i + 1];
            for (index_t p = start; p < end; ++p) {
                triplets.push_back({i, lp.csr_col_ind[p], lp.csr_values[p]});
            }
        }
    } else {
        for (const auto& tr : lp.A_coo.triplets()) {
            triplets.push_back({tr.row, tr.col, tr.val});
        }
    }

    int iter = 0;
    for (; iter < options_.max_iterations; ++iter) {
        std::vector<scalar_t> row_max(m, 0.0);
        std::vector<scalar_t> col_max(n, 0.0);

        for (const auto& tr : triplets) {
            scalar_t abs_v = std::abs(tr.val);
            row_max[tr.row] = std::max(row_max[tr.row], abs_v);
            col_max[tr.col] = std::max(col_max[tr.col], abs_v);
        }

        // Convergence check: are row and column infinity norms close to 1.0?
        scalar_t max_row_dev = 0.0;
        for (index_t i = 0; i < m; ++i) {
            if (row_max[i] > 1e-12) {
                max_row_dev = std::max(max_row_dev, std::abs(1.0 - row_max[i]));
            }
        }
        scalar_t max_col_dev = 0.0;
        for (index_t j = 0; j < n; ++j) {
            if (col_max[j] > 1e-12) {
                max_col_dev = std::max(max_col_dev, std::abs(1.0 - col_max[j]));
            }
        }

        if (max_row_dev < options_.tolerance && max_col_dev < options_.tolerance && iter > 0) {
            break;
        }

        // Compute step multipliers: delta_r = sqrt(row_max), delta_c = sqrt(col_max)
        std::vector<scalar_t> dr(m, 1.0);
        for (index_t i = 0; i < m; ++i) {
            if (row_max[i] > 1e-12) dr[i] = std::sqrt(row_max[i]);
        }

        std::vector<scalar_t> dc(n, 1.0);
        for (index_t j = 0; j < n; ++j) {
            if (col_max[j] > 1e-12) dc[j] = std::sqrt(col_max[j]);
        }

        // Scale triplets
        for (auto& tr : triplets) {
            tr.val /= (dr[tr.row] * dc[tr.col]);
        }

        // Accumulate cumulative scaling
        for (index_t i = 0; i < m; ++i) R[i] /= dr[i];
        for (index_t j = 0; j < n; ++j) C[j] /= dc[j];
    }

    result.iterations_performed = iter;
    result.row_scale_R = R;
    result.col_scale_C = C;
    result.inv_row_scale_R.resize(m);
    result.inv_col_scale_C.resize(n);
    for (index_t i = 0; i < m; ++i) result.inv_row_scale_R[i] = 1.0 / R[i];
    for (index_t j = 0; j < n; ++j) result.inv_col_scale_C[j] = 1.0 / C[j];

    // Build scaled LinearProgram
    model::LinearProgram scaled = lp;
    scaled.name = lp.name + "_scaled";

    // Update bounds
    const scalar_t inf = 1e19;
    for (index_t i = 0; i < m; ++i) {
        if (scaled.row_lower[i] > -inf) scaled.row_lower[i] *= R[i];
        if (scaled.row_upper[i] < inf) scaled.row_upper[i] *= R[i];
    }
    for (index_t j = 0; j < n; ++j) {
        if (scaled.col_lower[j] > -inf) scaled.col_lower[j] /= C[j];
        if (scaled.col_upper[j] < inf) scaled.col_upper[j] /= C[j];
        scaled.c[j] *= C[j];
    }

    // Assemble scaled sparse matrices
    sparse::COOMatrix coo(m, n);
    for (const auto& tr : triplets) {
        coo.add_entry(tr.row, tr.col, tr.val);
    }
    scaled.A_coo = coo;

    auto csr = coo.to_csr();
    scaled.csr_row_ptr.assign(csr.row_ptr().begin(), csr.row_ptr().end());
    scaled.csr_col_ind.assign(csr.col_ind().begin(), csr.col_ind().end());
    scaled.csr_values.assign(csr.values().begin(), csr.values().end());

    auto csc = coo.to_csc();
    scaled.csc_col_ptr.assign(csc.col_ptr().begin(), csc.col_ptr().end());
    scaled.csc_row_ind.assign(csc.row_ind().begin(), csc.row_ind().end());
    scaled.csc_values.assign(csc.values().begin(), csc.values().end());

    result.lp = std::move(scaled);
    result.diag_after = compute_diagnostics(result.lp);

    timer.stop();
    result.elapsed_ms = timer.elapsed_milliseconds();
    return result;
}

StatusOr<ScaledModel> Equilibrator::scale_pock_chambolle(const model::LinearProgram& lp) const {
    utils::CPUTimer timer;
    timer.start();

    ScaledModel result;
    result.diag_before = compute_diagnostics(lp);

    index_t m = lp.num_rows();
    index_t n = lp.num_cols();
    if (m == 0 || n == 0) {
        result.lp = lp;
        return result;
    }

    scalar_t alpha = options_.alpha;
    scalar_t p_row = 2.0 - alpha;
    scalar_t p_col = alpha;

    std::vector<scalar_t> row_sum(m, 0.0);
    std::vector<scalar_t> col_sum(n, 0.0);

    struct WorkingTriplet {
        index_t row;
        index_t col;
        scalar_t val;
    };
    std::vector<WorkingTriplet> triplets;

    if (!lp.csr_row_ptr.empty()) {
        for (index_t i = 0; i < m; ++i) {
            index_t start = lp.csr_row_ptr[i];
            index_t end = lp.csr_row_ptr[i + 1];
            for (index_t p = start; p < end; ++p) {
                scalar_t v = lp.csr_values[p];
                index_t j = lp.csr_col_ind[p];
                triplets.push_back({i, j, v});
                scalar_t abs_v = std::abs(v);
                row_sum[i] += std::pow(abs_v, p_row);
                col_sum[j] += std::pow(abs_v, p_col);
            }
        }
    } else {
        for (const auto& tr : lp.A_coo.triplets()) {
            triplets.push_back({tr.row, tr.col, tr.val});
            scalar_t abs_v = std::abs(tr.val);
            row_sum[tr.row] += std::pow(abs_v, p_row);
            col_sum[tr.col] += std::pow(abs_v, p_col);
        }
    }

    std::vector<scalar_t> R(m, 1.0);
    for (index_t i = 0; i < m; ++i) {
        if (row_sum[i] > 1e-12) R[i] = 1.0 / std::sqrt(row_sum[i]);
    }

    std::vector<scalar_t> C(n, 1.0);
    for (index_t j = 0; j < n; ++j) {
        if (col_sum[j] > 1e-12) C[j] = 1.0 / std::sqrt(col_sum[j]);
    }

    for (auto& tr : triplets) {
        tr.val *= (R[tr.row] * C[tr.col]);
    }

    result.iterations_performed = 1;
    result.row_scale_R = R;
    result.col_scale_C = C;
    result.inv_row_scale_R.resize(m);
    result.inv_col_scale_C.resize(n);
    for (index_t i = 0; i < m; ++i) result.inv_row_scale_R[i] = 1.0 / R[i];
    for (index_t j = 0; j < n; ++j) result.inv_col_scale_C[j] = 1.0 / C[j];

    model::LinearProgram scaled = lp;
    scaled.name = lp.name + "_scaled";

    const scalar_t inf = 1e19;
    for (index_t i = 0; i < m; ++i) {
        if (scaled.row_lower[i] > -inf) scaled.row_lower[i] *= R[i];
        if (scaled.row_upper[i] < inf) scaled.row_upper[i] *= R[i];
    }
    for (index_t j = 0; j < n; ++j) {
        if (scaled.col_lower[j] > -inf) scaled.col_lower[j] /= C[j];
        if (scaled.col_upper[j] < inf) scaled.col_upper[j] /= C[j];
        scaled.c[j] *= C[j];
    }

    sparse::COOMatrix coo(m, n);
    for (const auto& tr : triplets) {
        coo.add_entry(tr.row, tr.col, tr.val);
    }
    scaled.A_coo = coo;

    auto csr = coo.to_csr();
    scaled.csr_row_ptr.assign(csr.row_ptr().begin(), csr.row_ptr().end());
    scaled.csr_col_ind.assign(csr.col_ind().begin(), csr.col_ind().end());
    scaled.csr_values.assign(csr.values().begin(), csr.values().end());

    auto csc = coo.to_csc();
    scaled.csc_col_ptr.assign(csc.col_ptr().begin(), csc.col_ptr().end());
    scaled.csc_row_ind.assign(csc.row_ind().begin(), csc.row_ind().end());
    scaled.csc_values.assign(csc.values().begin(), csc.values().end());

    result.lp = std::move(scaled);
    result.diag_after = compute_diagnostics(result.lp);

    timer.stop();
    result.elapsed_ms = timer.elapsed_milliseconds();
    return result;
}

StatusOr<ScaledModel> Equilibrator::scale_l2(const model::LinearProgram& lp) const {
    // Single iteration L2 norm scaling: divide rows and cols by their L2 norm
    Equilibrator eq(options_);
    ScalingOptions opt = options_;
    opt.alpha = 1.0;
    opt.method = ScalingMethod::PockChambolle;
    eq.set_options(opt);
    return eq.scale(lp);
}

presolve::PrimalDualSolution ScaledModel::unscale_solution(
    const presolve::PrimalDualSolution& scaled_sol) const {

    presolve::PrimalDualSolution unscaled;
    index_t n = static_cast<index_t>(col_scale_C.size());
    index_t m = static_cast<index_t>(row_scale_R.size());

    unscaled.x.resize(n);
    for (index_t j = 0; j < n; ++j) {
        unscaled.x[j] = (j < static_cast<index_t>(scaled_sol.x.size()))
            ? (scaled_sol.x[j] * col_scale_C[j]) : 0.0;
    }

    unscaled.y.resize(m);
    for (index_t i = 0; i < m; ++i) {
        unscaled.y[i] = (i < static_cast<index_t>(scaled_sol.y.size()))
            ? (scaled_sol.y[i] * row_scale_R[i]) : 0.0;
    }

    unscaled.s.resize(n);
    for (index_t j = 0; j < n; ++j) {
        unscaled.s[j] = (j < static_cast<index_t>(scaled_sol.s.size()))
            ? (scaled_sol.s[j] * inv_col_scale_C[j]) : 0.0;
    }

    scalar_t obj = lp.obj_offset;
    for (index_t j = 0; j < n; ++j) {
        obj += lp.c[j] * unscaled.x[j];
    }
    unscaled.objective_value = obj;
    unscaled.is_feasible = scaled_sol.is_feasible;

    return unscaled;
}

presolve::PrimalDualSolution ScaledModel::scale_solution(
    const presolve::PrimalDualSolution& unscaled_sol) const {

    presolve::PrimalDualSolution scaled;
    index_t n = static_cast<index_t>(col_scale_C.size());
    index_t m = static_cast<index_t>(row_scale_R.size());

    scaled.x.resize(n);
    for (index_t j = 0; j < n; ++j) {
        scaled.x[j] = (j < static_cast<index_t>(unscaled_sol.x.size()))
            ? (unscaled_sol.x[j] * inv_col_scale_C[j]) : 0.0;
    }

    scaled.y.resize(m);
    for (index_t i = 0; i < m; ++i) {
        scaled.y[i] = (i < static_cast<index_t>(unscaled_sol.y.size()))
            ? (unscaled_sol.y[i] * inv_row_scale_R[i]) : 0.0;
    }

    scaled.s.resize(n);
    for (index_t j = 0; j < n; ++j) {
        scaled.s[j] = (j < static_cast<index_t>(unscaled_sol.s.size()))
            ? (unscaled_sol.s[j] * col_scale_C[j]) : 0.0;
    }

    scaled.objective_value = unscaled_sol.objective_value;
    scaled.is_feasible = unscaled_sol.is_feasible;

    return scaled;
}

model::LinearProgram ScaledModel::unscale_model() const {
    model::LinearProgram unscaled = lp;
    index_t m = static_cast<index_t>(row_scale_R.size());
    index_t n = static_cast<index_t>(col_scale_C.size());

    unscaled.name = lp.name + "_unscaled";

    const scalar_t inf = 1e19;
    for (index_t i = 0; i < m; ++i) {
        if (unscaled.row_lower[i] > -inf) unscaled.row_lower[i] /= row_scale_R[i];
        if (unscaled.row_upper[i] < inf) unscaled.row_upper[i] /= row_scale_R[i];
    }
    for (index_t j = 0; j < n; ++j) {
        if (unscaled.col_lower[j] > -inf) unscaled.col_lower[j] *= col_scale_C[j];
        if (unscaled.col_upper[j] < inf) unscaled.col_upper[j] *= col_scale_C[j];
        unscaled.c[j] /= col_scale_C[j];
    }

    sparse::COOMatrix coo(m, n);
    if (!lp.csr_row_ptr.empty()) {
        for (index_t i = 0; i < m; ++i) {
            index_t start = lp.csr_row_ptr[i];
            index_t end = lp.csr_row_ptr[i + 1];
            for (index_t p = start; p < end; ++p) {
                index_t j = lp.csr_col_ind[p];
                scalar_t v = lp.csr_values[p] / (row_scale_R[i] * col_scale_C[j]);
                coo.add_entry(i, j, v);
            }
        }
    }
    unscaled.A_coo = coo;

    auto csr = coo.to_csr();
    unscaled.csr_row_ptr.assign(csr.row_ptr().begin(), csr.row_ptr().end());
    unscaled.csr_col_ind.assign(csr.col_ind().begin(), csr.col_ind().end());
    unscaled.csr_values.assign(csr.values().begin(), csr.values().end());

    auto csc = coo.to_csc();
    unscaled.csc_col_ptr.assign(csc.col_ptr().begin(), csc.col_ptr().end());
    unscaled.csc_row_ind.assign(csc.row_ind().begin(), csc.row_ind().end());
    unscaled.csc_values.assign(csc.values().begin(), csc.values().end());

    return unscaled;
}

} // namespace pipepye::scaling
