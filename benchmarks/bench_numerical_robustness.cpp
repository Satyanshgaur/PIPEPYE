#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <filesystem>
#include <pipepye/model/mps_parser.hpp>
#include <pipepye/model/lp_model.hpp>
#include <pipepye/pipeline/model_pipeline.hpp>
#include <pipepye/sparse/matrix_generator.hpp>
#include <pipepye/sparse/cpu_ops.hpp>
#include <pipepye/utils/timer.hpp>

using namespace pipepye;
using namespace pipepye::model;
using namespace pipepye::pipeline;
using namespace pipepye::presolve;
using namespace pipepye::scaling;
using namespace pipepye::sparse;

namespace {

struct SolverMetrics {
    int iterations{0};
    scalar_t primal_residual{0.0};
    scalar_t dual_residual{0.0};
    scalar_t objective_value{0.0};
    bool converged{false};
    bool diverged{false};
    double solver_ms{0.0};
    std::vector<scalar_t> x;
    std::vector<scalar_t> y;
};

struct AblationModeResult {
    PipelineMode mode{PipelineMode::RAW};
    SolverMetrics solver_metrics;
    double prep_ms{0.0};
    double total_ms{0.0};
    index_t final_rows{0};
    index_t final_cols{0};
    size_t final_nnz{0};
    scalar_t recovered_obj{0.0};
};

struct RobustnessComparison {
    std::string model_name;
    std::string category;
    index_t orig_rows{0};
    index_t orig_cols{0};
    size_t orig_nnz{0};

    std::vector<AblationModeResult> ablations;
};

/// @brief Lightweight first-order Primal-Dual Hybrid Gradient (PDHG / Chambolle-Pock) solver
/// for standard bound-constrained LP: min c^T x s.t. A x = b, l <= x <= u.
SolverMetrics solve_pdhg(const LinearProgram& lp, int max_iters = 1000, scalar_t tol = 1e-4) {
    SolverMetrics metrics;
    index_t m = lp.num_rows();
    index_t n = lp.num_cols();

    if (m == 0 || n == 0) {
        metrics.converged = true;
        metrics.objective_value = lp.obj_offset;
        return metrics;
    }

    utils::CPUTimer timer;
    timer.start();

    const scalar_t kInf = 1e15;

    // Prepare RHS and target equality bounds
    // For general inequalities, we project row activities onto [row_lower, row_upper]
    std::vector<scalar_t> b(m, 0.0);
    for (index_t i = 0; i < m; ++i) {
        bool has_lb = (lp.row_lower[i] > -kInf);
        bool has_ub = (lp.row_upper[i] < kInf);
        if (has_lb && has_ub) {
            b[i] = 0.5 * (lp.row_lower[i] + lp.row_upper[i]);
        } else if (has_ub) {
            b[i] = lp.row_upper[i];
        } else if (has_lb) {
            b[i] = lp.row_lower[i];
        } else {
            b[i] = 0.0;
        }
    }

    scalar_t b_norm = 0.0;
    for (scalar_t val : b) b_norm += val * val;
    b_norm = std::sqrt(b_norm) + 1.0;

    scalar_t c_norm = 0.0;
    for (scalar_t val : lp.c) c_norm += val * val;
    c_norm = std::sqrt(c_norm) + 1.0;

    // Compute Pock-Chambolle diagonal step-sizes:
    // tau_j = 1 / sum_i |A_ij|, sigma_i = 1 / sum_j |A_ij|
    std::vector<scalar_t> col_abs_sum(n, 0.0);
    std::vector<scalar_t> row_abs_sum(m, 0.0);

    for (index_t i = 0; i < m; ++i) {
        index_t start = lp.csr_row_ptr[i];
        index_t end = lp.csr_row_ptr[i + 1];
        for (index_t p = start; p < end; ++p) {
            scalar_t val = std::abs(lp.csr_values[p]);
            row_abs_sum[i] += val;
            col_abs_sum[lp.csr_col_ind[p]] += val;
        }
    }

    std::vector<scalar_t> tau(n, 1.0);
    std::vector<scalar_t> sigma(m, 1.0);
    for (index_t j = 0; j < n; ++j) {
        if (col_abs_sum[j] > 1e-12) tau[j] = 0.99 / col_abs_sum[j];
    }
    for (index_t i = 0; i < m; ++i) {
        if (row_abs_sum[i] > 1e-12) sigma[i] = 0.99 / row_abs_sum[i];
    }

    // Primal and dual iterates
    std::vector<scalar_t> x(n, 0.0);
    std::vector<scalar_t> x_prev(n, 0.0);
    std::vector<scalar_t> x_bar(n, 0.0);
    std::vector<scalar_t> y(m, 0.0);

    // Initialize x within bounds
    for (index_t j = 0; j < n; ++j) {
        scalar_t lb = lp.col_lower[j];
        scalar_t ub = lp.col_upper[j];
        if (lb > -kInf && ub < kInf) x[j] = 0.5 * (lb + ub);
        else if (lb > -kInf) x[j] = std::max(0.0, lb);
        else if (ub < kInf) x[j] = std::min(0.0, ub);
    }
    x_prev = x;
    x_bar = x;

    std::vector<scalar_t> Ax(m, 0.0);
    std::vector<scalar_t> Aty(n, 0.0);

    for (int iter = 1; iter <= max_iters; ++iter) {
        metrics.iterations = iter;

        // Dual update: y = y + sigma * (A * x_bar - b)
        // SpMV: Ax = A * x_bar
        std::fill(Ax.begin(), Ax.end(), 0.0);
        for (index_t i = 0; i < m; ++i) {
            index_t start = lp.csr_row_ptr[i];
            index_t end = lp.csr_row_ptr[i + 1];
            scalar_t sum = 0.0;
            for (index_t p = start; p < end; ++p) {
                sum += lp.csr_values[p] * x_bar[lp.csr_col_ind[p]];
            }
            Ax[i] = sum;
            y[i] += sigma[i] * (sum - b[i]);

            // Bound dual variables for inequality constraints
            if (lp.row_senses[i] == RowSense::LessEqual && y[i] < 0.0) y[i] = 0.0;
            if (lp.row_senses[i] == RowSense::GreaterEqual && y[i] > 0.0) y[i] = 0.0;
        }

        // Primal update: x^{k+1} = proj_[l, u](x^k - tau * (A^T * y + c))
        // Transposed SpMV: Aty = A^T * y
        std::fill(Aty.begin(), Aty.end(), 0.0);
        for (index_t i = 0; i < m; ++i) {
            scalar_t yi = y[i];
            if (std::abs(yi) < 1e-15) continue;
            index_t start = lp.csr_row_ptr[i];
            index_t end = lp.csr_row_ptr[i + 1];
            for (index_t p = start; p < end; ++p) {
                Aty[lp.csr_col_ind[p]] += lp.csr_values[p] * yi;
            }
        }

        for (index_t j = 0; j < n; ++j) {
            x_prev[j] = x[j];
            scalar_t x_step = x[j] - tau[j] * (Aty[j] + lp.c[j]);
            // Project onto [col_lower, col_upper]
            if (x_step < lp.col_lower[j]) x_step = lp.col_lower[j];
            if (x_step > lp.col_upper[j]) x_step = lp.col_upper[j];
            x[j] = x_step;
            // Extrapolation: x_bar = 2 * x^{k+1} - x^k
            x_bar[j] = 2.0 * x[j] - x_prev[j];
        }

        // Convergence / Stagnation / Divergence Check every 10 iterations
        if (iter % 10 == 0 || iter == max_iters) {
            scalar_t p_res = 0.0;
            for (index_t i = 0; i < m; ++i) {
                scalar_t row_act = 0.0;
                index_t start = lp.csr_row_ptr[i];
                index_t end = lp.csr_row_ptr[i + 1];
                for (index_t p = start; p < end; ++p) {
                    row_act += lp.csr_values[p] * x[lp.csr_col_ind[p]];
                }
                scalar_t viol = 0.0;
                if (lp.row_lower[i] > -kInf && row_act < lp.row_lower[i]) viol = lp.row_lower[i] - row_act;
                if (lp.row_upper[i] < kInf && row_act > lp.row_upper[i]) viol = row_act - lp.row_upper[i];
                p_res += viol * viol;
            }
            p_res = std::sqrt(p_res) / b_norm;

            scalar_t d_res = 0.0;
            for (index_t j = 0; j < n; ++j) {
                scalar_t dual_viol = Aty[j] + lp.c[j];
                // In bound constraints, dual violations occur if not complementary
                if ((lp.col_lower[j] <= -kInf || x[j] > lp.col_lower[j] + 1e-6) && dual_viol < 0.0) d_res += dual_viol * dual_viol;
                if ((lp.col_upper[j] >= kInf || x[j] < lp.col_upper[j] - 1e-6) && dual_viol > 0.0) d_res += dual_viol * dual_viol;
            }
            d_res = std::sqrt(d_res) / c_norm;

            metrics.primal_residual = p_res;
            metrics.dual_residual = d_res;

            // Check divergence
            if (std::isnan(p_res) || std::isinf(p_res) || p_res > 1e10 ||
                std::isnan(d_res) || std::isinf(d_res) || d_res > 1e10) {
                metrics.diverged = true;
                break;
            }

            if (p_res < tol && d_res < tol) {
                metrics.converged = true;
                break;
            }
        }
    }

    timer.stop();
    metrics.solver_ms = timer.elapsed_milliseconds();

    scalar_t obj = lp.obj_offset;
    for (index_t j = 0; j < n; ++j) obj += lp.c[j] * x[j];
    metrics.objective_value = obj;
    metrics.x = std::move(x);
    metrics.y = std::move(y);

    return metrics;
}

std::string find_mps(const std::string& filename) {
    std::vector<std::string> dirs = {
        "tests/data/mps/netlib",
        "../tests/data/mps/netlib",
        "../../tests/data/mps/netlib",
        "/home/satyansh/pipepye/tests/data/mps/netlib"
    };
    for (const auto& d : dirs) {
        std::filesystem::path p = std::filesystem::path(d) / filename;
        if (std::filesystem::exists(p)) return p.string();
    }
    return "";
}

LinearProgram make_ill_conditioned_lp(index_t n = 150) {
    LinearProgram lp;
    lp.name = "ill_conditioned_1e12";
    index_t m = n;
    lp.c.assign(n, 1.0);
    lp.col_lower.assign(n, 0.0);
    lp.col_upper.assign(n, 100.0);
    lp.row_lower.assign(m, 1.0);
    lp.row_upper.assign(m, 1.0);
    lp.row_senses.assign(m, RowSense::Equality);
    lp.var_types.assign(n, VariableType::Continuous);
    for (index_t j = 0; j < n; ++j) lp.col_names.push_back("x" + std::to_string(j));
    for (index_t i = 0; i < m; ++i) lp.row_names.push_back("c" + std::to_string(i));

    COOMatrix coo(m, n);
    // Diagonals span from 1e-6 to 1e6
    for (index_t i = 0; i < m; ++i) {
        double factor = std::pow(10.0, -6.0 + 12.0 * static_cast<double>(i) / (m - 1));
        coo.add_entry(i, i, factor);
        if (i + 1 < n) {
            coo.add_entry(i, i + 1, 0.5 * factor);
        }
    }
    lp.A_coo = coo;
    auto csr = coo.to_csr();
    lp.csr_row_ptr.assign(csr.row_ptr().begin(), csr.row_ptr().end());
    lp.csr_col_ind.assign(csr.col_ind().begin(), csr.col_ind().end());
    lp.csr_values.assign(csr.values().begin(), csr.values().end());
    auto csc = coo.to_csc();
    lp.csc_col_ptr.assign(csc.col_ptr().begin(), csc.col_ptr().end());
    lp.csc_row_ind.assign(csc.row_ind().begin(), csc.row_ind().end());
    lp.csc_values.assign(csc.values().begin(), csc.values().end());
    return lp;
}

LinearProgram make_degenerate_lp(index_t n = 150) {
    LinearProgram lp;
    lp.name = "degenerate_cascaded";
    index_t m = n;
    lp.c.assign(n, 1.0);
    lp.col_lower.assign(n, 0.0);
    lp.col_upper.assign(n, 100.0);
    lp.row_lower.assign(m, 0.0);
    lp.row_upper.assign(m, 10.0);
    lp.row_senses.assign(m, RowSense::LessEqual);
    lp.var_types.assign(n, VariableType::Continuous);
    for (index_t j = 0; j < n; ++j) lp.col_names.push_back("x" + std::to_string(j));
    for (index_t i = 0; i < m; ++i) lp.row_names.push_back("c" + std::to_string(i));

    // First 20 variables are fixed:
    for (index_t j = 0; j < 20; ++j) {
        lp.col_lower[j] = 2.0;
        lp.col_upper[j] = 2.0;
    }

    COOMatrix coo(m, n);
    // Rows 0..9 are empty
    // Rows 10..19 are duplicates of row 20
    for (index_t i = 10; i < 20; ++i) {
        coo.add_entry(i, 20 % n, 3.0);
        coo.add_entry(i, (20 + 1) % n, -2.0);
    }
    // Active rows 20..m-1
    for (index_t i = 20; i < m; ++i) {
        coo.add_entry(i, i, 2.0);
        coo.add_entry(i, (i + 1) % n, -1.0);
    }
    lp.A_coo = coo;
    auto csr = coo.to_csr();
    lp.csr_row_ptr.assign(csr.row_ptr().begin(), csr.row_ptr().end());
    lp.csr_col_ind.assign(csr.col_ind().begin(), csr.col_ind().end());
    lp.csr_values.assign(csr.values().begin(), csr.values().end());
    auto csc = coo.to_csc();
    lp.csc_col_ptr.assign(csc.col_ptr().begin(), csc.col_ptr().end());
    lp.csc_row_ind.assign(csc.row_ind().begin(), csc.row_ind().end());
    lp.csc_values.assign(csc.values().begin(), csc.values().end());
    return lp;
}

LinearProgram make_hub_imbalance_lp(index_t n = 300) {
    LinearProgram lp;
    lp.name = "irregular_hub_extreme";
    index_t m = n;
    lp.c.assign(n, 1.0);
    lp.col_lower.assign(n, 0.0);
    lp.col_upper.assign(n, 50.0);
    lp.row_lower.assign(m, 0.0);
    lp.row_upper.assign(m, 50.0);
    lp.row_senses.assign(m, RowSense::LessEqual);
    lp.var_types.assign(n, VariableType::Continuous);
    for (index_t j = 0; j < n; ++j) lp.col_names.push_back("x" + std::to_string(j));
    for (index_t i = 0; i < m; ++i) lp.row_names.push_back("c" + std::to_string(i));

    size_t target_nnz = static_cast<size_t>(n * 10);
    COOMatrix coo = MatrixGenerator::generate_irregular(m, n, target_nnz, 0.02, 0.65, -10.0, 10.0, 99);
    lp.A_coo = coo;
    auto csr = coo.to_csr();
    lp.csr_row_ptr.assign(csr.row_ptr().begin(), csr.row_ptr().end());
    lp.csr_col_ind.assign(csr.col_ind().begin(), csr.col_ind().end());
    lp.csr_values.assign(csr.values().begin(), csr.values().end());
    auto csc = coo.to_csc();
    lp.csc_col_ptr.assign(csc.col_ptr().begin(), csc.col_ptr().end());
    lp.csc_row_ind.assign(csc.row_ind().begin(), csc.row_ind().end());
    lp.csc_values.assign(csc.values().begin(), csc.values().end());
    return lp;
}

RobustnessComparison evaluate_model(const std::string& category, const LinearProgram& orig_lp) {
    RobustnessComparison comp;
    comp.model_name = orig_lp.name;
    comp.category = category;
    comp.orig_rows = orig_lp.num_rows();
    comp.orig_cols = orig_lp.num_cols();
    comp.orig_nnz = orig_lp.num_nonzeros();

    std::vector<PipelineMode> modes = {
        PipelineMode::RAW,
        PipelineMode::PRESOLVE_ONLY,
        PipelineMode::SCALING_ONLY,
        PipelineMode::PRESOLVE_AND_SCALING
    };

    for (auto m : modes) {
        PipelineConfig cfg;
        cfg.set_mode(m);
        cfg.compute_characterization = true;

        utils::CPUTimer prep_timer;
        prep_timer.start();
        auto prep_res = ModelPipeline::prepare(orig_lp, cfg);
        prep_timer.stop();

        AblationModeResult mres;
        mres.mode = m;
        mres.prep_ms = prep_timer.elapsed_milliseconds();

        if (prep_res.is_ok()) {
            const auto& prepared = prep_res.value();
            mres.final_rows = prepared.lp.num_rows();
            mres.final_cols = prepared.lp.num_cols();
            mres.final_nnz = prepared.lp.num_nonzeros();

            mres.solver_metrics = solve_pdhg(prepared.lp, 500, 1e-4);
            mres.total_ms = mres.prep_ms + mres.solver_metrics.solver_ms;

            // Attempt solution recovery to verify original objective
            PrimalDualSolution sol;
            sol.x = mres.solver_metrics.x;
            sol.y = mres.solver_metrics.y;
            sol.s.assign(prepared.lp.num_cols(), 0.0);
            sol.is_feasible = mres.solver_metrics.converged;

            auto rec_res = prepared.recover_solution(sol, orig_lp);
            if (rec_res.is_ok()) {
                mres.recovered_obj = rec_res.value().objective_value;
            } else {
                mres.recovered_obj = mres.solver_metrics.objective_value;
            }
        }
        comp.ablations.push_back(std::move(mres));
    }

    return comp;
}

} // namespace

int main() {
    std::cout << "================================================================================\n";
    std::cout << "         PIPEPYE NUMERICAL ROBUSTNESS 4-WAY ABLATION EXPERIMENT SUITE           \n";
    std::cout << "================================================================================\n";

    std::vector<std::pair<std::string, LinearProgram>> suite;

    // 1. Easy baseline: afiro.mps
    std::string afiro_p = find_mps("afiro.mps");
    if (!afiro_p.empty()) {
        LinearProgram lp;
        if (MPSParser::parse_file(afiro_p, lp).is_ok()) {
            suite.push_back({"Easy_Baseline", std::move(lp)});
        }
    }

    // 2. Large sparse: beaconfd.mps
    std::string bfd_p = find_mps("beaconfd.mps");
    if (!bfd_p.empty()) {
        LinearProgram lp;
        if (MPSParser::parse_file(bfd_p, lp).is_ok()) {
            suite.push_back({"Large_Sparse", std::move(lp)});
        }
    }

    // 3. Ill-conditioned model: 12 orders of magnitude (10^-6 to 10^6)
    suite.push_back({"Ill_Conditioned", make_ill_conditioned_lp(150)});

    // 4. Degenerate model: cascaded redundancies, empty rows, fixed variables
    suite.push_back({"Degenerate", make_degenerate_lp(150)});

    // 5. Irregular hub model: extreme degree imbalance
    suite.push_back({"Irregular_Hub", make_hub_imbalance_lp(250)});

    std::vector<RobustnessComparison> results;
    for (const auto& [cat, lp] : suite) {
        std::cout << "Evaluating 4-way ablation on [" << cat << "] " << lp.name << "..." << std::flush;
        auto res = evaluate_model(cat, lp);
        results.push_back(res);
        std::cout << " Done.\n";
    }

    // Print Comprehensive Ablation Table
    std::cout << "\n====================================================================================================================================================\n";
    std::cout << "                                              4-WAY ABLATION DETAILED RESULTS                                                       \n";
    std::cout << "====================================================================================================================================================\n";
    std::cout << std::left << std::setw(20) << "Model"
              << std::setw(22) << "Pipeline Mode"
              << std::setw(14) << "Final Size"
              << std::setw(10) << "NNZ"
              << std::setw(8)  << "Iters"
              << std::setw(12) << "Status"
              << std::setw(14) << "Primal-Res"
              << std::setw(14) << "Dual-Res"
              << std::setw(16) << "Recovered Obj"
              << std::setw(12) << "Prep (ms)"
              << std::setw(12) << "Solve (ms)"
              << std::setw(12) << "Total (ms)" << "\n";
    std::cout << "----------------------------------------------------------------------------------------------------------------------------------------------------\n";

    for (const auto& r : results) {
        for (const auto& ab : r.ablations) {
            std::ostringstream sz_str, pres_str, dres_str, obj_str;
            sz_str << ab.final_rows << "x" << ab.final_cols;
            pres_str << std::scientific << std::setprecision(2) << ab.solver_metrics.primal_residual;
            dres_str << std::scientific << std::setprecision(2) << ab.solver_metrics.dual_residual;
            obj_str << std::scientific << std::setprecision(3) << ab.recovered_obj;

            std::string status = ab.solver_metrics.diverged ? "DIVERGED" : (ab.solver_metrics.converged ? "CONVERGED" : "MAX_ITER");

            std::cout << std::left << std::setw(20) << r.model_name
                      << std::setw(22) << to_string(ab.mode)
                      << std::setw(14) << sz_str.str()
                      << std::setw(10) << ab.final_nnz
                      << std::setw(8)  << ab.solver_metrics.iterations
                      << std::setw(12) << status
                      << std::setw(14) << pres_str.str()
                      << std::setw(14) << dres_str.str()
                      << std::setw(16) << obj_str.str()
                      << std::setw(12) << std::fixed << std::setprecision(2) << ab.prep_ms
                      << std::setw(12) << std::fixed << std::setprecision(2) << ab.solver_metrics.solver_ms
                      << std::setw(12) << std::fixed << std::setprecision(2) << ab.total_ms << "\n";
        }
        std::cout << "----------------------------------------------------------------------------------------------------------------------------------------------------\n";
    }
    std::cout << "====================================================================================================================================================\n";

    // Summary Statistics per Mode
    std::cout << "\n================================================================================================\n";
    std::cout << "                                ABLATION MODE SUMMARY COMPARISON                                \n";
    std::cout << "================================================================================================\n";
    std::cout << std::left << std::setw(24) << "Pipeline Mode"
              << std::setw(16) << "Convergence"
              << std::setw(16) << "Divergence"
              << std::setw(18) << "Avg Prep (ms)"
              << std::setw(18) << "Avg Solve (ms)"
              << std::setw(18) << "Avg Total (ms)" << "\n";
    std::cout << "------------------------------------------------------------------------------------------------\n";

    std::vector<PipelineMode> modes = {
        PipelineMode::RAW,
        PipelineMode::PRESOLVE_ONLY,
        PipelineMode::SCALING_ONLY,
        PipelineMode::PRESOLVE_AND_SCALING
    };

    for (auto m : modes) {
        int converged_cnt = 0;
        int diverged_cnt = 0;
        double sum_prep = 0.0;
        double sum_solve = 0.0;
        double sum_total = 0.0;
        int count = 0;

        for (const auto& r : results) {
            for (const auto& ab : r.ablations) {
                if (ab.mode == m) {
                    count++;
                    if (ab.solver_metrics.converged) converged_cnt++;
                    if (ab.solver_metrics.diverged) diverged_cnt++;
                    sum_prep += ab.prep_ms;
                    sum_solve += ab.solver_metrics.solver_ms;
                    sum_total += ab.total_ms;
                }
            }
        }

        std::ostringstream conv_str, div_str;
        conv_str << converged_cnt << "/" << count;
        div_str << diverged_cnt << "/" << count;

        std::cout << std::left << std::setw(24) << to_string(m)
                  << std::setw(16) << conv_str.str()
                  << std::setw(16) << div_str.str()
                  << std::setw(18) << std::fixed << std::setprecision(2) << (count > 0 ? sum_prep / count : 0.0)
                  << std::setw(18) << std::fixed << std::setprecision(2) << (count > 0 ? sum_solve / count : 0.0)
                  << std::setw(18) << std::fixed << std::setprecision(2) << (count > 0 ? sum_total / count : 0.0) << "\n";
    }
    std::cout << "================================================================================================\n";

    // Machine-readable exports
    std::filesystem::create_directories("reports");

    // CSV
    std::ofstream csv("reports/numerical_robustness.csv");
    if (csv.is_open()) {
        csv << "model_name,category,orig_rows,orig_cols,orig_nnz,mode,"
            << "final_rows,final_cols,final_nnz,iterations,converged,diverged,"
            << "primal_res,dual_res,recovered_obj,prep_ms,solver_ms,total_ms\n";

        for (const auto& r : results) {
            for (const auto& ab : r.ablations) {
                csv << r.model_name << "," << r.category << "," << r.orig_rows << "," << r.orig_cols << "," << r.orig_nnz << ","
                    << to_string(ab.mode) << "," << ab.final_rows << "," << ab.final_cols << "," << ab.final_nnz << ","
                    << ab.solver_metrics.iterations << "," << (ab.solver_metrics.converged ? 1 : 0) << ","
                    << (ab.solver_metrics.diverged ? 1 : 0) << "," << ab.solver_metrics.primal_residual << ","
                    << ab.solver_metrics.dual_residual << "," << ab.recovered_obj << ","
                    << ab.prep_ms << "," << ab.solver_metrics.solver_ms << "," << ab.total_ms << "\n";
            }
        }
        std::cout << "\nWrote CSV results to: reports/numerical_robustness.csv\n";
    }

    // JSON
    std::ofstream json("reports/numerical_robustness.json");
    if (json.is_open()) {
        json << "[\n";
        for (size_t i = 0; i < results.size(); ++i) {
            const auto& r = results[i];
            json << "  {\n"
                 << "    \"model_name\": \"" << r.model_name << "\",\n"
                 << "    \"category\": \"" << r.category << "\",\n"
                 << "    \"dimensions\": {\"orig_rows\": " << r.orig_rows << ", \"orig_cols\": " << r.orig_cols << ", \"orig_nnz\": " << r.orig_nnz << "},\n"
                 << "    \"ablations\": [\n";
            for (size_t j = 0; j < r.ablations.size(); ++j) {
                const auto& ab = r.ablations[j];
                json << "      {\n"
                     << "        \"mode\": \"" << to_string(ab.mode) << "\",\n"
                     << "        \"final_rows\": " << ab.final_rows << ",\n"
                     << "        \"final_cols\": " << ab.final_cols << ",\n"
                     << "        \"final_nnz\": " << ab.final_nnz << ",\n"
                     << "        \"iterations\": " << ab.solver_metrics.iterations << ",\n"
                     << "        \"converged\": " << (ab.solver_metrics.converged ? "true" : "false") << ",\n"
                     << "        \"diverged\": " << (ab.solver_metrics.diverged ? "true" : "false") << ",\n"
                     << "        \"primal_res\": " << ab.solver_metrics.primal_residual << ",\n"
                     << "        \"dual_res\": " << ab.solver_metrics.dual_residual << ",\n"
                     << "        \"recovered_obj\": " << ab.recovered_obj << ",\n"
                     << "        \"prep_ms\": " << ab.prep_ms << ",\n"
                     << "        \"solver_ms\": " << ab.solver_metrics.solver_ms << ",\n"
                     << "        \"total_ms\": " << ab.total_ms << "\n"
                     << "      }" << (j + 1 < r.ablations.size() ? ",\n" : "\n");
            }
            json << "    ]\n"
                 << "  }" << (i + 1 < results.size() ? ",\n" : "\n");
        }
        json << "]\n";
        std::cout << "Wrote JSON results to: reports/numerical_robustness.json\n";
    }

    return 0;
}
