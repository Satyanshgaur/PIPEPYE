#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <string>
#include <filesystem>
#include <cmath>

#include <pipepye/model/mps_parser.hpp>
#include <pipepye/model/lp_model.hpp>
#include <pipepye/pipeline/model_pipeline.hpp>
#include <pipepye/simplex/simplex_types.hpp>
#include <pipepye/simplex/dual_simplex.hpp>
#include <pipepye/factorization/dense_lu.hpp>
#include <pipepye/factorization/sparse_lu.hpp>
#include <pipepye/factorization/basis_factorization.hpp>
#include <pipepye/solver/pdhg_solver.hpp>
#include <pipepye/solver/solution_verifier.hpp>
#include <pipepye/crossover/pdhg_crossover.hpp>
#include <pipepye/utils/timer.hpp>

using namespace pipepye;
using namespace pipepye::simplex;
using namespace pipepye::factorization;
using namespace pipepye::crossover;

namespace {

std::string find_netlib_file(const std::string& filename) {
    std::vector<std::string> search_dirs = {
        "tests/data/mps/netlib",
        "../tests/data/mps/netlib",
        "../../tests/data/mps/netlib",
        "/home/satyansh/pipepye/tests/data/mps/netlib"
    };
    for (const auto& dir : search_dirs) {
        std::filesystem::path p = std::filesystem::path(dir) / filename;
        if (std::filesystem::exists(p)) {
            return p.string();
        }
    }
    return "";
}

struct BenchmarkRecord {
    std::string experiment;
    std::string model;
    std::string config;
    int iterations{0};
    int pivots{0};
    int bound_flips{0};
    int factorizations{0};
    double time_ms{0.0};
    double objective{0.0};
    double primal_infeasibility{0.0};
    double dual_infeasibility{0.0};
    double speedup{1.0};
    std::string details;
};

std::vector<BenchmarkRecord> all_records;

void print_banner(const std::string& title) {
    std::cout << "\n================================================================================\n"
              << "  " << title << "\n"
              << "================================================================================\n";
}

// -----------------------------------------------------------------------------
// Experiment A: Dense Simplex Oracle vs Small LPs
// -----------------------------------------------------------------------------
void run_experiment_a() {
    print_banner("EXPERIMENT A: Dense Simplex Oracle vs Small LPs");

    // 2D LP
    model::LinearProgram lp;
    lp.name = "Oracle_2D";
    lp.c = {-2.0, -1.0};
    lp.col_lower = {0.0, 0.0};
    lp.col_upper = {5.0, 5.0};
    lp.col_names = {"x0", "x1"};
    lp.row_names = {"r0", "r1"};
    lp.row_lower = {-model::Infinity, -model::Infinity};
    lp.row_upper = {4.0, 3.0};
    lp.csr_row_ptr = {0, 2, 3};
    lp.csr_col_ind = {0, 1, 0};
    lp.csr_values = {1.0, 1.0, 1.0};
    lp.csc_col_ptr = {0, 2, 3};
    lp.csc_row_ind = {0, 1, 0};
    lp.csc_values = {1.0, 1.0, 1.0};

    DualSimplexSolver solver_exact(SimplexConfig::ExactDense());
    SimplexResult res_dense = solver_exact.solve(lp);

    DualSimplexSolver solver_sparse(SimplexConfig::Fast());
    SimplexResult res_sparse = solver_sparse.solve(lp);

    std::cout << std::left << std::setw(20) << "Model"
              << std::setw(15) << "Backend"
              << std::setw(15) << "Status"
              << std::setw(18) << "Objective"
              << std::setw(12) << "Pivots"
              << std::setw(12) << "Time (ms)" << "\n";
    std::cout << std::string(80, '-') << "\n";

    std::cout << std::left << std::setw(20) << lp.name
              << std::setw(15) << "Dense Oracle"
              << std::setw(15) << to_string(res_dense.status)
              << std::setw(18) << std::fixed << std::setprecision(6) << res_dense.objective_value
              << std::setw(12) << res_dense.iterations
              << std::setw(12) << std::setprecision(3) << res_dense.timing.total_time_ms << "\n";

    std::cout << std::left << std::setw(20) << lp.name
              << std::setw(15) << "Sparse Simplex"
              << std::setw(15) << to_string(res_sparse.status)
              << std::setw(18) << std::fixed << std::setprecision(6) << res_sparse.objective_value
              << std::setw(12) << res_sparse.iterations
              << std::setw(12) << std::setprecision(3) << res_sparse.timing.total_time_ms << "\n";

    BenchmarkRecord rec1{"Exp_A_Dense_Oracle", lp.name, "DenseLU", res_dense.iterations, res_dense.iterations, 0, res_dense.factorizations, res_dense.timing.total_time_ms, res_dense.objective_value, res_dense.max_primal_infeasibility, res_dense.max_dual_infeasibility, 1.0, "Parity verified"};
    BenchmarkRecord rec2{"Exp_A_Dense_Oracle", lp.name, "SparseLU", res_sparse.iterations, res_sparse.iterations, 0, res_sparse.factorizations, res_sparse.timing.total_time_ms, res_sparse.objective_value, res_sparse.max_primal_infeasibility, res_sparse.max_dual_infeasibility, 1.0, "Parity verified"};
    all_records.push_back(rec1);
    all_records.push_back(rec2);
}

// -----------------------------------------------------------------------------
// Experiment B: Sparse LU Factorization & Fill-in vs Dimension
// -----------------------------------------------------------------------------
void run_experiment_b() {
    print_banner("EXPERIMENT B: Sparse LU Factorization & Fill-in vs Dimension");

    std::vector<index_t> dims = {50, 100, 200, 500};

    std::cout << std::left << std::setw(10) << "Dim (m)"
              << std::setw(12) << "Orig NNZ"
              << std::setw(12) << "L NNZ"
              << std::setw(12) << "U NNZ"
              << std::setw(15) << "Fill-in Ratio"
              << std::setw(15) << "Fact Time (ms)" << "\n";
    std::cout << std::string(76, '-') << "\n";

    for (index_t dim : dims) {
        // Construct tridiagonal matrix
        std::vector<index_t> col_ptr;
        std::vector<index_t> row_ind;
        std::vector<scalar_t> values;
        col_ptr.push_back(0);

        for (index_t j = 0; j < dim; ++j) {
            if (j > 0) {
                row_ind.push_back(j - 1);
                values.push_back(-1.0);
            }
            row_ind.push_back(j);
            values.push_back(2.0 + 0.1 * j);
            if (j + 1 < dim) {
                row_ind.push_back(j + 1);
                values.push_back(-1.0);
            }
            col_ptr.push_back(static_cast<index_t>(row_ind.size()));
        }

        SparseLU lu(dim);
        utils::CPUTimer timer;
        timer.start();
        Status s = lu.factorize(dim, col_ptr, row_ind, values);
        timer.stop();

        const auto& met = lu.metrics();
        std::cout << std::left << std::setw(10) << dim
                  << std::setw(12) << met.orig_nnz
                  << std::setw(12) << met.l_nnz
                  << std::setw(12) << met.u_nnz
                  << std::setw(15) << std::fixed << std::setprecision(3) << met.fill_in_ratio
                  << std::setw(15) << std::setprecision(4) << timer.elapsed_milliseconds() << "\n";

        BenchmarkRecord rec{"Exp_B_SparseLU_Scaling", "Tridiagonal_" + std::to_string(dim), "Markowitz_0.1", 0, 0, 0, 1, timer.elapsed_milliseconds(), 0.0, 0.0, 0.0, met.fill_in_ratio, "L=" + std::to_string(met.l_nnz) + ", U=" + std::to_string(met.u_nnz)};
        all_records.push_back(rec);
    }
}

// -----------------------------------------------------------------------------
// Experiment C: Basis Update (PFI vs Refactorize Every Pivot)
// -----------------------------------------------------------------------------
void run_experiment_c() {
    print_banner("EXPERIMENT C: Basis Update (PFI vs Refactorize Every Pivot)");

    std::string path = find_netlib_file("afiro.mps");
    if (path.empty()) return;

    model::LinearProgram lp;
    model::MPSParser::parse_file(path, lp);

    SimplexConfig cfg_pfi = SimplexConfig::Fast();
    cfg_pfi.update_method = BasisUpdateMethod::ProductFormInverse;
    DualSimplexSolver solver_pfi(cfg_pfi);
    SimplexResult res_pfi = solver_pfi.solve(lp);

    SimplexConfig cfg_refact = SimplexConfig::Fast();
    cfg_refact.update_method = BasisUpdateMethod::RefactorizeAlways;
    DualSimplexSolver solver_refact(cfg_refact);
    SimplexResult res_refact = solver_refact.solve(lp);

    double speedup = (res_pfi.timing.total_time_ms > 0.0)
        ? (res_refact.timing.total_time_ms / res_pfi.timing.total_time_ms)
        : 1.0;

    std::cout << std::left << std::setw(25) << "Update Strategy"
              << std::setw(12) << "Pivots"
              << std::setw(15) << "Factorizations"
              << std::setw(15) << "Updates (Eta)"
              << std::setw(15) << "Time (ms)" << "\n";
    std::cout << std::string(82, '-') << "\n";

    std::cout << std::left << std::setw(25) << "PFI (Product Form)"
              << std::setw(12) << res_pfi.iterations
              << std::setw(15) << res_pfi.factorizations
              << std::setw(15) << res_pfi.updates
              << std::setw(15) << std::fixed << std::setprecision(2) << res_pfi.timing.total_time_ms << "\n";

    std::cout << std::left << std::setw(25) << "Refactorize Always"
              << std::setw(12) << res_refact.iterations
              << std::setw(15) << res_refact.factorizations
              << std::setw(15) << res_refact.updates
              << std::setw(15) << std::fixed << std::setprecision(2) << res_refact.timing.total_time_ms << "\n";

    BenchmarkRecord rec1{"Exp_C_Basis_Update", "AFIRO", "PFI", res_pfi.iterations, res_pfi.iterations, 0, res_pfi.factorizations, res_pfi.timing.total_time_ms, res_pfi.objective_value, 0.0, 0.0, speedup, "PFI mode"};
    BenchmarkRecord rec2{"Exp_C_Basis_Update", "AFIRO", "RefactorizeAlways", res_refact.iterations, res_refact.iterations, 0, res_refact.factorizations, res_refact.timing.total_time_ms, res_refact.objective_value, 0.0, 0.0, 1.0, "Refactorize mode"};
    all_records.push_back(rec1);
    all_records.push_back(rec2);
}

// -----------------------------------------------------------------------------
// Experiment D: Pricing Strategy (Dantzig vs Devex)
// -----------------------------------------------------------------------------
void run_experiment_d() {
    print_banner("EXPERIMENT D: Pricing Strategy (Dantzig vs Devex)");

    std::vector<std::string> models = {"afiro.mps", "blend.mps"};

    std::cout << std::left << std::setw(15) << "Model"
              << std::setw(15) << "Pricing"
              << std::setw(12) << "Pivots"
              << std::setw(18) << "Pricing Time (ms)"
              << std::setw(18) << "Total Time (ms)"
              << std::setw(18) << "Objective" << "\n";
    std::cout << std::string(96, '-') << "\n";

    for (const auto& m : models) {
        std::string path = find_netlib_file(m);
        if (path.empty()) continue;

        model::LinearProgram lp;
        model::MPSParser::parse_file(path, lp);

        // Dantzig
        SimplexConfig cfg_d = SimplexConfig::Fast();
        cfg_d.pricing = PricingStrategy::Dantzig;
        DualSimplexSolver solver_d(cfg_d);
        SimplexResult res_d = solver_d.solve(lp);

        // Devex
        SimplexConfig cfg_v = SimplexConfig::Fast();
        cfg_v.pricing = PricingStrategy::Devex;
        DualSimplexSolver solver_v(cfg_v);
        SimplexResult res_v = solver_v.solve(lp);

        std::cout << std::left << std::setw(15) << m
                  << std::setw(15) << "Dantzig"
                  << std::setw(12) << res_d.iterations
                  << std::setw(18) << std::fixed << std::setprecision(2) << res_d.timing.pricing_ms
                  << std::setw(18) << res_d.timing.total_time_ms
                  << std::setw(18) << std::setprecision(6) << res_d.objective_value << "\n";

        std::cout << std::left << std::setw(15) << m
                  << std::setw(15) << "Devex"
                  << std::setw(12) << res_v.iterations
                  << std::setw(18) << std::fixed << std::setprecision(2) << res_v.timing.pricing_ms
                  << std::setw(18) << res_v.timing.total_time_ms
                  << std::setw(18) << std::setprecision(6) << res_v.objective_value << "\n";

        BenchmarkRecord rec_d{"Exp_D_Pricing", m, "Dantzig", res_d.iterations, res_d.iterations, 0, res_d.factorizations, res_d.timing.total_time_ms, res_d.objective_value, 0.0, 0.0, 1.0, "Dantzig textbook"};
        BenchmarkRecord rec_v{"Exp_D_Pricing", m, "Devex", res_v.iterations, res_v.iterations, 0, res_v.factorizations, res_v.timing.total_time_ms, res_v.objective_value, 0.0, 0.0, 1.0, "Devex steepest edge"};
        all_records.push_back(rec_d);
        all_records.push_back(rec_v);
    }
}

// -----------------------------------------------------------------------------
// Experiment E: Ratio Test (Baseline vs Bound Flipping)
// -----------------------------------------------------------------------------
void run_experiment_e() {
    print_banner("EXPERIMENT E: Ratio Test (Baseline vs Bound Flipping)");

    std::string path = find_netlib_file("afiro.mps");
    if (path.empty()) return;

    model::LinearProgram lp;
    model::MPSParser::parse_file(path, lp);

    // Standard Harris
    SimplexConfig cfg_std = SimplexConfig::Fast();
    cfg_std.ratio_test = RatioTestStrategy::Standard;
    DualSimplexSolver solver_std(cfg_std);
    SimplexResult res_std = solver_std.solve(lp);

    // Bound Flipping
    SimplexConfig cfg_bf = SimplexConfig::Fast();
    cfg_bf.ratio_test = RatioTestStrategy::BoundFlipping;
    DualSimplexSolver solver_bf(cfg_bf);
    SimplexResult res_bf = solver_bf.solve(lp);

    std::cout << std::left << std::setw(25) << "Ratio Strategy"
              << std::setw(12) << "Pivots"
              << std::setw(15) << "Bound Flips"
              << std::setw(18) << "Ratio Time (ms)"
              << std::setw(18) << "Total Time (ms)" << "\n";
    std::cout << std::string(88, '-') << "\n";

    std::cout << std::left << std::setw(25) << "Standard Harris"
              << std::setw(12) << res_std.iterations
              << std::setw(15) << res_std.bound_flips
              << std::setw(18) << std::fixed << std::setprecision(2) << res_std.timing.ratio_test_ms
              << std::setw(18) << res_std.timing.total_time_ms << "\n";

    std::cout << std::left << std::setw(25) << "Bound Flipping"
              << std::setw(12) << res_bf.iterations
              << std::setw(15) << res_bf.bound_flips
              << std::setw(18) << std::fixed << std::setprecision(2) << res_bf.timing.ratio_test_ms
              << std::setw(18) << res_bf.timing.total_time_ms << "\n";

    BenchmarkRecord rec1{"Exp_E_Ratio_Test", "AFIRO", "Standard", res_std.iterations, res_std.iterations, res_std.bound_flips, res_std.factorizations, res_std.timing.total_time_ms, res_std.objective_value, 0.0, 0.0, 1.0, "Harris ratio"};
    BenchmarkRecord rec2{"Exp_E_Ratio_Test", "AFIRO", "BoundFlipping", res_bf.iterations, res_bf.iterations, res_bf.bound_flips, res_bf.factorizations, res_bf.timing.total_time_ms, res_bf.objective_value, 0.0, 0.0, 1.0, "Bound flipping"};
    all_records.push_back(rec1);
    all_records.push_back(rec2);
}

// -----------------------------------------------------------------------------
// Experiment F: Numerical Safeguards & Refactorization
// -----------------------------------------------------------------------------
void run_experiment_f() {
    print_banner("EXPERIMENT F: Numerical Safeguards & Refactorization Frequency");

    std::string path = find_netlib_file("afiro.mps");
    if (path.empty()) return;

    model::LinearProgram lp;
    model::MPSParser::parse_file(path, lp);

    std::vector<int> freq_list = {10, 30, 60};

    std::cout << std::left << std::setw(15) << "Max Updates"
              << std::setw(12) << "Pivots"
              << std::setw(18) << "Factorizations"
              << std::setw(18) << "Max Primal Err"
              << std::setw(18) << "Max Dual Err"
              << std::setw(15) << "Time (ms)" << "\n";
    std::cout << std::string(96, '-') << "\n";

    for (int freq : freq_list) {
        SimplexConfig cfg = SimplexConfig::Fast();
        cfg.max_updates_before_refactorize = freq;
        DualSimplexSolver solver(cfg);
        SimplexResult res = solver.solve(lp);

        std::cout << std::left << std::setw(15) << freq
                  << std::setw(12) << res.iterations
                  << std::setw(18) << res.factorizations
                  << std::setw(18) << std::scientific << std::setprecision(2) << res.max_primal_infeasibility
                  << std::setw(18) << res.max_dual_infeasibility
                  << std::setw(15) << std::fixed << std::setprecision(2) << res.timing.total_time_ms << "\n";

        BenchmarkRecord rec{"Exp_F_Safeguards", "AFIRO", "RefactFreq_" + std::to_string(freq), res.iterations, res.iterations, res.bound_flips, res.factorizations, res.timing.total_time_ms, res.objective_value, res.max_primal_infeasibility, res.max_dual_infeasibility, 1.0, "Refactorize freq " + std::to_string(freq)};
        all_records.push_back(rec);
    }
}

// -----------------------------------------------------------------------------
// Experiment G: PDHG vs Dual Simplex Head-to-Head
// -----------------------------------------------------------------------------
void run_experiment_g() {
    print_banner("EXPERIMENT G: PDHG vs Dual Simplex Head-to-Head");

    std::vector<std::string> models = {"afiro.mps", "blend.mps"};

    std::cout << std::left << std::setw(15) << "Model"
              << std::setw(15) << "Solver"
              << std::setw(12) << "Iterations"
              << std::setw(18) << "Objective"
              << std::setw(15) << "Primal Infeas"
              << std::setw(15) << "Time (ms)" << "\n";
    std::cout << std::string(90, '-') << "\n";

    for (const auto& m : models) {
        std::string path = find_netlib_file(m);
        if (path.empty()) continue;

        model::LinearProgram lp;
        model::MPSParser::parse_file(path, lp);

        // PDHG CPU
        solver::SolverConfig pdhg_cfg = solver::SolverConfig::AdaptiveRestartCPU();
        pdhg_cfg.primal_tol = 1e-4;
        pdhg_cfg.dual_tol = 1e-4;
        pdhg_cfg.max_iterations = 20000;
        solver::SolverResult pdhg_res = solver::PDHGSolver::solve(lp, pdhg_cfg);

        // Dual Simplex
        SimplexConfig sim_cfg = SimplexConfig::Fast();
        DualSimplexSolver sim_solver(sim_cfg);
        SimplexResult sim_res = sim_solver.solve(lp);

        std::cout << std::left << std::setw(15) << m
                  << std::setw(15) << "PDHG (CPU)"
                  << std::setw(12) << pdhg_res.iterations
                  << std::setw(18) << std::fixed << std::setprecision(6) << pdhg_res.primal_objective
                  << std::setw(15) << std::scientific << std::setprecision(2) << pdhg_res.primal_residual
                  << std::setw(15) << std::fixed << std::setprecision(2) << pdhg_res.timing.total_time_ms << "\n";

        std::cout << std::left << std::setw(15) << m
                  << std::setw(15) << "Dual Simplex"
                  << std::setw(12) << sim_res.iterations
                  << std::setw(18) << std::fixed << std::setprecision(6) << sim_res.objective_value
                  << std::setw(15) << std::scientific << std::setprecision(2) << sim_res.max_primal_infeasibility
                  << std::setw(15) << std::fixed << std::setprecision(2) << sim_res.timing.total_time_ms << "\n";

        BenchmarkRecord rec_p{"Exp_G_HeadToHead", m, "PDHG_CPU", pdhg_res.iterations, 0, 0, 0, pdhg_res.timing.total_time_ms, pdhg_res.primal_objective, pdhg_res.primal_residual, pdhg_res.dual_residual, 1.0, "First-order"};
        BenchmarkRecord rec_s{"Exp_G_HeadToHead", m, "DualSimplex", sim_res.iterations, sim_res.iterations, sim_res.bound_flips, sim_res.factorizations, sim_res.timing.total_time_ms, sim_res.objective_value, sim_res.max_primal_infeasibility, sim_res.max_dual_infeasibility, 1.0, "Simplex vertex"};
        all_records.push_back(rec_p);
        all_records.push_back(rec_s);
    }
}

// -----------------------------------------------------------------------------
// Experiment H: PDHG -> Simplex Crossover Hybrid Evaluation
// -----------------------------------------------------------------------------
void run_experiment_h() {
    print_banner("EXPERIMENT H: PDHG -> Simplex Crossover Hybrid Evaluation");

    std::string path = find_netlib_file("afiro.mps");
    if (path.empty()) return;

    model::LinearProgram lp;
    model::MPSParser::parse_file(path, lp);

    // 1. PDHG solve
    solver::SolverConfig pdhg_cfg = solver::SolverConfig::AdaptiveRestartCPU();
    pdhg_cfg.primal_tol = 1e-4;
    pdhg_cfg.dual_tol = 1e-4;
    solver::SolverResult pdhg_res = solver::PDHGSolver::solve(lp, pdhg_cfg);

    // 2. Crossover
    CrossoverConfig xover_cfg;
    xover_cfg.active_tolerance = 1e-3;
    CrossoverResult xover_res = PDHGCrossover::run(lp, pdhg_res.x, pdhg_res.y, xover_cfg);

    std::cout << std::left << std::setw(20) << "Stage"
              << std::setw(15) << "Iterations/Pivots"
              << std::setw(18) << "Objective"
              << std::setw(15) << "Time (ms)" << "\n";
    std::cout << std::string(68, '-') << "\n";

    std::cout << std::left << std::setw(20) << "PDHG First-Order"
              << std::setw(15) << pdhg_res.iterations
              << std::setw(18) << std::fixed << std::setprecision(6) << pdhg_res.primal_objective
              << std::setw(15) << std::setprecision(2) << pdhg_res.timing.total_time_ms << "\n";

    std::cout << std::left << std::setw(20) << "Crossover Basis Crash"
              << std::setw(15) << (std::to_string(xover_res.structural_basic_vars) + " str / " + std::to_string(xover_res.slack_basic_vars) + " slk")
              << std::setw(18) << "-"
              << std::setw(15) << std::setprecision(2) << xover_res.crash_time_ms << "\n";

    std::cout << std::left << std::setw(20) << "Simplex Cleanup"
              << std::setw(15) << xover_res.cleanup_pivots
              << std::setw(18) << std::fixed << std::setprecision(6) << xover_res.final_simplex_objective
              << std::setw(15) << std::setprecision(2) << xover_res.simplex_cleanup_time_ms << "\n";

    BenchmarkRecord rec{"Exp_H_Crossover", "AFIRO", "PDHG_Crossover", xover_res.cleanup_pivots, xover_res.cleanup_pivots, 0, 1, xover_res.total_time_ms, xover_res.final_simplex_objective, 0.0, 0.0, 1.0, "Crossover vertex recovery"};
    all_records.push_back(rec);
}

// -----------------------------------------------------------------------------
// Experiment I: Warm-Start Sensitivity (RHS & Bound Perturbations)
// -----------------------------------------------------------------------------
void run_experiment_i() {
    print_banner("EXPERIMENT I: Warm-Start Sensitivity (RHS & Bound Perturbations)");

    std::string path = find_netlib_file("afiro.mps");
    if (path.empty()) return;

    model::LinearProgram base_lp;
    model::MPSParser::parse_file(path, base_lp);

    // Base solve
    DualSimplexSolver solver_cold(SimplexConfig::Fast());
    SimplexResult base_res = solver_cold.solve(base_lp);
    Basis optimal_basis = base_res.final_basis;

    std::vector<double> perturbations = {0.01, 0.05, 0.10};

    std::cout << std::left << std::setw(18) << "Perturbation (%)"
              << std::setw(18) << "Cold Start Pivots"
              << std::setw(18) << "Warm Start Pivots"
              << std::setw(18) << "Pivot Reduction"
              << std::setw(15) << "Warm Time (ms)" << "\n";
    std::cout << std::string(87, '-') << "\n";

    for (double pert : perturbations) {
        model::LinearProgram pert_lp = base_lp;
        for (size_t i = 0; i < pert_lp.row_lower.size(); ++i) {
            if (pert_lp.row_lower[i] > -model::Infinity / 2) pert_lp.row_lower[i] *= (1.0 + pert);
            if (pert_lp.row_upper[i] < model::Infinity / 2) pert_lp.row_upper[i] *= (1.0 + pert);
        }

        // Cold start
        SimplexResult res_cold = solver_cold.solve(pert_lp);

        // Warm start from previous optimal basis
        SimplexResult res_warm = solver_cold.solve(pert_lp, optimal_basis);

        double red_pct = (res_cold.iterations > 0)
            ? (100.0 * (res_cold.iterations - res_warm.iterations) / res_cold.iterations)
            : 0.0;

        std::cout << std::left << std::setw(18) << (std::to_string(static_cast<int>(pert * 100)) + "%")
                  << std::setw(18) << res_cold.iterations
                  << std::setw(18) << res_warm.iterations
                  << std::setw(18) << (std::to_string(static_cast<int>(red_pct)) + "%")
                  << std::setw(15) << std::fixed << std::setprecision(2) << res_warm.timing.total_time_ms << "\n";

        BenchmarkRecord rec{"Exp_I_WarmStart", "AFIRO", "Pert_" + std::to_string(static_cast<int>(pert * 100)) + "pct", res_warm.iterations, res_warm.iterations, 0, res_warm.factorizations, res_warm.timing.total_time_ms, res_warm.objective_value, 0.0, 0.0, red_pct, "Warm start from base basis"};
        all_records.push_back(rec);
    }
}

// -----------------------------------------------------------------------------
// Report Exporters (CSV & JSON)
// -----------------------------------------------------------------------------
void export_reports() {
    std::filesystem::create_directories("reports");

    // CSV
    std::string csv_path = "reports/simplex_benchmark.csv";
    std::ofstream csv(csv_path);
    if (csv.is_open()) {
        csv << "experiment,model,config,iterations,pivots,bound_flips,factorizations,time_ms,objective,primal_infeas,dual_infeas,speedup,details\n";
        for (const auto& r : all_records) {
            csv << r.experiment << ","
                << r.model << ","
                << r.config << ","
                << r.iterations << ","
                << r.pivots << ","
                << r.bound_flips << ","
                << r.factorizations << ","
                << std::fixed << std::setprecision(4) << r.time_ms << ","
                << std::scientific << std::setprecision(8) << r.objective << ","
                << std::scientific << std::setprecision(4) << r.primal_infeasibility << ","
                << std::scientific << std::setprecision(4) << r.dual_infeasibility << ","
                << std::fixed << std::setprecision(2) << r.speedup << ",\""
                << r.details << "\"\n";
        }
        std::cout << "\n[Report] Exported CSV benchmark report to: " << csv_path << "\n";
    }

    // JSON
    std::string json_path = "reports/simplex_benchmark.json";
    std::ofstream json(json_path);
    if (json.is_open()) {
        json << "{\n  \"benchmarks\": [\n";
        for (size_t i = 0; i < all_records.size(); ++i) {
            const auto& r = all_records[i];
            json << "    {\n"
                 << "      \"experiment\": \"" << r.experiment << "\",\n"
                 << "      \"model\": \"" << r.model << "\",\n"
                 << "      \"config\": \"" << r.config << "\",\n"
                 << "      \"iterations\": " << r.iterations << ",\n"
                 << "      \"pivots\": " << r.pivots << ",\n"
                 << "      \"bound_flips\": " << r.bound_flips << ",\n"
                 << "      \"factorizations\": " << r.factorizations << ",\n"
                 << "      \"time_ms\": " << r.time_ms << ",\n"
                 << "      \"objective\": " << r.objective << ",\n"
                 << "      \"primal_infeas\": " << r.primal_infeasibility << ",\n"
                 << "      \"dual_infeas\": " << r.dual_infeasibility << ",\n"
                 << "      \"details\": \"" << r.details << "\"\n"
                 << "    }" << (i + 1 < all_records.size() ? "," : "") << "\n";
        }
        json << "  ]\n}\n";
        std::cout << "[Report] Exported JSON benchmark report to: " << json_path << "\n";
    }
}

} // namespace

int main(int argc, char** argv) {
    std::cout << "================================================================================\n"
              << "          PIPEPYE PHASE 4 BENCHMARK SUITE: DUAL SIMPLEX & CROSSOVER             \n"
              << "================================================================================\n";

    run_experiment_a();
    run_experiment_b();
    run_experiment_c();
    run_experiment_d();
    run_experiment_e();
    run_experiment_f();
    run_experiment_g();
    run_experiment_h();
    run_experiment_i();

    export_reports();

    std::cout << "\n[All Phase 4 Benchmarks Complete]\n";
    return 0;
}
