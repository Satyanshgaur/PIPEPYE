#include <pipepye/solver/pdhg_solver.hpp>
#include <pipepye/solver/pdhg_solver_cpu.hpp>
#include <pipepye/solver/pdhg_solver_cuda.hpp>
#include <pipepye/solver/solution_verifier.hpp>
#include <pipepye/model/mps_parser.hpp>
#include <pipepye/pipeline/model_pipeline.hpp>
#include <pipepye/sparse/matrix_generator.hpp>
#include <pipepye/utils/timer.hpp>

#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <string>
#include <filesystem>
#include <sstream>

using namespace pipepye;
using namespace pipepye::solver;
using namespace pipepye::pipeline;

namespace {

std::string find_mps_file(const std::string& filename) {
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

enum class MatrixPattern {
    UniformRandom = 0,
    Banded,
    BlockDiagonal,
    Staircase,
    IrregularHub
};

inline std::string to_string(MatrixPattern p) {
    switch (p) {
        case MatrixPattern::UniformRandom: return "UniformRandom";
        case MatrixPattern::Banded: return "Banded";
        case MatrixPattern::BlockDiagonal: return "BlockDiagonal";
        case MatrixPattern::Staircase: return "Staircase";
        case MatrixPattern::IrregularHub: return "IrregularHub";
    }
    return "Unknown";
}

model::LinearProgram make_synthetic_lp(
    index_t m, index_t n,
    MatrixPattern pattern,
    uint32_t seed = 42) {

    model::LinearProgram lp;
    lp.name = "synth_" + to_string(pattern) + "_" + std::to_string(m) + "x" + std::to_string(n);

    sparse::COOMatrix coo;
    switch (pattern) {
        case MatrixPattern::Banded: {
            index_t bw = std::max(5, static_cast<int>(n / 50));
            coo = sparse::MatrixGenerator::generate_banded(m, n, bw, bw, -5.0, 5.0, seed);
            break;
        }
        case MatrixPattern::BlockDiagonal: {
            index_t num_blocks = 4;
            coo = sparse::MatrixGenerator::generate_block_diagonal(
                num_blocks, m / num_blocks, n / num_blocks, 0.05, 0.001, -5.0, 5.0, seed);
            break;
        }
        case MatrixPattern::Staircase: {
            index_t stages = 5;
            coo = sparse::MatrixGenerator::generate_staircase(
                stages, m / stages, n / (stages + 1), 0.05, -5.0, 5.0, seed);
            break;
        }
        case MatrixPattern::IrregularHub: {
            size_t target_nnz = std::max<size_t>(100, static_cast<size_t>(m * n * 0.005));
            coo = sparse::MatrixGenerator::generate_irregular(
                m, n, target_nnz, 0.05, 0.50, -5.0, 5.0, seed);
            break;
        }
        default: {
            coo = sparse::MatrixGenerator::generate_random(m, n, 0.005, -5.0, 5.0, seed);
            break;
        }
    }

    auto csr = coo.to_csr();
    index_t actual_m = csr.num_rows();
    index_t actual_n = csr.num_cols();

    lp.col_names.resize(actual_n);
    lp.row_names.resize(actual_m);
    for (index_t j = 0; j < actual_n; ++j) lp.col_names[j] = "x" + std::to_string(j);
    for (index_t i = 0; i < actual_m; ++i) lp.row_names[i] = "c" + std::to_string(i);

    lp.c.assign(actual_n, 1.0);
    lp.col_lower.assign(actual_n, 0.0);
    lp.col_upper.assign(actual_n, 10.0);
    lp.row_lower.assign(actual_m, -1e15);
    lp.row_upper.assign(actual_m, 0.0);

    lp.csr_row_ptr.assign(csr.row_ptr().begin(), csr.row_ptr().end());
    lp.csr_col_ind.assign(csr.col_ind().begin(), csr.col_ind().end());
    lp.csr_values.assign(csr.values().begin(), csr.values().end());

    auto csc = csr.to_csc();
    lp.csc_col_ptr.assign(csc.col_ptr().begin(), csc.col_ptr().end());
    lp.csc_row_ind.assign(csc.row_ind().begin(), csc.row_ind().end());
    lp.csc_values.assign(csc.values().begin(), csc.values().end());

    // Generate strictly feasible interior point x_int = 1.0 => b = A * x_int + 2.0
    for (index_t i = 0; i < actual_m; ++i) {
        scalar_t ax = 0.0;
        index_t start = lp.csr_row_ptr[i];
        index_t end = lp.csr_row_ptr[i + 1];
        for (index_t p = start; p < end; ++p) {
            ax += lp.csr_values[p] * 1.0;
        }
        lp.row_upper[i] = ax + 2.0; // Feasible slack
    }

    // Set non-trivial objective
    for (index_t j = 0; j < actual_n; ++j) {
        lp.c[j] = (j % 2 == 0) ? 1.5 : -0.5;
    }

    return lp;
}

struct BenchmarkRecord {
    std::string experiment;
    std::string model;
    std::string backend;
    std::string pipeline_mode;
    std::string step_size_strategy;
    std::string restart_strategy;
    index_t m{0};
    index_t n{0};
    size_t nnz{0};
    int iterations{0};
    std::string status;
    scalar_t primal_objective{0.0};
    scalar_t primal_residual{0.0};
    scalar_t dual_residual{0.0};
    double h2d_ms{0.0};
    double pure_solve_ms{0.0};
    double d2h_ms{0.0};
    double prep_ms{0.0};
    double postsolve_ms{0.0};
    double verification_ms{0.0};
    double total_time_ms{0.0};
    bool verified{false};
};

std::vector<BenchmarkRecord> g_records;

void print_separator() {
    std::cout << std::string(110, '-') << "\n";
}

void print_header(const std::string& title) {
    std::cout << "\n";
    print_separator();
    std::cout << "  " << title << "\n";
    print_separator();
}

} // namespace

int main() {
    std::cout << "================================================================================\n";
    std::cout << "        PIPEPYE PHASE 3: PDHG LP SOLVER BENCHMARK HARNESS                      \n";
    std::cout << "   Chambolle-Pock First-Order Solver Evaluation (CPU vs CUDA resident state)    \n";
    std::cout << "================================================================================\n";

    init_cuda_solver();

    std::vector<std::string> netlib_names = {"afiro.mps", "adlittle.mps", "blend.mps", "beaconfd.mps"};

    // -------------------------------------------------------------------------
    // EXPERIMENT A: CPU Baseline (Raw vs Prepared LP)
    // -------------------------------------------------------------------------
    print_header("EXPERIMENT A: CPU Baseline (Raw vs Prepared LP Pipeline)");
    std::cout << std::left << std::setw(16) << "Model"
              << std::setw(14) << "Pipeline"
              << std::setw(10) << "Iters"
              << std::setw(12) << "Solve(ms)"
              << std::setw(12) << "Total(ms)"
              << std::setw(14) << "Primal Obj"
              << std::setw(12) << "P-Res"
              << std::setw(10) << "Status"
              << "\n";
    print_separator();

    for (const auto& name : netlib_names) {
        std::string filepath = find_mps_file(name);
        if (filepath.empty()) continue;

        model::LinearProgram lp;
        if (!model::MPSParser::parse_file(filepath, lp).is_ok()) continue;

        // 1. Raw
        {
            SolverConfig cfg = SolverConfig::AdaptiveRestartCPU();
            cfg.primal_tol = 1e-4;
            cfg.dual_tol = 1e-4;
            cfg.max_iterations = 10000;

            PipelineConfig pcfg;
            pcfg.mode = PipelineMode::RAW;

            auto [res, ver] = PDHGSolver::solve_end_to_end(lp, pcfg, cfg, 0.05);

            BenchmarkRecord rec;
            rec.experiment = "ExpA_RawVsPrepared";
            rec.model = name;
            rec.backend = "CPU";
            rec.pipeline_mode = "RAW";
            rec.step_size_strategy = to_string(cfg.step_size_strategy);
            rec.restart_strategy = to_string(cfg.restart_strategy);
            rec.m = lp.num_rows(); rec.n = lp.num_cols(); rec.nnz = lp.num_nonzeros();
            rec.iterations = res.iterations;
            rec.status = to_string(res.status);
            rec.primal_objective = res.primal_objective;
            rec.primal_residual = res.primal_residual;
            rec.dual_residual = res.dual_residual;
            rec.pure_solve_ms = res.timing.pure_solve_ms;
            rec.total_time_ms = res.timing.total_time_ms;
            rec.verified = ver.is_valid();
            g_records.push_back(rec);

            std::cout << std::left << std::setw(16) << name
                      << std::setw(14) << "RAW"
                      << std::setw(10) << res.iterations
                      << std::fixed << std::setprecision(2)
                      << std::setw(12) << res.timing.pure_solve_ms
                      << std::setw(12) << res.timing.total_time_ms
                      << std::setprecision(4)
                      << std::setw(14) << res.primal_objective
                      << std::scientific << std::setprecision(2)
                      << std::setw(12) << res.primal_residual
                      << std::setw(10) << to_string(res.status)
                      << "\n";
        }

        // 2. Presolve and Scaling
        {
            SolverConfig cfg = SolverConfig::AdaptiveRestartCPU();
            cfg.primal_tol = 1e-4;
            cfg.dual_tol = 1e-4;
            cfg.max_iterations = 10000;

            PipelineConfig pcfg = PipelineConfig::PresolveAndScaling();

            auto [res, ver] = PDHGSolver::solve_end_to_end(lp, pcfg, cfg, 0.05);

            BenchmarkRecord rec;
            rec.experiment = "ExpA_RawVsPrepared";
            rec.model = name;
            rec.backend = "CPU";
            rec.pipeline_mode = "PRESOLVE_AND_SCALING";
            rec.step_size_strategy = to_string(cfg.step_size_strategy);
            rec.restart_strategy = to_string(cfg.restart_strategy);
            rec.m = lp.num_rows(); rec.n = lp.num_cols(); rec.nnz = lp.num_nonzeros();
            rec.iterations = res.iterations;
            rec.status = to_string(res.status);
            rec.primal_objective = res.primal_objective;
            rec.primal_residual = res.primal_residual;
            rec.dual_residual = res.dual_residual;
            rec.prep_ms = res.timing.prep_time_ms;
            rec.pure_solve_ms = res.timing.pure_solve_ms;
            rec.postsolve_ms = res.timing.postsolve_ms;
            rec.verification_ms = res.timing.verification_ms;
            rec.total_time_ms = res.timing.total_time_ms;
            rec.verified = ver.is_valid();
            g_records.push_back(rec);

            std::cout << std::left << std::setw(16) << name
                      << std::setw(14) << "PREPARED"
                      << std::setw(10) << res.iterations
                      << std::fixed << std::setprecision(2)
                      << std::setw(12) << res.timing.pure_solve_ms
                      << std::setw(12) << res.timing.total_time_ms
                      << std::setprecision(4)
                      << std::setw(14) << res.primal_objective
                      << std::scientific << std::setprecision(2)
                      << std::setw(12) << res.primal_residual
                      << std::setw(10) << to_string(res.status)
                      << "\n";
        }
    }

    // -------------------------------------------------------------------------
    // EXPERIMENT B: Step-Size Strategy Ablation
    // -------------------------------------------------------------------------
    print_header("EXPERIMENT B: Step-Size Strategy Ablation");
    std::cout << std::left << std::setw(16) << "Model"
              << std::setw(18) << "Step Strategy"
              << std::setw(10) << "Iters"
              << std::setw(12) << "Solve(ms)"
              << std::setw(14) << "Primal Obj"
              << std::setw(12) << "P-Res"
              << std::setw(12) << "D-Res"
              << std::setw(10) << "Status"
              << "\n";
    print_separator();

    auto synth_b = make_synthetic_lp(1000, 1000, MatrixPattern::UniformRandom, 101);
    std::vector<StepSizeStrategy> step_strategies = {
        StepSizeStrategy::CONSTANT,
        StepSizeStrategy::POCK_CHAMBOLLE,
        StepSizeStrategy::ADAPTIVE
    };

    for (auto strat : step_strategies) {
        SolverConfig cfg = SolverConfig::BaselineCPU();
        cfg.step_size_strategy = strat;
        cfg.primal_tol = 1e-4;
        cfg.dual_tol = 1e-4;
        cfg.max_iterations = 10000;

        CPUPDPOptimizer opt(cfg);
        SolverResult res = opt.solve(synth_b);

        BenchmarkRecord rec;
        rec.experiment = "ExpB_StepSizeAblation";
        rec.model = "synth_1000x1000";
        rec.backend = "CPU";
        rec.pipeline_mode = "RAW";
        rec.step_size_strategy = to_string(strat);
        rec.restart_strategy = to_string(cfg.restart_strategy);
        rec.m = synth_b.num_rows(); rec.n = synth_b.num_cols(); rec.nnz = synth_b.num_nonzeros();
        rec.iterations = res.iterations;
        rec.status = to_string(res.status);
        rec.primal_objective = res.primal_objective;
        rec.primal_residual = res.primal_residual;
        rec.dual_residual = res.dual_residual;
        rec.pure_solve_ms = res.timing.pure_solve_ms;
        rec.total_time_ms = res.timing.total_time_ms;
        g_records.push_back(rec);

        std::cout << std::left << std::setw(16) << "synth_1000x1000"
                  << std::setw(18) << to_string(strat)
                  << std::setw(10) << res.iterations
                  << std::fixed << std::setprecision(2)
                  << std::setw(12) << res.timing.pure_solve_ms
                  << std::setprecision(4)
                  << std::setw(14) << res.primal_objective
                  << std::scientific << std::setprecision(2)
                  << std::setw(12) << res.primal_residual
                  << std::setw(12) << res.dual_residual
                  << std::setw(10) << to_string(res.status)
                  << "\n";
    }

    // -------------------------------------------------------------------------
    // EXPERIMENT C: Restart Strategy Ablation
    // -------------------------------------------------------------------------
    print_header("EXPERIMENT C: Restart Strategy Ablation");
    std::cout << std::left << std::setw(16) << "Model"
              << std::setw(18) << "Restart Strategy"
              << std::setw(10) << "Iters"
              << std::setw(12) << "Solve(ms)"
              << std::setw(14) << "Primal Obj"
              << std::setw(12) << "P-Res"
              << std::setw(12) << "D-Res"
              << std::setw(10) << "Status"
              << "\n";
    print_separator();

    std::vector<RestartStrategy> restart_strategies = {
        RestartStrategy::NONE,
        RestartStrategy::ADAPTIVE
    };

    for (auto rst : restart_strategies) {
        SolverConfig cfg = SolverConfig::BaselineCPU();
        cfg.step_size_strategy = StepSizeStrategy::ADAPTIVE;
        cfg.restart_strategy = rst;
        cfg.primal_tol = 1e-4;
        cfg.dual_tol = 1e-4;
        cfg.max_iterations = 10000;

        CPUPDPOptimizer opt(cfg);
        SolverResult res = opt.solve(synth_b);

        BenchmarkRecord rec;
        rec.experiment = "ExpC_RestartAblation";
        rec.model = "synth_1000x1000";
        rec.backend = "CPU";
        rec.pipeline_mode = "RAW";
        rec.step_size_strategy = to_string(cfg.step_size_strategy);
        rec.restart_strategy = to_string(rst);
        rec.m = synth_b.num_rows(); rec.n = synth_b.num_cols(); rec.nnz = synth_b.num_nonzeros();
        rec.iterations = res.iterations;
        rec.status = to_string(res.status);
        rec.primal_objective = res.primal_objective;
        rec.primal_residual = res.primal_residual;
        rec.dual_residual = res.dual_residual;
        rec.pure_solve_ms = res.timing.pure_solve_ms;
        rec.total_time_ms = res.timing.total_time_ms;
        g_records.push_back(rec);

        std::cout << std::left << std::setw(16) << "synth_1000x1000"
                  << std::setw(18) << to_string(rst)
                  << std::setw(10) << res.iterations
                  << std::fixed << std::setprecision(2)
                  << std::setw(12) << res.timing.pure_solve_ms
                  << std::setprecision(4)
                  << std::setw(14) << res.primal_objective
                  << std::scientific << std::setprecision(2)
                  << std::setw(12) << res.primal_residual
                  << std::setw(12) << res.dual_residual
                  << std::setw(10) << to_string(res.status)
                  << "\n";
    }

    // -------------------------------------------------------------------------
    // EXPERIMENT D: CPU vs GPU Performance & Speedup
    // -------------------------------------------------------------------------
    print_header("EXPERIMENT D: CPU vs GPU Performance & Speedup Scaling");
    std::cout << std::left << std::setw(18) << "Dimensions (m, n)"
              << std::setw(12) << "NNZ"
              << std::setw(12) << "CPU Pure(ms)"
              << std::setw(12) << "GPU Pure(ms)"
              << std::setw(12) << "GPU Tot(ms)"
              << std::setw(12) << "Pure Speedup"
              << std::setw(12) << "End2End Spd"
              << "\n";
    print_separator();

    std::vector<index_t> test_sizes = {500, 1500, 3000, 6000};

    // Warmup CUDA driver context to avoid measuring 1-time driver initialization
    {
        cudaFree(0);
        auto warmup_lp = make_synthetic_lp(100, 100, MatrixPattern::UniformRandom, 999);
        SolverConfig cfg_warm = SolverConfig::BaselineCUDA();
        cfg_warm.max_iterations = 5;
        (void)CudaPDPOptimizer(cfg_warm).solve(warmup_lp);
    }

    for (index_t sz : test_sizes) {
        auto lp = make_synthetic_lp(sz, sz, MatrixPattern::UniformRandom, 200 + sz);

        SolverConfig cfg_cpu = SolverConfig::AdaptiveRestartCPU();
        cfg_cpu.primal_tol = 1e-4;
        cfg_cpu.dual_tol = 1e-4;
        cfg_cpu.max_iterations = 2000;

        SolverConfig cfg_cuda = SolverConfig::AdaptiveRestartCUDA();
        cfg_cuda.primal_tol = 1e-4;
        cfg_cuda.dual_tol = 1e-4;
        cfg_cuda.max_iterations = 2000;

        CPUPDPOptimizer opt_cpu(cfg_cpu);
        SolverResult res_cpu = opt_cpu.solve(lp);

        CudaPDPOptimizer opt_cuda(cfg_cuda);
        SolverResult res_cuda = opt_cuda.solve(lp);

        double pure_speedup = (res_cuda.timing.pure_solve_ms > 0.0)
            ? (res_cpu.timing.pure_solve_ms / res_cuda.timing.pure_solve_ms) : 1.0;
        double total_speedup = (res_cuda.timing.total_time_ms > 0.0)
            ? (res_cpu.timing.total_time_ms / res_cuda.timing.total_time_ms) : 1.0;

        BenchmarkRecord rec_cpu;
        rec_cpu.experiment = "ExpD_CPU_vs_GPU";
        rec_cpu.model = "synth_" + std::to_string(sz);
        rec_cpu.backend = "CPU";
        rec_cpu.m = sz; rec_cpu.n = sz; rec_cpu.nnz = lp.num_nonzeros();
        rec_cpu.iterations = res_cpu.iterations;
        rec_cpu.pure_solve_ms = res_cpu.timing.pure_solve_ms;
        rec_cpu.total_time_ms = res_cpu.timing.total_time_ms;
        g_records.push_back(rec_cpu);

        BenchmarkRecord rec_cuda;
        rec_cuda.experiment = "ExpD_CPU_vs_GPU";
        rec_cuda.model = "synth_" + std::to_string(sz);
        rec_cuda.backend = "CUDA";
        rec_cuda.m = sz; rec_cuda.n = sz; rec_cuda.nnz = lp.num_nonzeros();
        rec_cuda.iterations = res_cuda.iterations;
        rec_cuda.h2d_ms = res_cuda.timing.h2d_transfer_ms;
        rec_cuda.pure_solve_ms = res_cuda.timing.pure_solve_ms;
        rec_cuda.d2h_ms = res_cuda.timing.d2h_transfer_ms;
        rec_cuda.total_time_ms = res_cuda.timing.total_time_ms;
        g_records.push_back(rec_cuda);

        std::string dim_str = std::to_string(sz) + "x" + std::to_string(sz);
        std::cout << std::left << std::setw(18) << dim_str
                  << std::setw(12) << lp.num_nonzeros()
                  << std::fixed << std::setprecision(2)
                  << std::setw(12) << res_cpu.timing.pure_solve_ms
                  << std::setw(12) << res_cuda.timing.pure_solve_ms
                  << std::setw(12) << res_cuda.timing.total_time_ms
                  << std::setprecision(2)
                  << std::setw(12) << (std::to_string(pure_speedup).substr(0, 4) + "x")
                  << std::setw(12) << (std::to_string(total_speedup).substr(0, 4) + "x")
                  << "\n";
    }

    // -------------------------------------------------------------------------
    // EXPERIMENT E: Transfer Sensitivity & Timing Decomposition
    // -------------------------------------------------------------------------
    print_header("EXPERIMENT E: CUDA Resident State Timing Decomposition");
    std::cout << std::left << std::setw(16) << "Size"
              << std::setw(12) << "H2D (ms)"
              << std::setw(14) << "Pure Solve(ms)"
              << std::setw(12) << "D2H (ms)"
              << std::setw(12) << "Total(ms)"
              << std::setw(14) << "Transfer Ovhd%"
              << "\n";
    print_separator();

    for (index_t sz : test_sizes) {
        auto lp = make_synthetic_lp(sz, sz, MatrixPattern::UniformRandom, 300 + sz);

        SolverConfig cfg = SolverConfig::AdaptiveRestartCUDA();
        cfg.primal_tol = 1e-4;
        cfg.dual_tol = 1e-4;
        cfg.max_iterations = 2000;

        CudaPDPOptimizer opt(cfg);
        SolverResult res = opt.solve(lp);

        double transfer_time = res.timing.h2d_transfer_ms + res.timing.d2h_transfer_ms;
        double pct_transfer = (res.timing.total_time_ms > 0.0) ? (100.0 * transfer_time / res.timing.total_time_ms) : 0.0;

        BenchmarkRecord rec;
        rec.experiment = "ExpE_TransferSensitivity";
        rec.model = "synth_" + std::to_string(sz);
        rec.backend = "CUDA";
        rec.m = sz; rec.n = sz; rec.nnz = lp.num_nonzeros();
        rec.iterations = res.iterations;
        rec.h2d_ms = res.timing.h2d_transfer_ms;
        rec.pure_solve_ms = res.timing.pure_solve_ms;
        rec.d2h_ms = res.timing.d2h_transfer_ms;
        rec.total_time_ms = res.timing.total_time_ms;
        g_records.push_back(rec);

        std::string dim_str = std::to_string(sz) + "x" + std::to_string(sz);
        std::cout << std::left << std::setw(16) << dim_str
                  << std::fixed << std::setprecision(3)
                  << std::setw(12) << res.timing.h2d_transfer_ms
                  << std::setprecision(2)
                  << std::setw(14) << res.timing.pure_solve_ms
                  << std::setprecision(3)
                  << std::setw(12) << res.timing.d2h_transfer_ms
                  << std::setprecision(2)
                  << std::setw(12) << res.timing.total_time_ms
                  << std::setprecision(2)
                  << std::setw(14) << (std::to_string(pct_transfer).substr(0, 4) + "%")
                  << "\n";
    }

    // -------------------------------------------------------------------------
    // EXPERIMENT F: Problem Structure Sensitivity
    // -------------------------------------------------------------------------
    print_header("EXPERIMENT F: Problem Structure Sensitivity (1500x1500, CUDA)");
    std::cout << std::left << std::setw(20) << "Matrix Structure"
              << std::setw(10) << "Iters"
              << std::setw(12) << "Pure(ms)"
              << std::setw(14) << "Primal Obj"
              << std::setw(12) << "P-Res"
              << std::setw(12) << "D-Res"
              << std::setw(10) << "Status"
              << "\n";
    print_separator();

    std::vector<MatrixPattern> structures = {
        MatrixPattern::UniformRandom,
        MatrixPattern::Banded,
        MatrixPattern::BlockDiagonal,
        MatrixPattern::Staircase,
        MatrixPattern::IrregularHub
    };

    for (auto st : structures) {
        auto lp = make_synthetic_lp(1500, 1500, st, 400);

        SolverConfig cfg = SolverConfig::AdaptiveRestartCUDA();
        cfg.primal_tol = 1e-4;
        cfg.dual_tol = 1e-4;
        cfg.max_iterations = 5000;

        CudaPDPOptimizer opt(cfg);
        SolverResult res = opt.solve(lp);

        BenchmarkRecord rec;
        rec.experiment = "ExpF_StructureSensitivity";
        rec.model = to_string(st);
        rec.backend = "CUDA";
        rec.m = 1500; rec.n = 1500; rec.nnz = lp.num_nonzeros();
        rec.iterations = res.iterations;
        rec.status = to_string(res.status);
        rec.primal_objective = res.primal_objective;
        rec.primal_residual = res.primal_residual;
        rec.dual_residual = res.dual_residual;
        rec.pure_solve_ms = res.timing.pure_solve_ms;
        rec.total_time_ms = res.timing.total_time_ms;
        g_records.push_back(rec);

        std::cout << std::left << std::setw(20) << to_string(st)
                  << std::setw(10) << res.iterations
                  << std::fixed << std::setprecision(2)
                  << std::setw(12) << res.timing.pure_solve_ms
                  << std::setprecision(4)
                  << std::setw(14) << res.primal_objective
                  << std::scientific << std::setprecision(2)
                  << std::setw(12) << res.primal_residual
                  << std::setw(12) << res.dual_residual
                  << std::setw(10) << to_string(res.status)
                  << "\n";
    }

    // -------------------------------------------------------------------------
    // EXPERIMENT G: Accuracy Trade-off (tol = 1e-2, 1e-4, 1e-6)
    // -------------------------------------------------------------------------
    print_header("EXPERIMENT G: Accuracy vs Runtime Trade-Off (AFIRO Netlib, CUDA)");
    std::cout << std::left << std::setw(14) << "Target Tol"
              << std::setw(10) << "Iters"
              << std::setw(12) << "Solve(ms)"
              << std::setw(14) << "Primal Obj"
              << std::setw(12) << "P-Res"
              << std::setw(12) << "D-Res"
              << std::setw(10) << "Status"
              << "\n";
    print_separator();

    std::string afiro_path = find_mps_file("afiro.mps");
    if (!afiro_path.empty()) {
        model::LinearProgram afiro_lp;
        model::MPSParser::parse_file(afiro_path, afiro_lp);

        std::vector<scalar_t> tolerances = {1e-2, 1e-3, 1e-4, 1e-5, 1e-6};
        for (scalar_t tol : tolerances) {
            SolverConfig cfg = SolverConfig::AdaptiveRestartCUDA();
            cfg.primal_tol = tol;
            cfg.dual_tol = tol;
            cfg.max_iterations = 40000;

            PipelineConfig pcfg = PipelineConfig::PresolveAndScaling();
            auto [res, ver] = PDHGSolver::solve_end_to_end(afiro_lp, pcfg, cfg, 0.05);

            BenchmarkRecord rec;
            rec.experiment = "ExpG_AccuracyTradeoff";
            rec.model = "afiro";
            rec.backend = "CUDA";
            rec.m = afiro_lp.num_rows(); rec.n = afiro_lp.num_cols(); rec.nnz = afiro_lp.num_nonzeros();
            rec.iterations = res.iterations;
            rec.status = to_string(res.status);
            rec.primal_objective = res.primal_objective;
            rec.primal_residual = res.primal_residual;
            rec.dual_residual = res.dual_residual;
            rec.pure_solve_ms = res.timing.pure_solve_ms;
            rec.total_time_ms = res.timing.total_time_ms;
            rec.verified = ver.is_valid();
            g_records.push_back(rec);

            std::stringstream tol_ss;
            tol_ss << std::scientific << std::setprecision(0) << tol;

            std::cout << std::left << std::setw(14) << tol_ss.str()
                      << std::setw(10) << res.iterations
                      << std::fixed << std::setprecision(2)
                      << std::setw(12) << res.timing.pure_solve_ms
                      << std::setprecision(4)
                      << std::setw(14) << res.primal_objective
                      << std::scientific << std::setprecision(2)
                      << std::setw(12) << res.primal_residual
                      << std::setw(12) << res.dual_residual
                      << std::setw(10) << to_string(res.status)
                      << "\n";
        }
    }

    // -------------------------------------------------------------------------
    // EXPORT BENCHMARK REPORTS (CSV & JSON)
    // -------------------------------------------------------------------------
    std::filesystem::create_directories("reports");

    // CSV
    std::ofstream csv("reports/pdhg_benchmark.csv");
    csv << "experiment,model,backend,pipeline_mode,step_size_strategy,restart_strategy,"
        << "rows,cols,nnz,iterations,status,primal_objective,primal_residual,dual_residual,"
        << "h2d_ms,pure_solve_ms,d2h_ms,prep_ms,postsolve_ms,verification_ms,total_time_ms,verified\n";
    for (const auto& r : g_records) {
        csv << r.experiment << "," << r.model << "," << r.backend << "," << r.pipeline_mode << ","
            << r.step_size_strategy << "," << r.restart_strategy << ","
            << r.m << "," << r.n << "," << r.nnz << "," << r.iterations << "," << r.status << ","
            << std::setprecision(8) << r.primal_objective << ","
            << std::scientific << std::setprecision(4) << r.primal_residual << "," << r.dual_residual << ","
            << std::fixed << std::setprecision(3)
            << r.h2d_ms << "," << r.pure_solve_ms << "," << r.d2h_ms << ","
            << r.prep_ms << "," << r.postsolve_ms << "," << r.verification_ms << ","
            << r.total_time_ms << "," << (r.verified ? "true" : "false") << "\n";
    }
    csv.close();

    // JSON
    std::ofstream json("reports/pdhg_benchmark.json");
    json << "{\n  \"benchmark\": \"PipePye PDHG LP Solver Comprehensive Evaluation\",\n"
         << "  \"records\": [\n";
    for (size_t i = 0; i < g_records.size(); ++i) {
        const auto& r = g_records[i];
        json << "    {\n"
             << "      \"experiment\": \"" << r.experiment << "\",\n"
             << "      \"model\": \"" << r.model << "\",\n"
             << "      \"backend\": \"" << r.backend << "\",\n"
             << "      \"pipeline_mode\": \"" << r.pipeline_mode << "\",\n"
             << "      \"step_size_strategy\": \"" << r.step_size_strategy << "\",\n"
             << "      \"restart_strategy\": \"" << r.restart_strategy << "\",\n"
             << "      \"rows\": " << r.m << ",\n"
             << "      \"cols\": " << r.n << ",\n"
             << "      \"nnz\": " << r.nnz << ",\n"
             << "      \"iterations\": " << r.iterations << ",\n"
             << "      \"status\": \"" << r.status << "\",\n"
             << "      \"primal_objective\": " << std::setprecision(8) << r.primal_objective << ",\n"
             << "      \"primal_residual\": " << std::scientific << r.primal_residual << ",\n"
             << "      \"dual_residual\": " << std::scientific << r.dual_residual << ",\n"
             << "      \"timing\": {\n"
             << "        \"h2d_ms\": " << std::fixed << std::setprecision(3) << r.h2d_ms << ",\n"
             << "        \"pure_solve_ms\": " << r.pure_solve_ms << ",\n"
             << "        \"d2h_ms\": " << r.d2h_ms << ",\n"
             << "        \"prep_ms\": " << r.prep_ms << ",\n"
             << "        \"postsolve_ms\": " << r.postsolve_ms << ",\n"
             << "        \"verification_ms\": " << r.verification_ms << ",\n"
             << "        \"total_time_ms\": " << r.total_time_ms << "\n"
             << "      },\n"
             << "      \"verified\": " << (r.verified ? "true" : "false") << "\n"
             << "    }" << (i + 1 < g_records.size() ? "," : "") << "\n";
    }
    json << "  ]\n}\n";
    json.close();

    std::cout << "\n================================================================================\n";
    std::cout << "  Benchmark reports successfully exported to:\n";
    std::cout << "    - reports/pdhg_benchmark.csv\n";
    std::cout << "    - reports/pdhg_benchmark.json\n";
    std::cout << "================================================================================\n";

    return 0;
}
