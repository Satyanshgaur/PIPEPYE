#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <iomanip>
#include <filesystem>
#include <cmath>
#include <sstream>
#include <algorithm>

#include <pipepye/model/mps_parser.hpp>
#include <pipepye/model/lp_model.hpp>
#include <pipepye/pipeline/model_pipeline.hpp>
#include <pipepye/analysis/problem_analyzer.hpp>
#include <pipepye/analysis/solver_selector.hpp>
#include <pipepye/solver/pdhg_solver.hpp>
#include <pipepye/solver/solution_verifier.hpp>
#include <pipepye/simplex/dual_simplex.hpp>
#include <pipepye/crossover/pdhg_crossover.hpp>
#include <pipepye/milp/branch_and_bound.hpp>
#include <pipepye/utils/timer.hpp>

#if defined(PIPEPYE_CUDA_ENABLED) || defined(PIPEPYE_HAS_CUDA)
#include <pipepye/solver/pdhg_solver_cuda.hpp>
#endif

namespace fs = std::filesystem;
using namespace pipepye;
using namespace pipepye::model;
using namespace pipepye::pipeline;
using namespace pipepye::analysis;
using namespace pipepye::solver;
using namespace pipepye::simplex;
using namespace pipepye::crossover;
using namespace pipepye::milp;

namespace {

std::string escape_json(const std::string& s) {
    std::string out;
    for (char c : s) {
        if (c == '"') out += "\\\"";
        else if (c == '\\') out += "\\\\";
        else if (c == '\n') out += "\\n";
        else if (c == '\r') out += "\\r";
        else if (c == '\t') out += "\\t";
        else out += c;
    }
    return out;
}

std::string format_double(double val) {
    if (std::isnan(val)) return "null";
    if (std::isinf(val)) return val > 0 ? "1e308" : "-1e308";
    std::ostringstream ss;
    ss << std::setprecision(8) << val;
    return ss.str();
}

} // anonymous namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_mps> [--max-iters <N>] [--out <output.json>]\n";
        return 1;
    }

    std::string mps_path = argv[1];
    int max_iters = 3000;
    std::string out_json_path = "";
    std::string milp_cfg_name = "advanced";

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--max-iters" && i + 1 < argc) {
            max_iters = std::stoi(argv[++i]);
        } else if (arg == "--out" && i + 1 < argc) {
            out_json_path = argv[++i];
        } else if (arg == "--milp-config" && i + 1 < argc) {
            milp_cfg_name = argv[++i];
        }
    }

    if (!fs::exists(mps_path)) {
        std::cerr << "Error: File not found: " << mps_path << "\n";
        std::cout << "{\"error\": \"File not found: " << escape_json(mps_path) << "\"}\n";
        return 1;
    }

#if defined(PIPEPYE_CUDA_ENABLED) || defined(PIPEPYE_HAS_CUDA)
    init_cuda_solver();
#endif

    // =========================================================================
    // 1. INGESTION & PHASE 1 ANALYSIS (Raw Model)
    // =========================================================================
    utils::CPUTimer timer;
    timer.start();
    LinearProgram raw_lp;
    Status parse_status = MPSParser::parse_file(mps_path, raw_lp);
    if (!parse_status.is_ok()) {
        std::cout << "{\"error\": \"MPS Parse Error: " << escape_json(parse_status.to_string()) << "\"}\n";
        return 1;
    }
    double parse_time_ms = timer.elapsed_milliseconds();

    index_t raw_m = raw_lp.num_rows();
    index_t raw_n = raw_lp.num_cols();
    size_t raw_nnz = raw_lp.num_nonzeros();
    double raw_density = (raw_m > 0 && raw_n > 0) ? static_cast<double>(raw_nnz) / (static_cast<double>(raw_m) * raw_n) : 0.0;

    index_t count_continuous = 0;
    index_t count_binary = 0;
    index_t count_integer = 0;
    for (auto vt : raw_lp.var_types) {
        if (vt == VariableType::Binary) count_binary++;
        else if (vt == VariableType::Integer) count_integer++;
        else count_continuous++;
    }

    index_t count_le = 0, count_ge = 0, count_eq = 0, count_range = 0;
    for (auto rs : raw_lp.row_senses) {
        if (rs == RowSense::LessEqual) count_le++;
        else if (rs == RowSense::GreaterEqual) count_ge++;
        else if (rs == RowSense::Equality) count_eq++;
        else count_range++;
    }

    size_t coo_bytes = raw_nnz * (sizeof(index_t) * 2 + sizeof(scalar_t));
    size_t csr_bytes = (raw_m + 1) * sizeof(index_t) + raw_nnz * (sizeof(index_t) + sizeof(scalar_t));
    size_t csc_bytes = (raw_n + 1) * sizeof(index_t) + raw_nnz * (sizeof(index_t) + sizeof(scalar_t));

    // =========================================================================
    // 2. PHASE 2 MODEL PREPARATION (Presolve & Ruiz Scaling)
    // =========================================================================
    PipelineConfig pipe_cfg = PipelineConfig::PresolveAndScaling();
    ModelPipeline pipeline(pipe_cfg);

    timer.start();
    auto prep_res = pipeline.prepare(raw_lp);
    if (!prep_res.is_ok()) {
        std::cout << "{\"error\": \"ModelPipeline Error: " << escape_json(prep_res.status().to_string()) << "\"}\n";
        return 1;
    }
    PreparedLP prepared_lp = prep_res.value();
    double prep_time_ms = timer.elapsed_milliseconds();

    index_t prep_m = prepared_lp.lp.num_rows();
    index_t prep_n = prepared_lp.lp.num_cols();
    size_t prep_nnz = prepared_lp.lp.num_nonzeros();
    double prep_density = (prep_m > 0 && prep_n > 0) ? static_cast<double>(prep_nnz) / (static_cast<double>(prep_m) * prep_n) : 0.0;

    index_t rows_removed = raw_m - prep_m;
    index_t cols_removed = raw_n - prep_n;
    size_t nnz_removed = (raw_nnz >= prep_nnz) ? (raw_nnz - prep_nnz) : 0;
    double nnz_reduction_pct = raw_nnz > 0 ? (100.0 * nnz_removed / raw_nnz) : 0.0;

    double dyn_range_before = prepared_lp.scaling.diag_before.dynamic_range;
    double dyn_range_after = prepared_lp.scaling.diag_after.dynamic_range;

    // =========================================================================
    // 3. PHASE 5 STRUCTURAL ANALYSIS & PRE-REGISTERED PREDICTION
    // =========================================================================
    ProblemStats stats = ProblemAnalyzer::analyze(prepared_lp.lp);
    auto rec = StructureAwareSelector::recommend(raw_lp, stats);

    std::string predicted_solver = to_string(rec.solver);
    std::string predicted_backend = to_string(rec.backend);
    std::string prediction_rationale = rec.rationale;

    // =========================================================================
    // 4. SOLVER EXECUTION
    // =========================================================================
    bool is_milp = std::any_of(raw_lp.var_types.begin(), raw_lp.var_types.end(),
        [](VariableType vt) { return vt != VariableType::Continuous; });

    // Results storage
    bool cpu_pdhg_run = false;
    SolverResult cpu_pdhg_res;
    bool gpu_pdhg_run = false;
    SolverResult gpu_pdhg_res;
    bool simplex_run = false;
    SimplexResult simplex_res;
    bool crossover_run = false;
    CrossoverResult crossover_res;
    bool milp_run = false;
    MILPResult milp_warm_res;
    MILPResult milp_cold_res;

    std::string actual_winner_solver = "None";
    std::string actual_winner_backend = "None";
    double best_runtime_ms = 1e9;
    scalar_t best_objective = 0.0;

    if (is_milp) {
        // Run MILP Branch-and-Bound with Phase 7 Advanced Features
        milp_run = true;
        MILPConfig warm_cfg;
        if (milp_cfg_name == "advanced" || milp_cfg_name == "full") {
            warm_cfg = MILPConfig::Advanced();
        } else if (milp_cfg_name == "cuts") {
            warm_cfg = MILPConfig::WithCuts();
        } else if (milp_cfg_name == "heuristics") {
            warm_cfg = MILPConfig::WithHeuristics();
        } else if (milp_cfg_name == "pseudocost") {
            warm_cfg = MILPConfig::PseudoCostBranching();
        } else if (milp_cfg_name == "depth_first" || milp_cfg_name == "dfs") {
            warm_cfg = MILPConfig::DepthFirst();
        } else if (milp_cfg_name == "cold") {
            warm_cfg = MILPConfig::ColdStartBaseline();
        } else {
            warm_cfg = MILPConfig::Default();
        }
        warm_cfg.time_limit_sec = 30.0;
        warm_cfg.max_nodes = 2000;

        BranchAndBoundSolver bnb_warm(warm_cfg);
        milp_warm_res = bnb_warm.solve(raw_lp);

        MILPConfig cold_cfg = MILPConfig::ColdStartBaseline();
        cold_cfg.time_limit_sec = std::min(warm_cfg.time_limit_sec, 15.0);
        cold_cfg.max_nodes = warm_cfg.max_nodes;
        BranchAndBoundSolver bnb_cold(cold_cfg);
        milp_cold_res = bnb_cold.solve(raw_lp);

        actual_winner_solver = "BranchAndBound";
        actual_winner_backend = "CPU";
        best_runtime_ms = milp_warm_res.total_time_ms;
        best_objective = milp_warm_res.best_objective;

        // Also run continuous root relaxation via Dual Simplex
        DualSimplexSolver simplex_solver;
        simplex_res = simplex_solver.solve(prepared_lp);
        simplex_run = true;
    } else {
        // Continuous LP: Run Simplex, CPU PDHG, GPU PDHG, and Crossover

        // 1. Dual Revised Simplex
        DualSimplexSolver simplex_solver;
        simplex_res = simplex_solver.solve(prepared_lp);
        simplex_run = true;
        if (simplex_res.is_optimal()) {
            actual_winner_solver = "DualSimplex";
            actual_winner_backend = "CPU";
            best_runtime_ms = simplex_res.timing.total_time_ms;
            best_objective = simplex_res.objective_value;
        }

        // 2. CPU PDHG
        SolverConfig cpu_cfg = SolverConfig::BaselineCPU();
        cpu_cfg.max_iterations = max_iters;
        cpu_cfg.primal_tol = 1e-4;
        cpu_cfg.dual_tol = 1e-4;
        cpu_cfg.record_history = true;
        cpu_pdhg_res = PDHGSolver::solve(prepared_lp, cpu_cfg);
        cpu_pdhg_run = true;

        if (cpu_pdhg_res.is_converged() && cpu_pdhg_res.timing.total_time_ms < best_runtime_ms) {
            actual_winner_solver = "PDHG_CPU";
            actual_winner_backend = "CPU";
            best_runtime_ms = cpu_pdhg_res.timing.total_time_ms;
            best_objective = cpu_pdhg_res.primal_objective;
        }

        // 3. GPU PDHG
#if defined(PIPEPYE_CUDA_ENABLED) || defined(PIPEPYE_HAS_CUDA)
        try {
            SolverConfig cuda_cfg = SolverConfig::BaselineCUDA();
            cuda_cfg.max_iterations = max_iters;
            cuda_cfg.primal_tol = 1e-4;
            cuda_cfg.dual_tol = 1e-4;
            cuda_cfg.record_history = true;
            gpu_pdhg_res = PDHGSolver::solve(prepared_lp, cuda_cfg);
            gpu_pdhg_run = true;

            if (gpu_pdhg_res.is_converged() && gpu_pdhg_res.timing.total_time_ms < best_runtime_ms) {
                actual_winner_solver = "PDHG_GPU";
                actual_winner_backend = "CUDA";
                best_runtime_ms = gpu_pdhg_res.timing.total_time_ms;
                best_objective = gpu_pdhg_res.primal_objective;
            }
        } catch (const std::exception& e) {
            std::cerr << "GPU PDHG not executed: " << e.what() << "\n";
            gpu_pdhg_run = false;
        }
#endif

        // 4. PDHG -> Simplex Crossover
        if (cpu_pdhg_res.x.size() == static_cast<size_t>(prep_n)) {
            try {
                crossover_res = PDHGCrossover::run(prepared_lp, cpu_pdhg_res.x, cpu_pdhg_res.y);
                crossover_run = true;
            } catch (const std::exception& e) {
                std::cerr << "Crossover exception: " << e.what() << "\n";
                crossover_run = false;
            }
        }
    }

    // =========================================================================
    // 5. INDEPENDENT SOLUTION VERIFICATION
    // =========================================================================
    std::string audited_solver = "None";
    VerificationResult verif_res;

    if (is_milp && milp_warm_res.is_feasible()) {
        audited_solver = "BranchAndBound";
        verif_res = SolutionVerifier::verify(raw_lp, milp_warm_res.x, {}, milp_warm_res.best_objective);
    } else if (simplex_res.is_optimal()) {
        audited_solver = "DualSimplex";
        presolve::PrimalDualSolution s_sol;
        s_sol.x = simplex_res.x;
        s_sol.y = simplex_res.y;
        s_sol.s = simplex_res.s;
        s_sol.objective_value = simplex_res.objective_value;
        auto rec_status = prepared_lp.recover_solution(s_sol, raw_lp);
        if (rec_status.is_ok()) {
            verif_res = SolutionVerifier::verify(raw_lp, rec_status.value().x, rec_status.value().y, simplex_res.objective_value);
        }
    } else if (cpu_pdhg_res.is_converged()) {
        audited_solver = "PDHG_CPU";
        presolve::PrimalDualSolution p_sol;
        p_sol.x = cpu_pdhg_res.x;
        p_sol.y = cpu_pdhg_res.y;
        p_sol.objective_value = cpu_pdhg_res.primal_objective;
        auto rec_status = prepared_lp.recover_solution(p_sol, raw_lp);
        if (rec_status.is_ok()) {
            verif_res = SolutionVerifier::verify(raw_lp, rec_status.value().x, rec_status.value().y, cpu_pdhg_res.primal_objective);
        }
    }

    // Evaluate Prediction Outcome
    SolverCandidate cand_solver = SolverCandidate::DualSimplex;
    BackendCandidate cand_backend = BackendCandidate::CPU;
    if (actual_winner_solver == "PDHG_GPU") {
        cand_solver = SolverCandidate::PDHG_GPU;
        cand_backend = BackendCandidate::GPU;
    } else if (actual_winner_solver == "PDHG_CPU") {
        cand_solver = SolverCandidate::PDHG_CPU;
        cand_backend = BackendCandidate::CPU;
    } else if (actual_winner_solver == "BranchAndBound") {
        cand_solver = SolverCandidate::BranchAndBound;
        cand_backend = BackendCandidate::CPU;
    }
    auto outcome = StructureAwareSelector::evaluate_outcome(rec, cand_solver, cand_backend);
    std::string prediction_outcome = to_string(outcome);

    // =========================================================================
    // 6. JSON EMISSION
    // =========================================================================
    std::ostringstream json;
    json << "{\n";

    // Meta
    json << "  \"model\": {\n"
         << "    \"path\": \"" << escape_json(mps_path) << "\",\n"
         << "    \"name\": \"" << escape_json(raw_lp.name) << "\",\n"
         << "    \"parse_time_ms\": " << format_double(parse_time_ms) << "\n"
         << "  },\n";

    // Phase 1: Sparse Core
    json << "  \"phase1_sparse_core\": {\n"
         << "    \"raw_rows\": " << raw_m << ",\n"
         << "    \"raw_cols\": " << raw_n << ",\n"
         << "    \"raw_nonzeros\": " << raw_nnz << ",\n"
         << "    \"raw_density\": " << format_double(raw_density) << ",\n"
         << "    \"coo_bytes\": " << coo_bytes << ",\n"
         << "    \"csr_bytes\": " << csr_bytes << ",\n"
         << "    \"csc_bytes\": " << csc_bytes << ",\n"
         << "    \"continuous_vars\": " << count_continuous << ",\n"
         << "    \"binary_vars\": " << count_binary << ",\n"
         << "    \"integer_vars\": " << count_integer << ",\n"
         << "    \"less_equal_rows\": " << count_le << ",\n"
         << "    \"greater_equal_rows\": " << count_ge << ",\n"
         << "    \"equal_rows\": " << count_eq << ",\n"
         << "    \"range_rows\": " << count_range << ",\n"
         << "    \"spmv_suitability\": \"" << (raw_density > 0.1 ? "Scalar (Short/Dense Rows)" : (stats.staircase_score > 0.7 ? "Adaptive (Block-Angular)" : "Standard CSR")) << "\"\n"
         << "  },\n";

    // Phase 2: Model Preparation
    json << "  \"phase2_model_preparation\": {\n"
         << "    \"prepared_rows\": " << prep_m << ",\n"
         << "    \"prepared_cols\": " << prep_n << ",\n"
         << "    \"prepared_nonzeros\": " << prep_nnz << ",\n"
         << "    \"prepared_density\": " << format_double(prep_density) << ",\n"
         << "    \"rows_removed\": " << rows_removed << ",\n"
         << "    \"cols_removed\": " << cols_removed << ",\n"
         << "    \"nnz_removed\": " << nnz_removed << ",\n"
         << "    \"nnz_reduction_pct\": " << format_double(nnz_reduction_pct) << ",\n"
         << "    \"prep_time_ms\": " << format_double(prep_time_ms) << ",\n"
         << "    \"dynamic_range_before\": " << format_double(dyn_range_before) << ",\n"
         << "    \"dynamic_range_after\": " << format_double(dyn_range_after) << ",\n"
         << "    \"scaling_algorithm\": \"Ruiz Equilibration (L-inf)\"\n"
         << "  },\n";

    // Phase 5: Structure & Prediction
    json << "  \"phase5_structure_and_prediction\": {\n"
         << "    \"staircase_score\": " << format_double(stats.staircase_score) << ",\n"
         << "    \"row_gini_index\": " << format_double(stats.row_length_gini) << ",\n"
         << "    \"max_row_degree\": " << stats.row_nnz_max << ",\n"
         << "    \"avg_row_degree\": " << format_double(stats.row_nnz_avg) << ",\n"
         << "    \"half_bandwidth\": " << stats.half_bandwidth << ",\n"
         << "    \"connected_components\": " << stats.num_connected_components << ",\n"
         << "    \"integrality_ratio\": " << format_double((raw_n > 0) ? static_cast<double>(count_binary + count_integer) / raw_n : 0.0) << ",\n"
         << "    \"predicted_solver\": \"" << escape_json(predicted_solver) << "\",\n"
         << "    \"predicted_backend\": \"" << escape_json(predicted_backend) << "\",\n"
         << "    \"prediction_rationale\": \"" << escape_json(prediction_rationale) << "\"\n"
         << "  },\n";

    // Phase 3: PDHG (CPU & GPU)
    json << "  \"phase3_pdhg\": {\n"
         << "    \"cpu\": {\n"
         << "      \"executed\": " << (cpu_pdhg_run ? "true" : "false") << ",\n"
         << "      \"status\": \"" << (cpu_pdhg_run ? to_string(cpu_pdhg_res.status) : "SKIPPED") << "\",\n"
         << "      \"objective\": " << (cpu_pdhg_run ? format_double(cpu_pdhg_res.primal_objective) : "null") << ",\n"
         << "      \"iterations\": " << (cpu_pdhg_run ? cpu_pdhg_res.iterations : 0) << ",\n"
         << "      \"primal_residual\": " << (cpu_pdhg_run ? format_double(cpu_pdhg_res.primal_residual) : "null") << ",\n"
         << "      \"dual_residual\": " << (cpu_pdhg_run ? format_double(cpu_pdhg_res.dual_residual) : "null") << ",\n"
         << "      \"solve_time_ms\": " << (cpu_pdhg_run ? format_double(cpu_pdhg_res.timing.total_time_ms) : "null") << ",\n"
         << "      \"history\": [\n";
    if (cpu_pdhg_run) {
        for (size_t k = 0; k < cpu_pdhg_res.history.size(); ++k) {
            const auto& h = cpu_pdhg_res.history[k];
            json << "        {\"iter\": " << h.iteration << ", \"primal\": " << format_double(h.primal_residual)
                 << ", \"dual\": " << format_double(h.dual_residual) << ", \"obj\": " << format_double(h.objective)
                 << ", \"time_ms\": " << format_double(h.elapsed_ms) << "}"
                 << (k + 1 < cpu_pdhg_res.history.size() ? ",\n" : "\n");
        }
    }
    json << "      ]\n    },\n";

    json << "    \"gpu\": {\n"
         << "      \"supported\": " <<
#if defined(PIPEPYE_CUDA_ENABLED) || defined(PIPEPYE_HAS_CUDA)
            "true"
#else
            "false"
#endif
         << ",\n"
         << "      \"executed\": " << (gpu_pdhg_run ? "true" : "false") << ",\n"
         << "      \"status\": \"" << (gpu_pdhg_run ? to_string(gpu_pdhg_res.status) : "NOT_AVAILABLE") << "\",\n"
         << "      \"objective\": " << (gpu_pdhg_run ? format_double(gpu_pdhg_res.primal_objective) : "null") << ",\n"
         << "      \"iterations\": " << (gpu_pdhg_run ? gpu_pdhg_res.iterations : 0) << ",\n"
         << "      \"primal_residual\": " << (gpu_pdhg_run ? format_double(gpu_pdhg_res.primal_residual) : "null") << ",\n"
         << "      \"dual_residual\": " << (gpu_pdhg_run ? format_double(gpu_pdhg_res.dual_residual) : "null") << ",\n"
         << "      \"pure_solve_ms\": " << (gpu_pdhg_run ? format_double(gpu_pdhg_res.timing.pure_solve_ms) : "null") << ",\n"
         << "      \"h2d_transfer_ms\": " << (gpu_pdhg_run ? format_double(gpu_pdhg_res.timing.h2d_transfer_ms) : "null") << ",\n"
         << "      \"d2h_transfer_ms\": " << (gpu_pdhg_run ? format_double(gpu_pdhg_res.timing.d2h_transfer_ms) : "null") << ",\n"
         << "      \"total_time_ms\": " << (gpu_pdhg_run ? format_double(gpu_pdhg_res.timing.total_time_ms) : "null") << ",\n"
         << "      \"history\": [\n";
    if (gpu_pdhg_run) {
        for (size_t k = 0; k < gpu_pdhg_res.history.size(); ++k) {
            const auto& h = gpu_pdhg_res.history[k];
            json << "        {\"iter\": " << h.iteration << ", \"primal\": " << format_double(h.primal_residual)
                 << ", \"dual\": " << format_double(h.dual_residual) << ", \"obj\": " << format_double(h.objective)
                 << ", \"time_ms\": " << format_double(h.elapsed_ms) << "}"
                 << (k + 1 < gpu_pdhg_res.history.size() ? ",\n" : "\n");
        }
    }
    json << "      ]\n    }\n";
    json << "  },\n";

    // Phase 4: Dual Simplex
    json << "  \"phase4_dual_simplex\": {\n"
         << "    \"executed\": " << (simplex_run ? "true" : "false") << ",\n"
         << "    \"status\": \"" << (simplex_run ? to_string(simplex_res.status) : "SKIPPED") << "\",\n"
         << "    \"objective\": " << (simplex_run ? format_double(simplex_res.objective_value) : "null") << ",\n"
         << "    \"pivots\": " << (simplex_run ? simplex_res.iterations : 0) << ",\n"
         << "    \"bound_flips\": " << (simplex_run ? simplex_res.bound_flips : 0) << ",\n"
         << "    \"factorizations\": " << (simplex_run ? simplex_res.factorizations : 0) << ",\n"
         << "    \"pfi_eta_updates\": " << (simplex_run ? simplex_res.updates : 0) << ",\n"
         << "    \"ftran_count\": " << (simplex_run ? simplex_res.ftran_count : 0) << ",\n"
         << "    \"btran_count\": " << (simplex_run ? simplex_res.btran_count : 0) << ",\n"
         << "    \"max_primal_infeasibility\": " << (simplex_run ? format_double(simplex_res.max_primal_infeasibility) : "null") << ",\n"
         << "    \"max_dual_infeasibility\": " << (simplex_run ? format_double(simplex_res.max_dual_infeasibility) : "null") << ",\n"
         << "    \"solve_time_ms\": " << (simplex_run ? format_double(simplex_res.timing.total_time_ms) : "null") << "\n"
         << "  },\n";

    // Phase 4: Crossover
    json << "  \"phase4_crossover\": {\n"
         << "    \"executed\": " << (crossover_run ? "true" : "false") << ",\n"
         << "    \"succeeded\": " << (crossover_run && crossover_res.is_optimal() ? "true" : "false") << ",\n"
         << "    \"active_bounds_detected\": " << (crossover_run ? crossover_res.active_bounds_detected : 0) << ",\n"
         << "    \"structural_basic_vars\": " << (crossover_run ? crossover_res.structural_basic_vars : 0) << ",\n"
         << "    \"slack_basic_vars\": " << (crossover_run ? crossover_res.slack_basic_vars : 0) << ",\n"
         << "    \"cleanup_pivots\": " << (crossover_run ? crossover_res.cleanup_pivots : 0) << ",\n"
         << "    \"crash_time_ms\": " << (crossover_run ? format_double(crossover_res.crash_time_ms) : "null") << ",\n"
         << "    \"simplex_cleanup_time_ms\": " << (crossover_run ? format_double(crossover_res.simplex_cleanup_time_ms) : "null") << ",\n"
         << "    \"total_crossover_time_ms\": " << (crossover_run ? format_double(crossover_res.total_time_ms) : "null") << ",\n"
         << "    \"final_objective\": " << (crossover_run ? format_double(crossover_res.final_simplex_objective) : "null") << ",\n"
         << "    \"crossover_faster\": " << (crossover_run && crossover_res.total_time_ms < simplex_res.timing.total_time_ms ? "true" : "false") << ",\n"
         << "    \"note\": \"Crossover success denotes vertex basis recovery; runtime speedup depends on problem scale and initial PDHG precision.\"\n"
         << "  },\n";

    // Phase 7: MILP Branch-and-Bound (with Phase 5 compatibility)
    json << "  \"phase7_milp\": {\n"
         << "    \"is_milp\": " << (is_milp ? "true" : "false") << ",\n"
         << "    \"executed\": " << (milp_run ? "true" : "false") << ",\n"
         << "    \"config_name\": \"" << escape_json(milp_cfg_name) << "\",\n"
         << "    \"integer_objective\": " << (milp_run ? format_double(milp_warm_res.best_objective) : "null") << ",\n"
         << "    \"best_dual_bound\": " << (milp_run ? format_double(milp_warm_res.best_bound) : "null") << ",\n"
         << "    \"mip_gap\": " << (milp_run ? format_double(milp_warm_res.mip_gap) : "null") << ",\n"
         << "    \"nodes_created\": " << (milp_run ? milp_warm_res.nodes_created : 0) << ",\n"
         << "    \"nodes_explored\": " << (milp_run ? milp_warm_res.nodes_explored : 0) << ",\n"
         << "    \"nodes_per_sec\": " << (milp_run ? format_double(milp_warm_res.nodes_per_second) : "0.0") << ",\n"
         << "    \"peak_tree_size\": " << (milp_run ? milp_warm_res.peak_tree_size : 0) << ",\n"
         << "    \"active_tree_size\": " << (milp_run ? milp_warm_res.active_tree_size : 0) << ",\n"
         << "    \"nodes_pruned_bound\": " << (milp_run ? milp_warm_res.nodes_pruned_bound : 0) << ",\n"
         << "    \"nodes_pruned_infeasible\": " << (milp_run ? milp_warm_res.nodes_pruned_infeasible : 0) << ",\n"
         << "    \"nodes_pruned_integral\": " << (milp_run ? milp_warm_res.nodes_pruned_integral : 0) << ",\n"
         << "    \"root_cuts_added\": " << (milp_run ? milp_warm_res.root_cuts_added : 0) << ",\n"
         << "    \"root_bound_after_cuts\": " << (milp_run ? format_double(milp_warm_res.root_bound_after_cuts) : "null") << ",\n"
         << "    \"heuristic_solutions_found\": " << (milp_run ? milp_warm_res.heuristic_solutions_found : 0) << ",\n"
         << "    \"warm_pivots\": " << (milp_run ? milp_warm_res.total_pivots : 0) << ",\n"
         << "    \"cold_pivots\": " << (milp_run ? milp_cold_res.total_pivots : 0) << ",\n"
         << "    \"pivot_reduction_pct\": " << (milp_run && milp_cold_res.total_pivots > 0 ?
                format_double(100.0 * (milp_cold_res.total_pivots - milp_warm_res.total_pivots) / milp_cold_res.total_pivots) : "0.0") << ",\n"
         << "    \"warm_nodes_explored\": " << (milp_run ? milp_warm_res.nodes_explored : 0) << ",\n"
         << "    \"cold_nodes_explored\": " << (milp_run ? milp_cold_res.nodes_explored : 0) << ",\n"
         << "    \"warm_time_ms\": " << (milp_run ? format_double(milp_warm_res.total_time_ms) : "null") << ",\n"
         << "    \"cold_time_ms\": " << (milp_run ? format_double(milp_cold_res.total_time_ms) : "null") << ",\n"
         << "    \"time_to_first_incumbent_ms\": " << (milp_run ? format_double(milp_warm_res.time_to_first_incumbent_ms) : "null") << ",\n"
         << "    \"time_to_gap_10pct_ms\": " << (milp_run ? format_double(milp_warm_res.time_to_gap_10pct_ms) : "null") << ",\n"
         << "    \"time_to_gap_1pct_ms\": " << (milp_run ? format_double(milp_warm_res.time_to_gap_1pct_ms) : "null") << ",\n"
         << "    \"time_to_gap_01pct_ms\": " << (milp_run ? format_double(milp_warm_res.time_to_gap_01pct_ms) : "null") << ",\n"
         << "    \"cut_time_ms\": " << (milp_run ? format_double(milp_warm_res.cut_generation_time_ms) : "null") << ",\n"
         << "    \"heuristic_time_ms\": " << (milp_run ? format_double(milp_warm_res.heuristic_time_ms) : "null") << ",\n"
         << "    \"lp_relaxation_time_ms\": " << (milp_run ? format_double(milp_warm_res.lp_relaxation_time_ms) : "null") << ",\n"
         << "    \"gap_history\": [\n";
    if (milp_run) {
        for (size_t k = 0; k < milp_warm_res.gap_history.size(); ++k) {
            const auto& m = milp_warm_res.gap_history[k];
            json << "        {\"node\": " << m.node << ", \"time_ms\": " << format_double(m.time_ms)
                 << ", \"bound\": " << format_double(m.dual_bound) << ", \"obj\": " << format_double(m.incumbent_obj)
                 << ", \"gap\": " << format_double(m.gap) << "}"
                 << (k + 1 < milp_warm_res.gap_history.size() ? ",\n" : "\n");
        }
    }
    json << "    ]\n"
         << "  },\n";

    // Phase 5 backward compatibility alias
    json << "  \"phase5_milp\": {\n"
         << "    \"is_milp\": " << (is_milp ? "true" : "false") << ",\n"
         << "    \"executed\": " << (milp_run ? "true" : "false") << ",\n"
         << "    \"integer_objective\": " << (milp_run ? format_double(milp_warm_res.best_objective) : "null") << ",\n"
         << "    \"warm_pivots\": " << (milp_run ? milp_warm_res.total_pivots : 0) << ",\n"
         << "    \"cold_pivots\": " << (milp_run ? milp_cold_res.total_pivots : 0) << ",\n"
         << "    \"pivot_reduction_pct\": " << (milp_run && milp_cold_res.total_pivots > 0 ?
                format_double(100.0 * (milp_cold_res.total_pivots - milp_warm_res.total_pivots) / milp_cold_res.total_pivots) : "0.0") << ",\n"
         << "    \"warm_nodes_explored\": " << (milp_run ? milp_warm_res.nodes_explored : 0) << ",\n"
         << "    \"cold_nodes_explored\": " << (milp_run ? milp_cold_res.nodes_explored : 0) << ",\n"
         << "    \"warm_time_ms\": " << (milp_run ? format_double(milp_warm_res.total_time_ms) : "null") << ",\n"
         << "    \"cold_time_ms\": " << (milp_run ? format_double(milp_cold_res.total_time_ms) : "null") << "\n"
         << "  },\n";

    // Verification
    json << "  \"solution_verification\": {\n"
         << "    \"audited_solver\": \"" << escape_json(audited_solver) << "\",\n"
         << "    \"is_valid\": " << (verif_res.is_valid() ? "true" : "false") << ",\n"
         << "    \"is_feasible\": " << (verif_res.is_feasible ? "true" : "false") << ",\n"
         << "    \"is_objective_consistent\": " << (verif_res.is_objective_consistent ? "true" : "false") << ",\n"
         << "    \"max_bound_violation\": " << format_double(verif_res.max_bound_violation) << ",\n"
         << "    \"max_constraint_violation\": " << format_double(verif_res.max_constraint_violation) << ",\n"
         << "    \"recomputed_objective\": " << format_double(verif_res.recomputed_objective) << ",\n"
         << "    \"objective_mismatch\": " << format_double(verif_res.objective_mismatch) << ",\n"
         << "    \"status\": \"" << (verif_res.is_valid() ? "PASSED_FEASIBLE" : "FAILED") << "\"\n"
         << "  },\n";

    // Executive Summary
    json << "  \"executive_summary\": {\n"
         << "    \"actual_winner_solver\": \"" << escape_json(actual_winner_solver) << "\",\n"
         << "    \"actual_winner_backend\": \"" << escape_json(actual_winner_backend) << "\",\n"
         << "    \"best_runtime_ms\": " << format_double(best_runtime_ms) << ",\n"
         << "    \"best_objective\": " << format_double(best_objective) << ",\n"
         << "    \"prediction_outcome\": \"" << escape_json(prediction_outcome) << "\",\n"
         << "    \"confirmed\": " << (prediction_outcome == "CONFIRMED" ? "true" : "false") << "\n"
         << "  }\n";

    json << "}\n";

    std::string json_str = json.str();

    if (!out_json_path.empty()) {
        std::ofstream out_file(out_json_path);
        out_file << json_str;
    }

    std::cout << json_str;
    return 0;
}
