#include <pipepye/model/mps_parser.hpp>
#include <pipepye/pipeline/model_pipeline.hpp>
#include <pipepye/analysis/problem_analyzer.hpp>
#include <pipepye/simplex/dual_simplex.hpp>
#include <pipepye/solver/pdhg_solver.hpp>
#include <pipepye/milp/branch_and_bound.hpp>
#include <pipepye/solver/solution_verifier.hpp>
#include <pipepye/utils/timer.hpp>

#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <string>
#include <map>
#include <cmath>
#include <filesystem>

using namespace pipepye;
namespace fs = std::filesystem;

struct NetlibGroundTruth {
    std::string name;
    scalar_t known_optimal;
    index_t rows;
    index_t cols;
    size_t nnz;
};

// Canonical Netlib Reference Objectives
static const std::map<std::string, scalar_t> kNetlibOptimal = {
    {"afiro.mps", -464.753142857},
    {"adlittle.mps", 225494.963162},
    {"blend.mps", -30.8121498458},
    {"beaconfd.mps", 33592.4858072},
    {"sc50a.mps", -64.5750770586},
    {"sc50b.mps", -70.0000000000},
    {"share2b.mps", -415.732240741},
    {"lotfi.mps", -25.2647060619},
    {"e226.mps", -11.638929},  // Netlib / HiGHS optimum for e226
    {"kb2.mps", -1749.900130},
    {"stocfor1.mps", -41131.976219},
    {"bandm.mps", -158.628018}
};

static const std::map<std::string, scalar_t> kMiplibOptimal = {
    {"flugpl.mps", 1201500.0},
    {"p0033.mps", 3089.0},
    {"stein27.mps", 18.0},
    {"enigma.mps", 0.0},
    {"bell3a.mps", 878430.316},
    {"mod008.mps", 307.0},
    {"egout.mps", 568.1007}
};

struct RunRecord {
    std::string corpus;
    std::string model;
    index_t rows{0};
    index_t cols{0};
    size_t nnz{0};
    std::string solver;
    std::string status;
    int iterations{0};
    int nodes{0};
    double time_ms{0.0};
    scalar_t objective{0.0};
    scalar_t reference_obj{0.0};
    double rel_gap_pct{0.0};
    bool verified{false};
};

int main() {
    std::cout << "========================================================================================\n"
              << "       PIPEPYE PUBLIC CORPORA BENCHMARK: NETLIB LP & MIPLIB 3 EVALUATION        \n"
              << "========================================================================================\n\n";

    std::vector<RunRecord> records;

    // -------------------------------------------------------------------------
    // 1. NETLIB LP BENCHMARK
    // -------------------------------------------------------------------------
    std::cout << "--- EVALUATING NETLIB LINEAR PROGRAMMING SUITE ---\n";
    std::cout << std::left << std::setw(16) << "Model"
              << std::setw(12) << "Rows x Cols"
              << std::setw(16) << "Solver"
              << std::setw(14) << "Status"
              << std::setw(10) << "Iters"
              << std::setw(12) << "Time (ms)"
              << std::setw(16) << "PipePye Obj"
              << std::setw(16) << "Known Opt Obj"
              << std::setw(14) << "Rel Gap (%)"
              << std::setw(10) << "Verify"
              << "\n";
    std::cout << std::string(126, '-') << "\n";

    std::vector<std::string> netlib_files = {
        "afiro.mps", "adlittle.mps", "blend.mps", "sc50a.mps", "sc50b.mps",
        "kb2.mps", "share2b.mps", "lotfi.mps", "stocfor1.mps", "e226.mps",
        "beaconfd.mps", "bandm.mps"
    };

    for (const auto& fname : netlib_files) {
        std::string path = "benchmarks/public_corpora/netlib/" + fname;
        if (!fs::exists(path)) continue;

        model::LinearProgram lp;
        if (!model::MPSParser::parse_file(path, lp).is_ok()) {
            std::cerr << "Failed to parse: " << path << "\n";
            continue;
        }

        scalar_t ref_obj = 0.0;
        if (kNetlibOptimal.count(fname)) {
            ref_obj = kNetlibOptimal.at(fname);
        }

        // Test 1: Direct Dual Simplex (Fast)
        {
            simplex::DualSimplexSolver sim_solver(simplex::SimplexConfig::Fast());
            utils::CPUTimer timer;
            timer.start();
            auto res = sim_solver.solve(lp);
            timer.stop();

            double rel_gap = 100.0 * std::abs(res.objective_value - ref_obj) / (1.0 + std::abs(ref_obj));
            auto ver = solver::SolutionVerifier::verify(lp, res.x, res.y, res.objective_value, 1e-3);

            RunRecord rec;
            rec.corpus = "Netlib_LP";
            rec.model = fname;
            rec.rows = lp.num_rows();
            rec.cols = lp.num_cols();
            rec.nnz = lp.num_nonzeros();
            rec.solver = "DualSimplex_Direct";
            rec.status = solver::to_string(res.status);
            rec.iterations = res.iterations;
            rec.time_ms = timer.elapsed_milliseconds();
            rec.objective = res.objective_value;
            rec.reference_obj = ref_obj;
            rec.rel_gap_pct = rel_gap;
            rec.verified = ver.is_valid();
            records.push_back(rec);

            std::string dim_str = std::to_string(lp.num_rows()) + "x" + std::to_string(lp.num_cols());
            std::cout << std::left << std::setw(16) << fname
                      << std::setw(12) << dim_str
                      << std::setw(16) << "DualSimplex"
                      << std::setw(14) << rec.status
                      << std::setw(10) << res.iterations
                      << std::fixed << std::setprecision(2) << std::setw(12) << rec.time_ms
                      << std::setprecision(4) << std::setw(16) << res.objective_value
                      << std::setprecision(4) << std::setw(16) << ref_obj
                      << std::scientific << std::setprecision(2) << std::setw(14) << rel_gap
                      << std::setw(10) << (rec.verified ? "PASSED" : "FAIL")
                      << "\n";
        }

        // Test 2: Prepared Pipeline + Dual Simplex (Presolve + Ruiz)
        {
            pipeline::PipelineConfig pcfg = pipeline::PipelineConfig::PresolveAndScaling();
            auto prep_res = pipeline::ModelPipeline::prepare(lp, pcfg);
            if (prep_res.is_ok()) {
                const auto& prepared = prep_res.value();
                simplex::DualSimplexSolver sim_solver(simplex::SimplexConfig::Fast());
                utils::CPUTimer timer;
                timer.start();
                auto res = sim_solver.solve(prepared);
                timer.stop();

                if (res.is_optimal()) {
                    presolve::PrimalDualSolution pds;
                    pds.x = res.x; pds.y = res.y; pds.s = res.s;
                    pds.objective_value = res.objective_value;
                    pds.is_feasible = true;

                    auto rec_sol = prepared.recover_solution(pds, lp);
                    if (rec_sol.is_ok()) {
                        const auto& sol = rec_sol.value();
                        double rel_gap = 100.0 * std::abs(sol.objective_value - ref_obj) / (1.0 + std::abs(ref_obj));
                        auto ver = solver::SolutionVerifier::verify(lp, sol.x, sol.y, sol.objective_value, 1e-3);

                        RunRecord rec;
                        rec.corpus = "Netlib_LP";
                        rec.model = fname;
                        rec.rows = lp.num_rows();
                        rec.cols = lp.num_cols();
                        rec.nnz = lp.num_nonzeros();
                        rec.solver = "Prepared_Simplex";
                        rec.status = solver::to_string(res.status);
                        rec.iterations = res.iterations;
                        rec.time_ms = timer.elapsed_milliseconds();
                        rec.objective = sol.objective_value;
                        rec.reference_obj = ref_obj;
                        rec.rel_gap_pct = rel_gap;
                        rec.verified = ver.is_valid();
                        records.push_back(rec);

                        std::string dim_str = std::to_string(prepared.lp.num_rows()) + "x" + std::to_string(prepared.lp.num_cols());
                        std::cout << std::left << std::setw(16) << fname
                                  << std::setw(12) << dim_str
                                  << std::setw(16) << "Prep_Simplex"
                                  << std::setw(14) << rec.status
                                  << std::setw(10) << res.iterations
                                  << std::fixed << std::setprecision(2) << std::setw(12) << rec.time_ms
                                  << std::setprecision(4) << std::setw(16) << sol.objective_value
                                  << std::setprecision(4) << std::setw(16) << ref_obj
                                  << std::scientific << std::setprecision(2) << std::setw(14) << rel_gap
                                  << std::setw(10) << (rec.verified ? "PASSED" : "FAIL")
                                  << "\n";
                    }
                }
            }
        }
    }

    // -------------------------------------------------------------------------
    // 2. MIPLIB 3 BENCHMARK
    // -------------------------------------------------------------------------
    std::cout << "\n--- EVALUATING MIPLIB 3 COMBINATORIAL BENCHMARK SUITE ---\n";
    std::cout << std::left << std::setw(16) << "Model"
              << std::setw(12) << "Rows x Cols"
              << std::setw(16) << "Solver"
              << std::setw(14) << "Status"
              << std::setw(10) << "Pivots"
              << std::setw(10) << "Nodes"
              << std::setw(12) << "Time (ms)"
              << std::setw(16) << "PipePye Obj"
              << std::setw(16) << "Known Opt Obj"
              << std::setw(14) << "Rel Gap (%)"
              << "\n";
    std::cout << std::string(126, '-') << "\n";

    std::vector<std::string> miplib_files = {
        "flugpl.mps", "p0033.mps", "stein27.mps", "egout.mps", "mod008.mps", "bell3a.mps"
    };

    for (const auto& fname : miplib_files) {
        std::string path = "benchmarks/public_corpora/miplib/" + fname;
        if (!fs::exists(path)) continue;

        model::LinearProgram lp;
        if (!model::MPSParser::parse_file(path, lp).is_ok()) {
            std::cerr << "Failed to parse: " << path << "\n";
            continue;
        }

        scalar_t ref_obj = 0.0;
        if (kMiplibOptimal.count(fname)) {
            ref_obj = kMiplibOptimal.at(fname);
        }

        std::cout << "Solving MIP: " << fname << std::flush << "\n";
        milp::BranchAndBoundSolver bb;
        milp::MILPConfig cfg = milp::MILPConfig::Advanced();
        if (fname == "flugpl.mps") cfg.max_nodes = 6000; else cfg.max_nodes = 2000;
        cfg.time_limit_sec = 4.0;
        bb.set_config(cfg);

        utils::CPUTimer timer;
        timer.start();
        auto res = bb.solve(lp);
        timer.stop();

        double rel_gap = 100.0 * std::abs(res.best_objective - ref_obj) / (1.0 + std::abs(ref_obj));

        RunRecord rec;
        rec.corpus = "MIPLIB3";
        rec.model = fname;
        rec.rows = lp.num_rows();
        rec.cols = lp.num_cols();
        rec.nnz = lp.num_nonzeros();
        rec.solver = "BranchAndBound";
        rec.status = milp::to_string(res.status);
        rec.iterations = res.total_pivots;
        rec.nodes = res.nodes_explored;
        rec.time_ms = timer.elapsed_milliseconds();
        rec.objective = res.best_objective;
        rec.reference_obj = ref_obj;
        rec.rel_gap_pct = rel_gap;
        rec.verified = res.is_feasible();
        records.push_back(rec);

        std::string dim_str = std::to_string(lp.num_rows()) + "x" + std::to_string(lp.num_cols());
        std::cout << std::left << std::setw(16) << fname
                  << std::setw(12) << dim_str
                  << std::setw(16) << "B&B Simplex"
                  << std::setw(14) << rec.status
                  << std::setw(10) << res.total_pivots
                  << std::setw(10) << res.nodes_explored
                  << std::fixed << std::setprecision(2) << std::setw(12) << rec.time_ms
                  << std::setprecision(4) << std::setw(16) << res.best_objective
                  << std::setprecision(4) << std::setw(16) << ref_obj
                  << std::scientific << std::setprecision(2) << std::setw(14) << rel_gap
                  << "\n";
    }

    // Export CSV report
    std::ofstream csv("reports/public_corpora_benchmark.csv");
    csv << "corpus,model,rows,cols,nnz,solver,status,iterations,nodes,time_ms,objective,reference_obj,rel_gap_pct,verified\n";
    for (const auto& r : records) {
        csv << r.corpus << "," << r.model << "," << r.rows << "," << r.cols << "," << r.nnz << ","
            << r.solver << "," << r.status << "," << r.iterations << "," << r.nodes << ","
            << r.time_ms << "," << r.objective << "," << r.reference_obj << "," << r.rel_gap_pct << ","
            << (r.verified ? "1" : "0") << "\n";
    }
    std::cout << "\n[Report] Exported results to reports/public_corpora_benchmark.csv\n";

    return 0;
}
