#include <pipepye/model/mps_parser.hpp>
#include <pipepye/pipeline/model_pipeline.hpp>
#include <pipepye/analysis/problem_analyzer.hpp>
#include <pipepye/analysis/solver_selector.hpp>
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
#include <filesystem>

using namespace pipepye;
namespace fs = std::filesystem;

struct PipelineRunResult {
    std::string case_id;
    std::string instance_name;
    std::string mps_file;
    index_t orig_rows{0};
    index_t orig_cols{0};
    size_t orig_nnz{0};
    index_t prep_rows{0};
    index_t prep_cols{0};
    size_t prep_nnz{0};
    index_t num_binaries{0};
    
    double parse_time_ms{0.0};
    double prep_time_ms{0.0};
    double solve_time_ms{0.0};
    double total_time_ms{0.0};
    
    std::string chosen_solver;
    std::string termination_status;
    int iterations_or_pivots{0};
    int nodes_explored{0};
    scalar_t reported_objective{0.0};
    scalar_t recovered_objective{0.0};
    
    bool verification_passed{false};
    scalar_t max_bound_violation{0.0};
    scalar_t max_row_violation{0.0};
    scalar_t objective_mismatch{0.0};
    
    scalar_t highs_reference_obj{0.0};
    double relative_gap_vs_highs{0.0};
};

int main(int argc, char** argv) {
    std::cout << "========================================================================================\n"
              << "          PipePye Complete End-to-End Industrial Pipeline Execution Harness             \n"
              << "  [MPS Parsing -> Characterization -> Presolve & Scaling -> Solver -> Verification]    \n"
              << "========================================================================================\n\n";

    struct InstanceSpec {
        std::string case_id;
        std::string name;
        std::string mps_path;
        scalar_t highs_obj;
        bool is_milp;
    };

    std::vector<InstanceSpec> instances = {
        {"case_a_crude_blending", "Crude_Blending_Small", "workloads/case_a_crude_blending/Small.mps", -1710944.3857, false},
        {"case_a_crude_blending", "Crude_Blending_Medium", "workloads/case_a_crude_blending/Medium.mps", -2858453.2144, false},
        {"case_a_crude_blending", "Crude_Blending_Large", "workloads/case_a_crude_blending/Large.mps", -5548348.0065, false},
        {"case_b_multi_period_planning", "Multi_Period_T10", "workloads/case_b_multi_period_planning/T10.mps", 268440.0, false},
        {"case_b_multi_period_planning", "Multi_Period_T25", "workloads/case_b_multi_period_planning/T25.mps", 661056.4, false},
        {"case_b_multi_period_planning", "Multi_Period_T50", "workloads/case_b_multi_period_planning/T50.mps", 3434088.0, false},
        {"case_b_multi_period_planning", "Multi_Period_T100", "workloads/case_b_multi_period_planning/T100.mps", 6746602.0, false},
        {"case_c_refinery_scheduling", "Refinery_Sched_Small", "workloads/case_c_refinery_scheduling/Small.mps", -6645.986, true},
        {"case_c_refinery_scheduling", "Refinery_Sched_Medium", "workloads/case_c_refinery_scheduling/Medium.mps", 84193.61, true},
        {"case_d_unit_commitment", "Unit_Commit_Small", "workloads/case_d_unit_commitment/Small.mps", 204174.4, true},
        {"case_d_unit_commitment", "Unit_Commit_Medium", "workloads/case_d_unit_commitment/Medium.mps", 671250.8, true}
    };

    std::vector<PipelineRunResult> run_results;

    for (const auto& spec : instances) {
        std::cout << ">>> Running Instance: " << spec.name << " (" << spec.mps_path << ")\n";
        PipelineRunResult run;
        run.case_id = spec.case_id;
        run.instance_name = spec.name;
        run.mps_file = spec.mps_path;
        run.highs_reference_obj = spec.highs_obj;

        // 1. MPS Parsing
        utils::CPUTimer parse_timer;
        model::LinearProgram raw_lp;
        Status parse_status = model::MPSParser::parse_file(spec.mps_path, raw_lp);
        run.parse_time_ms = parse_timer.elapsed_milliseconds();

        if (!parse_status.is_ok()) {
            std::cerr << "  [ERROR] MPS Parse failed: " << parse_status.message() << "\n";
            continue;
        }

        run.orig_rows = raw_lp.num_rows();
        run.orig_cols = raw_lp.num_cols();
        run.orig_nnz = raw_lp.num_nonzeros();

        for (auto vt : raw_lp.var_types) {
            if (vt == model::VariableType::Binary) run.num_binaries++;
        }

        // 2. Characterization
        analysis::ProblemStats stats = analysis::ProblemAnalyzer::analyze(raw_lp);

        // 3. Pipeline Preparation (Presolve & Scaling)
        utils::CPUTimer prep_timer;
        pipeline::PipelineConfig pcfg;
        // For continuous LP, apply full Presolve + Scaling.
        // For MILP, keep raw bounds to preserve integer lattice geometry.
        pcfg.mode = spec.is_milp ? pipeline::PipelineMode::RAW : pipeline::PipelineMode::PRESOLVE_AND_SCALING;
        auto prep_res = pipeline::ModelPipeline::prepare(raw_lp, pcfg);
        run.prep_time_ms = prep_timer.elapsed_milliseconds();

        if (!prep_res.is_ok()) {
            std::cerr << "  [ERROR] Pipeline preparation failed!\n";
            continue;
        }

        const auto& prepared = prep_res.value();
        run.prep_rows = prepared.lp.num_rows();
        run.prep_cols = prepared.lp.num_cols();
        run.prep_nnz = prepared.lp.num_nonzeros();

        // 4. Solve using recommended solver
        utils::CPUTimer solve_timer;
        if (spec.is_milp) {
            run.chosen_solver = "BranchAndBound_DualSimplex";
            milp::BranchAndBoundSolver bb_solver;
            milp::MILPConfig bb_cfg;
            bb_cfg.max_nodes = 500;
            bb_cfg.time_limit_sec = 20.0;
            bb_solver.set_config(bb_cfg);

            auto bb_res = bb_solver.solve(raw_lp);
            run.solve_time_ms = solve_timer.elapsed_milliseconds();
            run.termination_status = milp::to_string(bb_res.status);
            run.iterations_or_pivots = bb_res.total_pivots;
            run.nodes_explored = bb_res.nodes_explored;
            run.reported_objective = bb_res.best_objective;
            run.recovered_objective = bb_res.best_objective;

            if (bb_res.is_feasible()) {
                auto verif = solver::SolutionVerifier::verify(raw_lp, bb_res.x, {}, bb_res.best_objective);
                run.verification_passed = verif.is_valid();
                run.max_bound_violation = verif.max_bound_violation;
                run.max_row_violation = verif.max_constraint_violation;
                run.objective_mismatch = verif.objective_mismatch;
            }
        } else {
            run.chosen_solver = "DualSimplex";
            simplex::DualSimplexSolver simplex_solver;
            auto s_res = simplex_solver.solve(prepared);
            run.solve_time_ms = solve_timer.elapsed_milliseconds();
            run.termination_status = solver::to_string(s_res.status);
            run.iterations_or_pivots = s_res.iterations;
            run.reported_objective = s_res.objective_value;

            if (s_res.is_optimal()) {
                presolve::PrimalDualSolution pds;
                pds.x = s_res.x;
                pds.y = s_res.y;
                pds.s = s_res.s;
                pds.objective_value = s_res.objective_value;
                pds.is_feasible = true;

                auto rec_res = prepared.recover_solution(pds, raw_lp);
                if (rec_res.is_ok()) {
                    const auto& rec = rec_res.value();
                    run.recovered_objective = rec.objective_value;
                    auto verif = solver::SolutionVerifier::verify(raw_lp, rec.x, rec.y, rec.objective_value);
                    run.verification_passed = verif.is_valid();
                    run.max_bound_violation = verif.max_bound_violation;
                    run.max_row_violation = verif.max_constraint_violation;
                    run.objective_mismatch = verif.objective_mismatch;
                }
            }
        }

        run.total_time_ms = run.parse_time_ms + run.prep_time_ms + run.solve_time_ms;
        if (std::abs(run.highs_reference_obj) > 1e-6) {
            run.relative_gap_vs_highs = std::abs(run.recovered_objective - run.highs_reference_obj) /
                                       std::abs(run.highs_reference_obj);
        }

        std::cout << "  - Status: " << run.termination_status 
                  << " | Solved in " << std::fixed << std::setprecision(2) << run.solve_time_ms << " ms"
                  << " (" << run.iterations_or_pivots << " pivots/iters)\n"
                  << "  - PipePye Obj: " << std::scientific << std::setprecision(6) << run.recovered_objective
                  << " | HiGHS Obj: " << run.highs_reference_obj
                  << " (Gap: " << std::setprecision(2) << (run.relative_gap_vs_highs * 100.0) << "%)\n"
                  << "  - Verification: " << (run.verification_passed ? "PASSED" : "FAILED")
                  << " | Max Bound Viol: " << run.max_bound_violation
                  << " | Max Row Viol: " << run.max_row_violation << "\n\n";

        run_results.push_back(run);
    }

    // Export pipeline results to CSV
    std::ofstream out_csv("reports/pipeline_end_to_end_results.csv");
    out_csv << "Case,Instance,Rows,Cols,NNZ,Binaries,Solver,Status,PivotsIters,Nodes,ParseMs,PrepMs,SolveMs,TotalMs,"
            << "PipePyeObj,HiGHSObj,RelativeGapPct,Verified,MaxBoundViol,MaxRowViol\n";
    for (const auto& r : run_results) {
        out_csv << r.case_id << "," << r.instance_name << ","
                << r.orig_rows << "," << r.orig_cols << "," << r.orig_nnz << "," << r.num_binaries << ","
                << r.chosen_solver << "," << r.termination_status << ","
                << r.iterations_or_pivots << "," << r.nodes_explored << ","
                << r.parse_time_ms << "," << r.prep_time_ms << "," << r.solve_time_ms << "," << r.total_time_ms << ","
                << r.recovered_objective << "," << r.highs_reference_obj << ","
                << (r.relative_gap_vs_highs * 100.0) << ","
                << (r.verification_passed ? "YES" : "NO") << ","
                << r.max_bound_violation << "," << r.max_row_violation << "\n";
    }

    std::cout << "Successfully exported reports/pipeline_end_to_end_results.csv!\n";
    return 0;
}
