#include <pipepye/solver/pdhg_solver.hpp>
#include <pipepye/solver/pdhg_solver_cpu.hpp>
#include <pipepye/utils/timer.hpp>
#include <iostream>

namespace pipepye::solver {

static PDHGSolver::CudaSolverFactory s_cuda_factory = nullptr;

void PDHGSolver::register_cuda_solver_factory(CudaSolverFactory factory) {
    s_cuda_factory = std::move(factory);
}

SolverResult PDHGSolver::solve(
    const pipeline::PreparedLP& prepared_lp,
    const SolverConfig& config) {

    // 1. Check if model was already fully solved by presolve
    if (prepared_lp.is_solved_by_presolve) {
        SolverResult res;
        res.status = TerminationStatus::OPTIMAL;
        res.primal_objective = prepared_lp.lp.obj_offset;
        res.dual_objective = prepared_lp.lp.obj_offset;
        res.iterations = 0;
        return res;
    }

    if (prepared_lp.is_infeasible) {
        SolverResult res;
        res.status = TerminationStatus::PRIMAL_INFEASIBLE;
        res.iterations = 0;
        return res;
    }

    if (prepared_lp.is_unbounded) {
        SolverResult res;
        res.status = TerminationStatus::DUAL_INFEASIBLE_UNBOUNDED;
        res.iterations = 0;
        return res;
    }

    return solve(prepared_lp.lp, config);
}

SolverResult PDHGSolver::solve(
    const model::LinearProgram& lp,
    const SolverConfig& config) {

    if (config.backend == SolverBackend::CUDA) {
        if (s_cuda_factory) {
            return s_cuda_factory(lp, config);
        }
        std::cerr << "PDHGSolver: CUDA backend requested but pipepye_cuda not initialized or registered.\n";
        SolverResult res;
        res.status = TerminationStatus::NUMERICAL_FAILURE;
        return res;
    }

    CPUPDPOptimizer optimizer(config);
    return optimizer.solve(lp);
}

std::pair<SolverResult, VerificationResult> PDHGSolver::solve_end_to_end(
    const model::LinearProgram& orig_lp,
    const pipeline::PipelineConfig& pipe_config,
    const SolverConfig& solver_config,
    scalar_t verification_tol) {

    utils::CPUTimer total_timer;
    total_timer.start();

    // 1. Model preparation through pipeline (presolve, scaling, characterization)
    utils::CPUTimer prep_timer;
    prep_timer.start();
    auto prep_res = pipeline::ModelPipeline::prepare(orig_lp, pipe_config);
    prep_timer.stop();
    double prep_ms = prep_timer.elapsed_milliseconds();

    if (!prep_res.is_ok()) {
        SolverResult err_res;
        err_res.status = TerminationStatus::NUMERICAL_FAILURE;
        VerificationResult err_ver;
        err_ver.details = "Pipeline preparation failure: " + prep_res.status().to_string();
        return {err_res, err_ver};
    }

    const auto& prepared = prep_res.value();

    // 2. Solve the prepared LP
    SolverResult solver_res = solve(prepared, solver_config);
    solver_res.timing.prep_time_ms = prep_ms;

    // 3. Solution recovery back to original model
    utils::CPUTimer post_timer;
    post_timer.start();

    presolve::PrimalDualSolution sol_in;
    sol_in.x = solver_res.x;
    sol_in.y = solver_res.y;
    sol_in.s = solver_res.s;
    sol_in.is_feasible = solver_res.is_converged();

    auto rec_res = prepared.recover_solution(sol_in, orig_lp);
    post_timer.stop();
    solver_res.timing.postsolve_ms = post_timer.elapsed_milliseconds();

    SolverResult recovered_result = solver_res;
    if (rec_res.is_ok()) {
        const auto& rec = rec_res.value();
        recovered_result.x = rec.x;
        recovered_result.y = rec.y;
        recovered_result.s = rec.s;
        recovered_result.primal_objective = rec.objective_value;
    }

    // 4. Independent Verification directly against original model
    utils::CPUTimer ver_timer;
    ver_timer.start();
    VerificationResult ver_res = SolutionVerifier::verify(
        orig_lp,
        recovered_result.x,
        recovered_result.y,
        recovered_result.primal_objective,
        verification_tol);
    ver_timer.stop();

    total_timer.stop();
    recovered_result.timing.verification_ms = ver_timer.elapsed_milliseconds();
    recovered_result.timing.total_time_ms = total_timer.elapsed_milliseconds();

    return {recovered_result, ver_res};
}

} // namespace pipepye::solver
