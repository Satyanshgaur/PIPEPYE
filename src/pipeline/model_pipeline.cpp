#include <pipepye/pipeline/model_pipeline.hpp>

namespace pipepye::pipeline {

StatusOr<PreparedLP> ModelPipeline::prepare(const model::LinearProgram& original_lp) const {
    PreparedLP out;
    model::LinearProgram current_lp = original_lp;
    std::shared_ptr<presolve::PostsolveManager> postsolve_mgr = nullptr;
    bool was_presolved = false;
    bool was_scaled = false;

    // 1. Presolve Reduction Pass Pipeline
    if (config_.enable_presolve) {
        presolve::PresolvePassManager pm(config_.presolve_options);
        auto presolve_res = pm.run(original_lp);
        if (!presolve_res.is_ok()) {
            return presolve_res.status();
        }

        auto presolved_model = std::move(presolve_res.value());
        out.presolve_stats = presolved_model.stats;
        out.is_solved_by_presolve = presolved_model.is_solved();
        out.is_infeasible = presolved_model.is_infeasible();
        out.is_unbounded = presolved_model.is_unbounded();

        postsolve_mgr = std::make_shared<presolve::PostsolveManager>(
            std::move(presolved_model.postsolve_mgr));
        current_lp = std::move(presolved_model.lp);
        was_presolved = true;

        // If presolve concluded the problem conclusively, no need for scaling
        if (out.is_solved_by_presolve || out.is_infeasible || out.is_unbounded) {
            if (config_.compute_characterization && current_lp.num_cols() > 0 && current_lp.num_rows() > 0) {
                out.problem_stats = analysis::ProblemAnalyzer::analyze(current_lp);
            }
            out.recovery_map = SolutionRecoveryMap(postsolve_mgr, {}, was_presolved, false);
            out.lp = std::move(current_lp);
            return out;
        }
    }

    // 2. Matrix Scaling & Equilibration
    if (config_.enable_scaling &&
        current_lp.num_cols() > 0 &&
        current_lp.num_rows() > 0 &&
        current_lp.num_nonzeros() > 0) {

        scaling::Equilibrator equilibrator(config_.scaling_options);
        auto scale_res = equilibrator.scale(current_lp);
        if (!scale_res.is_ok()) {
            return scale_res.status();
        }

        out.scaling = std::move(scale_res.value());
        current_lp = out.scaling.lp;
        was_scaled = true;
    }

    // 3. Problem Structural Characterization & Engine Recommendation
    if (config_.compute_characterization &&
        current_lp.num_cols() > 0 &&
        current_lp.num_rows() > 0) {
        out.problem_stats = analysis::ProblemAnalyzer::analyze(current_lp);
    }

    // 4. Finalize Prepared Container and Recovery Pipeline
    out.recovery_map = SolutionRecoveryMap(postsolve_mgr, out.scaling, was_presolved, was_scaled);
    out.lp = std::move(current_lp);

    return out;
}

} // namespace pipepye::pipeline
