#include <pipepye/presolve/presolve_pass_manager.hpp>
#include <pipepye/utils/timer.hpp>
#include <iostream>

namespace pipepye::presolve {

PresolvePassManager::PresolvePassManager()
    : options_() {
    set_default_pipeline();
}

PresolvePassManager::PresolvePassManager(PresolveOptions options)
    : options_(options) {
    set_default_pipeline();
}

void PresolvePassManager::add_pass(std::unique_ptr<PresolvePass> pass) {
    passes_.push_back(std::move(pass));
}

void PresolvePassManager::set_default_pipeline() {
    passes_.clear();
    if (options_.enable_empty_row_col) {
        passes_.push_back(std::make_unique<EmptyRowColPass>());
    }
    if (options_.enable_fixed_variables) {
        passes_.push_back(std::make_unique<FixedVariablePass>());
    }
    if (options_.enable_singletons) {
        passes_.push_back(std::make_unique<SingletonPass>());
    }
    if (options_.enable_forcing_redundancy) {
        passes_.push_back(std::make_unique<ForcingRedundancyPass>());
    }
    if (options_.enable_bound_tightening) {
        passes_.push_back(std::make_unique<BoundTighteningPass>());
    }
}

StatusOr<PresolvedModel> PresolvePassManager::run(const model::LinearProgram& original_lp) {
    PresolveStats stats;
    stats.initial_rows = original_lp.num_rows();
    stats.initial_cols = original_lp.num_cols();
    stats.initial_nnz = original_lp.num_nonzeros();

    if (stats.initial_rows == 0 && stats.initial_cols == 0) {
        PresolvedModel model;
        model.status = PresolveStatus::EmptyModel;
        model.stats = stats;
        return model;
    }

    PresolveContext ctx(original_lp, options_);
    utils::CPUTimer total_timer;
    total_timer.start();

    for (int iter = 0; iter < options_.max_passes; ++iter) {
        bool reduced_in_iteration = false;

        if (ctx.num_active_cols() == 0) {
            ctx.set_status(PresolveStatus::OptimalSolved);
            break;
        }

        for (const auto& pass : passes_) {
            utils::CPUTimer pass_timer;
            pass_timer.start();

            auto pass_res = pass->run(ctx);
            pass_timer.stop();

            if (!pass_res.is_ok()) {
                return pass_res.status();
            }

            PassStats pstat = pass_res.value();
            pstat.elapsed_ms = pass_timer.elapsed_milliseconds();
            stats.pass_history.push_back(pstat);

            if (ctx.status() == PresolveStatus::Infeasible ||
                ctx.status() == PresolveStatus::Unbounded) {
                break;
            }

            if (pstat.rows_removed > 0 || pstat.cols_removed > 0 ||
                pstat.bounds_tightened > 0 || pstat.variables_fixed > 0 ||
                pstat.redundant_rows > 0) {
                reduced_in_iteration = true;
            }
        }

        ++stats.total_passes_executed;

        if (ctx.status() == PresolveStatus::Infeasible ||
            ctx.status() == PresolveStatus::Unbounded) {
            break;
        }

        if (ctx.num_active_cols() == 0) {
            ctx.set_status(PresolveStatus::OptimalSolved);
            break;
        }

        // Fixed point reached if no reduction occurred across all enabled passes
        if (!reduced_in_iteration) {
            break;
        }
    }

    total_timer.stop();
    stats.total_elapsed_ms = total_timer.elapsed_milliseconds();

    PresolveStatus final_status = ctx.status();
    if (final_status == PresolveStatus::Infeasible || final_status == PresolveStatus::Unbounded) {
        PresolvedModel res;
        res.status = final_status;
        res.stats = stats;
        return res;
    }

    if (ctx.num_active_cols() == 0) {
        final_status = PresolveStatus::OptimalSolved;
    } else if (ctx.num_active_cols() < original_lp.num_cols() ||
               ctx.num_active_rows() < original_lp.num_rows()) {
        final_status = PresolveStatus::Reduced;
    } else {
        final_status = PresolveStatus::Unchanged;
    }

    model::LinearProgram pre_lp = ctx.to_presolved_lp();
    stats.final_rows = pre_lp.num_rows();
    stats.final_cols = pre_lp.num_cols();
    stats.final_nnz = pre_lp.num_nonzeros();

    PresolvedModel result;
    result.lp = std::move(pre_lp);
    result.postsolve_mgr = std::move(ctx.postsolve_mgr());
    result.stats = stats;
    result.status = final_status;

    return result;
}

} // namespace pipepye::presolve
