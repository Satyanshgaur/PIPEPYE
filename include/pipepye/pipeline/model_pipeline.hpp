#pragma once

#include <memory>
#include <pipepye/core/types.hpp>
#include <pipepye/core/status.hpp>
#include <pipepye/model/lp_model.hpp>
#include <pipepye/presolve/presolve_types.hpp>
#include <pipepye/presolve/presolve_pass_manager.hpp>
#include <pipepye/scaling/scaling_types.hpp>
#include <pipepye/scaling/equilibrator.hpp>
#include <pipepye/analysis/problem_stats.hpp>
#include <pipepye/analysis/problem_analyzer.hpp>
#include <pipepye/pipeline/prepared_lp.hpp>

namespace pipepye::pipeline {

/// @brief Configuration specifying transformations for the model preparation pipeline.
struct PipelineConfig {
    bool enable_presolve{true};
    presolve::PresolveOptions presolve_options{};
    bool enable_scaling{true};
    scaling::ScalingOptions scaling_options{};
    bool compute_characterization{true};
};

/// @brief Unified pipeline orchestrating presolve reduction, matrix equilibration/scaling,
/// problem structural characterization, and reversible solution mapping.
class ModelPipeline {
public:
    explicit ModelPipeline(PipelineConfig config = {}) : config_(std::move(config)) {}

    /// @brief Executes the end-to-end preparation pipeline on the raw input LP.
    /// Produces a PreparedLP ready for solver consumption with 1-step solution recovery.
    [[nodiscard]] StatusOr<PreparedLP> prepare(const model::LinearProgram& original_lp) const;

    /// @brief Convenient static single-call helper.
    [[nodiscard]] static StatusOr<PreparedLP> prepare(
        const model::LinearProgram& original_lp,
        const PipelineConfig& config) {
        ModelPipeline pipeline(config);
        return pipeline.prepare(original_lp);
    }

    [[nodiscard]] const PipelineConfig& config() const noexcept { return config_; }
    void set_config(PipelineConfig config) noexcept { config_ = std::move(config); }

private:
    PipelineConfig config_;
};

} // namespace pipepye::pipeline
