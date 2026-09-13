#pragma once

#include <memory>
#include <sstream>
#include <iomanip>
#include <pipepye/core/types.hpp>
#include <pipepye/core/status.hpp>
#include <pipepye/model/lp_model.hpp>
#include <pipepye/presolve/presolve_types.hpp>
#include <pipepye/presolve/presolve_pass_manager.hpp>
#include <pipepye/scaling/scaling_types.hpp>
#include <pipepye/scaling/equilibrator.hpp>
#include <pipepye/analysis/problem_stats.hpp>
#include <pipepye/analysis/problem_analyzer.hpp>
#include <pipepye/pipeline/pipeline_types.hpp>
#include <pipepye/pipeline/prepared_lp.hpp>

namespace pipepye::pipeline {

/// @brief Fully reproducible and configurable parameters for the model preparation pipeline.
struct PipelineConfig {
    PipelineMode mode{PipelineMode::PRESOLVE_AND_SCALING}; ///< Explicit ablation mode

    presolve::PresolveOptions presolve_options{};          ///< Presolve iteration & pass toggles
    scaling::ScalingOptions scaling_options{};              ///< Matrix scaling algorithm & convergence
    bool compute_characterization{true};                   ///< Compute higher-order problem metrics

    uint32_t random_seed{42};                              ///< Deterministic seed for reproducible runs
    bool verbose{false};                                   ///< Telemetry logging flag

    /// @brief True if presolve reduction passes are active in current ablation mode.
    [[nodiscard]] bool presolve_enabled() const noexcept {
        return mode == PipelineMode::PRESOLVE_ONLY || mode == PipelineMode::PRESOLVE_AND_SCALING;
    }

    /// @brief True if matrix scaling/equilibration is active in current ablation mode.
    [[nodiscard]] bool scaling_enabled() const noexcept {
        return mode == PipelineMode::SCALING_ONLY || mode == PipelineMode::PRESOLVE_AND_SCALING;
    }

    /// @brief Explicitly configure ablation mode.
    PipelineConfig& set_mode(PipelineMode m) noexcept {
        mode = m;
        return *this;
    }

    /// @brief Helper to configure mode via boolean flags.
    PipelineConfig& set_flags(bool presolve, bool scaling) noexcept {
        if (presolve && scaling) mode = PipelineMode::PRESOLVE_AND_SCALING;
        else if (presolve) mode = PipelineMode::PRESOLVE_ONLY;
        else if (scaling) mode = PipelineMode::SCALING_ONLY;
        else mode = PipelineMode::RAW;
        return *this;
    }

    // Factory helpers for canonical ablation configurations
    static PipelineConfig Raw(bool compute_stats = true) {
        PipelineConfig c;
        c.mode = PipelineMode::RAW;
        c.compute_characterization = compute_stats;
        return c;
    }

    static PipelineConfig PresolveOnly(presolve::PresolveOptions p_opts = {}, bool compute_stats = true) {
        PipelineConfig c;
        c.mode = PipelineMode::PRESOLVE_ONLY;
        c.presolve_options = p_opts;
        c.compute_characterization = compute_stats;
        return c;
    }

    static PipelineConfig ScalingOnly(scaling::ScalingOptions s_opts = {}, bool compute_stats = true) {
        PipelineConfig c;
        c.mode = PipelineMode::SCALING_ONLY;
        c.scaling_options = s_opts;
        c.compute_characterization = compute_stats;
        return c;
    }

    static PipelineConfig PresolveAndScaling(presolve::PresolveOptions p_opts = {},
                                            scaling::ScalingOptions s_opts = {},
                                            bool compute_stats = true) {
        PipelineConfig c;
        c.mode = PipelineMode::PRESOLVE_AND_SCALING;
        c.presolve_options = p_opts;
        c.scaling_options = s_opts;
        c.compute_characterization = compute_stats;
        return c;
    }

    [[nodiscard]] std::string format_config() const {
        std::ostringstream ss;
        ss << "PipelineConfig [mode=" << to_string(mode)
           << ", presolve=" << (presolve_enabled() ? "ENABLED" : "DISABLED")
           << ", scaling=" << (scaling_enabled() ? to_string(scaling_options.method) : "DISABLED")
           << ", characterization=" << (compute_characterization ? "ENABLED" : "DISABLED")
           << ", seed=" << random_seed << "]";
        return ss.str();
    }
};

/// @brief Unified pipeline orchestrating presolve reduction, matrix equilibration/scaling,
/// problem structural characterization, and reversible solution mapping with full ablation support.
class ModelPipeline {
public:
    explicit ModelPipeline(PipelineConfig config = {}) : config_(std::move(config)) {}

    /// @brief Executes the end-to-end preparation pipeline on the raw input LP according to configured mode.
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
