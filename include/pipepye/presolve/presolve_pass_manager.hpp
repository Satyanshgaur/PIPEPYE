#pragma once

#include <vector>
#include <memory>
#include <string>
#include <pipepye/core/types.hpp>
#include <pipepye/core/status.hpp>
#include <pipepye/model/lp_model.hpp>
#include <pipepye/presolve/presolve_types.hpp>
#include <pipepye/presolve/presolve_pass.hpp>
#include <pipepye/presolve/postsolve.hpp>

namespace pipepye::presolve {

/// @brief Result container produced by PresolvePassManager.
struct PresolvedModel {
    model::LinearProgram lp;             ///< Compact reduced linear program (empty if solved/infeasible).
    PostsolveManager postsolve_mgr;      ///< Reverse transformation stack for full solution reconstruction.
    PresolveStats stats;                 ///< Profiling metrics and reduction statistics.
    PresolveStatus status;               ///< Presolve outcome status.

    [[nodiscard]] bool is_solved() const noexcept {
        return status == PresolveStatus::OptimalSolved;
    }

    [[nodiscard]] bool is_infeasible() const noexcept {
        return status == PresolveStatus::Infeasible;
    }

    [[nodiscard]] bool is_unbounded() const noexcept {
        return status == PresolveStatus::Unbounded;
    }

    [[nodiscard]] bool is_reduced() const noexcept {
        return status == PresolveStatus::Reduced || status == PresolveStatus::OptimalSolved;
    }
};

/// @brief Orchestrates and executes the iterative presolve reduction pipeline.
class PresolvePassManager {
public:
    PresolvePassManager();
    explicit PresolvePassManager(PresolveOptions options);

    /// @brief Registers a custom reduction pass into the execution pipeline.
    void add_pass(std::unique_ptr<PresolvePass> pass);

    /// @brief Clears existing passes and configures standard canonical pipeline.
    void set_default_pipeline();

    /// @brief Executes the presolve pipeline on the input LinearProgram.
    [[nodiscard]] StatusOr<PresolvedModel> run(const model::LinearProgram& original_lp);

    [[nodiscard]] const PresolveOptions& options() const noexcept { return options_; }
    void set_options(PresolveOptions options) noexcept { options_ = options; }

private:
    PresolveOptions options_;
    std::vector<std::unique_ptr<PresolvePass>> passes_;
};

} // namespace pipepye::presolve
