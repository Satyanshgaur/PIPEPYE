#pragma once

#include <pipepye/core/types.hpp>
#include <pipepye/core/status.hpp>
#include <pipepye/model/lp_model.hpp>
#include <pipepye/scaling/scaling_types.hpp>

namespace pipepye::scaling {

/// @brief Matrix equilibrator implementing Ruiz equilibration, Pock-Chambolle preconditioning,
/// and numerical diagnostics.
class Equilibrator {
public:
    explicit Equilibrator(ScalingOptions options = {}) : options_(options) {}

    /// @brief Computes comprehensive diagnostic telemetry for any given LinearProgram.
    [[nodiscard]] static ScalingDiagnostics compute_diagnostics(const model::LinearProgram& lp);

    /// @brief Scales the linear program using the configured equilibration strategy.
    [[nodiscard]] StatusOr<ScaledModel> scale(const model::LinearProgram& lp) const;

    [[nodiscard]] const ScalingOptions& options() const noexcept { return options_; }
    void set_options(ScalingOptions options) noexcept { options_ = options; }

private:
    ScalingOptions options_;

    [[nodiscard]] StatusOr<ScaledModel> scale_ruiz(const model::LinearProgram& lp) const;
    [[nodiscard]] StatusOr<ScaledModel> scale_pock_chambolle(const model::LinearProgram& lp) const;
    [[nodiscard]] StatusOr<ScaledModel> scale_l2(const model::LinearProgram& lp) const;
};

} // namespace pipepye::scaling
