#pragma once

#include <pipepye/model/lp_model.hpp>
#include <pipepye/solver/solver_types.hpp>

namespace pipepye::solver {

/// @brief First-order Primal-Dual Hybrid Gradient (PDHG / Chambolle-Pock) CPU solver.
class CPUPDPOptimizer {
public:
    explicit CPUPDPOptimizer(SolverConfig config = SolverConfig::BaselineCPU())
        : config_(std::move(config)) {}

    /// @brief Solves the bounded linear program on CPU according to configured options.
    /// @param lp The bounded linear program (min c^T x s.t. l_r <= Ax <= u_r, l_c <= x <= u_c).
    [[nodiscard]] SolverResult solve(const model::LinearProgram& lp) const;

    [[nodiscard]] const SolverConfig& config() const noexcept { return config_; }
    void set_config(SolverConfig config) noexcept { config_ = std::move(config); }

private:
    SolverConfig config_;
};

} // namespace pipepye::solver
