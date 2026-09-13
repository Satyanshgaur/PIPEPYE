#pragma once

#include <pipepye/model/lp_model.hpp>
#include <pipepye/solver/solver_types.hpp>

namespace pipepye::solver {

#if defined(PIPEPYE_CUDA_ENABLED) || defined(PIPEPYE_HAS_CUDA)

/// @brief First-order Primal-Dual Hybrid Gradient (PDHG / Chambolle-Pock) CUDA GPU solver.
/// Maintains resident problem state and iterations entirely in GPU VRAM.
class CudaPDPOptimizer {
public:
    explicit CudaPDPOptimizer(SolverConfig config = SolverConfig::BaselineCUDA())
        : config_(std::move(config)) {}

    /// @brief Solves the bounded linear program on GPU according to configured options.
    /// @param lp The bounded linear program (min c^T x s.t. l_r <= Ax <= u_r, l_c <= x <= u_c).
    [[nodiscard]] SolverResult solve(const model::LinearProgram& lp) const;

    [[nodiscard]] const SolverConfig& config() const noexcept { return config_; }
    void set_config(SolverConfig config) noexcept { config_ = std::move(config); }

private:
    SolverConfig config_;
};

/// @brief Explicitly registers the CUDA solver backend with PDHGSolver router.
void init_cuda_solver();

#endif

} // namespace pipepye::solver
