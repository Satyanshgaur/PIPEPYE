#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <sstream>
#include <iomanip>
#include <pipepye/core/types.hpp>

namespace pipepye::solver {

/// @brief Termination status codes for the PDHG solver.
enum class TerminationStatus : uint8_t {
    OPTIMAL = 0,                ///< Converged to required primal and dual feasibility tolerances
    MAX_ITERATIONS,             ///< Reached maximum allowed iteration limit
    TIME_LIMIT,                 ///< Reached wall-clock execution time limit
    NUMERICAL_FAILURE,          ///< Encountered NaN / Inf or catastrophic divergence
    PRIMAL_INFEASIBLE,          ///< Infeasibility certificate detected
    DUAL_INFEASIBLE_UNBOUNDED   ///< Unboundedness ray detected
};

[[nodiscard]] inline std::string to_string(TerminationStatus status) {
    switch (status) {
        case TerminationStatus::OPTIMAL:                   return "OPTIMAL";
        case TerminationStatus::MAX_ITERATIONS:            return "MAX_ITERATIONS";
        case TerminationStatus::TIME_LIMIT:                return "TIME_LIMIT";
        case TerminationStatus::NUMERICAL_FAILURE:         return "NUMERICAL_FAILURE";
        case TerminationStatus::PRIMAL_INFEASIBLE:         return "PRIMAL_INFEASIBLE";
        case TerminationStatus::DUAL_INFEASIBLE_UNBOUNDED: return "DUAL_INFEASIBLE_UNBOUNDED";
    }
    return "UNKNOWN";
}

/// @brief Step-size selection strategies for primal and dual updates.
enum class StepSizeStrategy : uint8_t {
    CONSTANT = 0,       ///< Scalar constant step-sizes satisfying tau * sigma * ||A||^2 < 1
    POCK_CHAMBOLLE,     ///< Diagonal coordinate step-sizes based on row/column absolute sums
    ADAPTIVE            ///< Dynamically adjusted step-sizes based on primal/dual residual balance
};

[[nodiscard]] inline std::string to_string(StepSizeStrategy strat) {
    switch (strat) {
        case StepSizeStrategy::CONSTANT:       return "CONSTANT";
        case StepSizeStrategy::POCK_CHAMBOLLE: return "POCK_CHAMBOLLE";
        case StepSizeStrategy::ADAPTIVE:       return "ADAPTIVE";
    }
    return "UNKNOWN";
}

/// @brief Restart strategies to accelerate first-order convergence.
enum class RestartStrategy : uint8_t {
    NONE = 0,           ///< Standard fixed-momentum iterations without restarts
    ADAPTIVE            ///< Restart extrapolation when residual reduction stalls
};

[[nodiscard]] inline std::string to_string(RestartStrategy strat) {
    switch (strat) {
        case RestartStrategy::NONE:     return "NONE";
        case RestartStrategy::ADAPTIVE: return "ADAPTIVE";
    }
    return "UNKNOWN";
}

/// @brief Execution engine target for the solver.
enum class SolverBackend : uint8_t {
    CPU = 0,            ///< Multithreaded or single-threaded CPU execution
    CUDA                ///< Resident GPU VRAM accelerated execution
};

[[nodiscard]] inline std::string to_string(SolverBackend backend) {
    switch (backend) {
        case SolverBackend::CPU:  return "CPU";
        case SolverBackend::CUDA: return "CUDA";
    }
    return "UNKNOWN";
}

/// @brief Snapshot of solver state at a specific iteration check.
struct IterationLog {
    int iteration{0};
    double elapsed_ms{0.0};
    scalar_t objective{0.0};
    scalar_t primal_residual{0.0};
    scalar_t dual_residual{0.0};
    scalar_t duality_gap{0.0};
    scalar_t step_primal{0.0};
    scalar_t step_dual{0.0};
    bool restarted{false};
};

/// @brief High-resolution end-to-end timing decomposition.
struct TimingDecomposition {
    double prep_time_ms{0.0};       ///< Presolve reduction + scaling time (if end-to-end)
    double h2d_transfer_ms{0.0};    ///< Host-to-Device memory transfer time
    double pure_solve_ms{0.0};      ///< Pure solver iteration loop wall-clock time
    double d2h_transfer_ms{0.0};    ///< Device-to-Host memory transfer time
    double postsolve_ms{0.0};       ///< Solution unscaling and postsolve reconstruction time
    double verification_ms{0.0};    ///< Independent verification time
    double total_time_ms{0.0};      ///< Overall end-to-end execution time

    // GPU internal timing breakdown (via CUDA events if enabled)
    double spmv_time_ms{0.0};
    double vector_time_ms{0.0};
    double reduction_time_ms{0.0};
};

/// @brief User-configurable options controlling PDHG execution.
struct SolverConfig {
    SolverBackend backend{SolverBackend::CPU};
    int max_iterations{5000};
    double time_limit_sec{60.0};

    scalar_t primal_tol{1e-4};
    scalar_t dual_tol{1e-4};
    scalar_t gap_tol{1e-4};

    StepSizeStrategy step_size_strategy{StepSizeStrategy::POCK_CHAMBOLLE};
    RestartStrategy restart_strategy{RestartStrategy::NONE};

    int check_interval{10};         ///< Evaluate residuals and convergence every N iterations
    int restart_interval{50};       ///< Check window for adaptive restart condition

    scalar_t step_size_ratio{1.0};  ///< Initial ratio tau / sigma
    scalar_t theta{1.0};            ///< Extrapolation factor in x_bar = x + theta * (x - x_prev)
    scalar_t omega{0.99};           ///< Pock-Chambolle step-size safety damping (< 1.0)

    bool record_history{true};      ///< Store IterationLog entries
    bool verbose{false};

    static SolverConfig BaselineCPU() {
        SolverConfig c;
        c.backend = SolverBackend::CPU;
        c.step_size_strategy = StepSizeStrategy::POCK_CHAMBOLLE;
        c.restart_strategy = RestartStrategy::NONE;
        return c;
    }

    static SolverConfig AdaptiveCPU() {
        SolverConfig c;
        c.backend = SolverBackend::CPU;
        c.step_size_strategy = StepSizeStrategy::ADAPTIVE;
        c.restart_strategy = RestartStrategy::NONE;
        return c;
    }

    static SolverConfig AdaptiveRestartCPU() {
        SolverConfig c;
        c.backend = SolverBackend::CPU;
        c.step_size_strategy = StepSizeStrategy::ADAPTIVE;
        c.restart_strategy = RestartStrategy::ADAPTIVE;
        return c;
    }

    static SolverConfig BaselineCUDA() {
        SolverConfig c;
        c.backend = SolverBackend::CUDA;
        c.step_size_strategy = StepSizeStrategy::POCK_CHAMBOLLE;
        c.restart_strategy = RestartStrategy::NONE;
        return c;
    }

    static SolverConfig AdaptiveCUDA() {
        SolverConfig c;
        c.backend = SolverBackend::CUDA;
        c.step_size_strategy = StepSizeStrategy::ADAPTIVE;
        c.restart_strategy = RestartStrategy::NONE;
        return c;
    }

    static SolverConfig AdaptiveRestartCUDA() {
        SolverConfig c;
        c.backend = SolverBackend::CUDA;
        c.step_size_strategy = StepSizeStrategy::ADAPTIVE;
        c.restart_strategy = RestartStrategy::ADAPTIVE;
        return c;
    }
};

/// @brief Comprehensive output produced by the PDHG solver.
struct SolverResult {
    std::vector<scalar_t> x;            ///< Primal solution vector
    std::vector<scalar_t> y;            ///< Dual multiplier vector on constraints
    std::vector<scalar_t> s;            ///< Dual slack / reduced cost vector on variables

    scalar_t primal_objective{0.0};
    scalar_t dual_objective{0.0};
    scalar_t primal_residual{0.0};
    scalar_t dual_residual{0.0};
    scalar_t duality_gap{0.0};

    int iterations{0};
    int restarts{0};
    TerminationStatus status{TerminationStatus::MAX_ITERATIONS};
    TimingDecomposition timing;
    std::vector<IterationLog> history;

    [[nodiscard]] bool is_converged() const noexcept {
        return status == TerminationStatus::OPTIMAL;
    }

    [[nodiscard]] std::string format_summary() const {
        std::ostringstream ss;
        ss << "SolverResult: [" << to_string(status) << "]\n"
           << "  Iterations:       " << iterations << " (Restarts: " << restarts << ")\n"
           << "  Primal Objective: " << std::scientific << std::setprecision(6) << primal_objective << "\n"
           << "  Primal Residual:  " << std::scientific << std::setprecision(2) << primal_residual << "\n"
           << "  Dual Residual:    " << std::scientific << std::setprecision(2) << dual_residual << "\n"
           << "  Duality Gap:      " << std::scientific << std::setprecision(2) << duality_gap << "\n"
           << "  Pure Solve Time:  " << std::fixed << std::setprecision(2) << timing.pure_solve_ms << " ms\n"
           << "  Total Time:       " << std::fixed << std::setprecision(2) << timing.total_time_ms << " ms";
        return ss.str();
    }
};

} // namespace pipepye::solver
