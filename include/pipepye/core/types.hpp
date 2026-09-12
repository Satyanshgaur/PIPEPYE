#pragma once

#include <cstdint>
#include <cstddef>
#include <string>

namespace pipepye {

/// @brief Default floating-point precision for optimization mathematics (LP/MILP/QP).
/// Double precision (64-bit IEEE 754) is required for numerical stability in simplex and KKT verification.
using scalar_t = double;

/// @brief Default integer index type for matrix row/column addressing and sparse NNZ counts.
using index_t = int32_t;

/// @brief Extended 64-bit index type for very large sparse matrices (NNZ > 2^31).
using big_index_t = int64_t;

/// @brief StatusCode enumeration for operational outcomes across CPU and GPU pipelines.
enum class StatusCode {
    Success = 0,
    InvalidArgument,
    OutOfMemory,
    CudaError,
    ConvergenceFailed,
    NumericalFailure,
    NotImplemented,
    UnknownError
};

/// @brief High-level optimization solver status indicators.
enum class SolverStatus {
    NotStarted,
    Running,
    Optimal,
    Feasible,
    Infeasible,
    Unbounded,
    IterationLimit,
    TimeLimit,
    NumericalFailure
};

/// @brief Hardware execution target policy.
enum class DevicePolicy {
    CPU = 0,
    GPU = 1,
    Auto = 2
};

} // namespace pipepye
