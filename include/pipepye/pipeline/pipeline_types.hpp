#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <pipepye/core/status.hpp>

namespace pipepye::pipeline {

/// @brief Explicit ablation modes controlling which preparation transformations are applied.
enum class PipelineMode : uint8_t {
    RAW = 0,                    ///< Raw model: no presolve, no scaling. Identity solution recovery.
    PRESOLVE_ONLY,              ///< Presolve reductions only, skip scaling.
    SCALING_ONLY,               ///< Matrix equilibration/scaling only, skip presolve.
    PRESOLVE_AND_SCALING        ///< Complete pipeline: presolve reduction followed by Ruiz equilibration.
};

[[nodiscard]] inline std::string to_string(PipelineMode mode) {
    switch (mode) {
        case PipelineMode::RAW:                  return "RAW";
        case PipelineMode::PRESOLVE_ONLY:        return "PRESOLVE_ONLY";
        case PipelineMode::SCALING_ONLY:         return "SCALING_ONLY";
        case PipelineMode::PRESOLVE_AND_SCALING: return "PRESOLVE_AND_SCALING";
    }
    return "UNKNOWN";
}

[[nodiscard]] inline StatusOr<PipelineMode> parse_pipeline_mode(std::string_view str) {
    if (str == "RAW" || str == "raw") {
        return PipelineMode::RAW;
    }
    if (str == "PRESOLVE_ONLY" || str == "presolve_only" || str == "PRESOLVE" || str == "presolve") {
        return PipelineMode::PRESOLVE_ONLY;
    }
    if (str == "SCALING_ONLY" || str == "scaling_only" || str == "SCALING" || str == "scaling") {
        return PipelineMode::SCALING_ONLY;
    }
    if (str == "PRESOLVE_AND_SCALING" || str == "presolve_and_scaling" ||
        str == "FULL" || str == "full" || str == "ALL" || str == "all") {
        return PipelineMode::PRESOLVE_AND_SCALING;
    }
    return Status::InvalidArgument("Unknown pipeline mode: " + std::string(str) +
                                   ". Valid options: RAW, PRESOLVE_ONLY, SCALING_ONLY, PRESOLVE_AND_SCALING");
}

} // namespace pipepye::pipeline
