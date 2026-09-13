#pragma once

#include <pipepye/core/types.hpp>
#include <pipepye/model/lp_model.hpp>
#include <pipepye/analysis/problem_stats.hpp>

namespace pipepye::analysis {

/// @brief Static analyzer computing comprehensive structural and geometric metrics for LP models.
class ProblemAnalyzer {
public:
    ProblemAnalyzer() = default;

    /// @brief Performs full structural analysis of the linear program.
    [[nodiscard]] static ProblemStats analyze(const model::LinearProgram& lp);
};

} // namespace pipepye::analysis
