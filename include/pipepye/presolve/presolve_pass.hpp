#pragma once

#include <string>
#include <vector>
#include <memory>
#include <pipepye/core/types.hpp>
#include <pipepye/core/status.hpp>
#include <pipepye/presolve/presolve_types.hpp>
#include <pipepye/presolve/presolve_context.hpp>

namespace pipepye::presolve {

/// @brief Abstract interface for an isolated, measurable presolve reduction pass.
class PresolvePass {
public:
    virtual ~PresolvePass() = default;

    /// @brief Unique identifier name of the pass.
    [[nodiscard]] virtual std::string name() const = 0;

    /// @brief Executes the pass on the working context.
    /// @return StatusOr<PassStats> detailing reductions performed, or error/infeasibility.
    virtual StatusOr<PassStats> run(PresolveContext& ctx) = 0;
};

/// @brief Detects and removes empty rows and unconstrained empty columns.
class EmptyRowColPass final : public PresolvePass {
public:
    [[nodiscard]] std::string name() const override { return "EmptyRowColPass"; }
    StatusOr<PassStats> run(PresolveContext& ctx) override;
};

/// @brief Eliminates fixed variables (lower_bound == upper_bound) by substitution.
class FixedVariablePass final : public PresolvePass {
public:
    [[nodiscard]] std::string name() const override { return "FixedVariablePass"; }
    StatusOr<PassStats> run(PresolveContext& ctx) override;
};

/// @brief Detects and reduces singleton rows and singleton columns.
class SingletonPass final : public PresolvePass {
public:
    [[nodiscard]] std::string name() const override { return "SingletonPass"; }
    StatusOr<PassStats> run(PresolveContext& ctx) override;
};

/// @brief Identifies redundant and forcing constraints from row activity bounds.
class ForcingRedundancyPass final : public PresolvePass {
public:
    [[nodiscard]] std::string name() const override { return "ForcingRedundancyPass"; }
    StatusOr<PassStats> run(PresolveContext& ctx) override;
};

/// @brief Tightens variable bounds using constraint bounds and row activity.
class BoundTighteningPass final : public PresolvePass {
public:
    [[nodiscard]] std::string name() const override { return "BoundTighteningPass"; }
    StatusOr<PassStats> run(PresolveContext& ctx) override;
};

} // namespace pipepye::presolve
