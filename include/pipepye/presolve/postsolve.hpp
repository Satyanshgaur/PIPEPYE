#pragma once

#include <string>
#include <vector>
#include <memory>
#include <pipepye/core/types.hpp>
#include <pipepye/core/status.hpp>
#include <pipepye/model/lp_model.hpp>
#include <pipepye/presolve/presolve_types.hpp>

namespace pipepye::presolve {

/// @brief Type of reduction transformation recorded on the postsolve stack.
enum class PostsolveType : uint8_t {
    FixedVariable = 0,
    EmptyColumn,
    SingletonRow,
    SingletonColumn,
    RedundantRow,
    ForcedVariable
};

/// @brief Base class for an individual reversible postsolve action.
class PostsolveAction {
public:
    virtual ~PostsolveAction() = default;
    [[nodiscard]] virtual PostsolveType type() const noexcept = 0;
    virtual void postsolve(const model::LinearProgram& original_lp,
                          PrimalDualSolution& sol) const = 0;
};

/// @brief Restores a fixed variable: x_j = fixed_value, computes reduced cost.
class FixedVariableAction final : public PostsolveAction {
public:
    FixedVariableAction(index_t col_idx, scalar_t fixed_val, scalar_t cost)
        : col_idx_(col_idx), fixed_val_(fixed_val), cost_(cost) {}

    [[nodiscard]] PostsolveType type() const noexcept override {
        return PostsolveType::FixedVariable;
    }

    void postsolve(const model::LinearProgram& original_lp,
                   PrimalDualSolution& sol) const override;

private:
    index_t col_idx_;
    scalar_t fixed_val_;
    scalar_t cost_;
};

/// @brief Restores an empty column: x_j = val, s_j = cost.
class EmptyColumnAction final : public PostsolveAction {
public:
    EmptyColumnAction(index_t col_idx, scalar_t val, scalar_t cost)
        : col_idx_(col_idx), val_(val), cost_(cost) {}

    [[nodiscard]] PostsolveType type() const noexcept override {
        return PostsolveType::EmptyColumn;
    }

    void postsolve(const model::LinearProgram& original_lp,
                   PrimalDualSolution& sol) const override;

private:
    index_t col_idx_;
    scalar_t val_;
    scalar_t cost_;
};

/// @brief Restores a singleton row: assigns dual multiplier y_i if binding.
class SingletonRowAction final : public PostsolveAction {
public:
    SingletonRowAction(index_t row_idx, index_t col_idx, scalar_t coeff,
                       scalar_t row_lb, scalar_t row_ub)
        : row_idx_(row_idx), col_idx_(col_idx), coeff_(coeff),
          row_lb_(row_lb), row_ub_(row_ub) {}

    [[nodiscard]] PostsolveType type() const noexcept override {
        return PostsolveType::SingletonRow;
    }

    void postsolve(const model::LinearProgram& original_lp,
                   PrimalDualSolution& sol) const override;

private:
    index_t row_idx_;
    index_t col_idx_;
    scalar_t coeff_;
    scalar_t row_lb_;
    scalar_t row_ub_;
};

/// @brief Restores a singleton column: computes x_j from row equality/slack equation.
class SingletonColumnAction final : public PostsolveAction {
public:
    SingletonColumnAction(index_t col_idx, index_t row_idx, scalar_t coeff,
                          scalar_t cost, scalar_t rhs,
                          std::vector<std::pair<index_t, scalar_t>> other_entries)
        : col_idx_(col_idx), row_idx_(row_idx), coeff_(coeff),
          cost_(cost), rhs_(rhs), other_entries_(std::move(other_entries)) {}

    [[nodiscard]] PostsolveType type() const noexcept override {
        return PostsolveType::SingletonColumn;
    }

    void postsolve(const model::LinearProgram& original_lp,
                   PrimalDualSolution& sol) const override;

private:
    index_t col_idx_;
    index_t row_idx_;
    scalar_t coeff_;
    scalar_t cost_;
    scalar_t rhs_;
    std::vector<std::pair<index_t, scalar_t>> other_entries_;
};

/// @brief Restores a redundant row: y_i = 0.
class RedundantRowAction final : public PostsolveAction {
public:
    explicit RedundantRowAction(index_t row_idx) : row_idx_(row_idx) {}

    [[nodiscard]] PostsolveType type() const noexcept override {
        return PostsolveType::RedundantRow;
    }

    void postsolve(const model::LinearProgram& original_lp,
                   PrimalDualSolution& sol) const override;

private:
    index_t row_idx_;
};

/// @brief Manager holding the reverse postsolve stack and index mapping arrays.
class PostsolveManager {
public:
    PostsolveManager() = default;

    void set_dimensions(index_t orig_rows, index_t orig_cols) {
        orig_rows_ = orig_rows;
        orig_cols_ = orig_cols;
        orig_to_presolved_col_.assign(orig_cols, -1);
        orig_to_presolved_row_.assign(orig_rows, -1);
    }

    void add_action(std::shared_ptr<PostsolveAction> action) {
        actions_.push_back(std::move(action));
    }

    void set_col_mapping(index_t orig_col, index_t presolved_col) {
        if (orig_col >= 0 && orig_col < static_cast<index_t>(orig_to_presolved_col_.size())) {
            orig_to_presolved_col_[orig_col] = presolved_col;
        }
    }

    void set_row_mapping(index_t orig_row, index_t presolved_row) {
        if (orig_row >= 0 && orig_row < static_cast<index_t>(orig_to_presolved_row_.size())) {
            orig_to_presolved_row_[orig_row] = presolved_row;
        }
    }

    [[nodiscard]] const std::vector<index_t>& orig_to_presolved_col() const noexcept {
        return orig_to_presolved_col_;
    }

    [[nodiscard]] const std::vector<index_t>& orig_to_presolved_row() const noexcept {
        return orig_to_presolved_row_;
    }

    [[nodiscard]] size_t num_actions() const noexcept {
        return actions_.size();
    }

    /// @brief Reconstructs the original solution from the presolved solution.
    [[nodiscard]] StatusOr<PrimalDualSolution> postsolve(
        const model::LinearProgram& original_lp,
        const PrimalDualSolution& presolved_sol) const;

private:
    index_t orig_rows_{0};
    index_t orig_cols_{0};
    std::vector<index_t> orig_to_presolved_col_;
    std::vector<index_t> orig_to_presolved_row_;
    std::vector<std::shared_ptr<PostsolveAction>> actions_;
};

} // namespace pipepye::presolve
