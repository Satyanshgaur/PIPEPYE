#pragma once

#include <string>
#include <vector>
#include <sstream>
#include <iomanip>
#include <pipepye/core/types.hpp>

namespace pipepye::presolve {

struct PrimalDualSolution;

/// @brief Lifecycle status of a primal variable across presolve.
enum class VariableStatus : uint8_t {
    Retained = 0,
    Fixed,
    EliminatedEmpty,
    EliminatedSingleton,
    EliminatedForced
};

[[nodiscard]] inline std::string to_string(VariableStatus status) {
    switch (status) {
        case VariableStatus::Retained:            return "retained";
        case VariableStatus::Fixed:               return "fixed/eliminated";
        case VariableStatus::EliminatedEmpty:     return "eliminated (empty)";
        case VariableStatus::EliminatedSingleton: return "eliminated (singleton)";
        case VariableStatus::EliminatedForced:    return "eliminated (forced)";
    }
    return "unknown";
}

/// @brief Lifecycle status of a constraint row across presolve.
enum class ConstraintStatus : uint8_t {
    Retained = 0,
    EliminatedEmpty,
    EliminatedSingleton,
    EliminatedRedundant,
    EliminatedForcing
};

[[nodiscard]] inline std::string to_string(ConstraintStatus status) {
    switch (status) {
        case ConstraintStatus::Retained:            return "retained";
        case ConstraintStatus::EliminatedEmpty:     return "eliminated (empty)";
        case ConstraintStatus::EliminatedSingleton: return "eliminated (singleton)";
        case ConstraintStatus::EliminatedRedundant: return "eliminated (redundant)";
        case ConstraintStatus::EliminatedForcing:   return "eliminated (forcing)";
    }
    return "unknown";
}

/// @brief Trace record for a single primal variable.
struct VariableTrace {
    index_t orig_idx{0};
    std::string name;
    VariableStatus status{VariableStatus::Retained};
    index_t presolved_idx{-1};
    scalar_t fixed_val{0.0};
    scalar_t reconstructed_val{0.0};
    scalar_t reconstructed_reduced_cost{0.0};
    bool has_reconstructed_val{false};
    std::string rationale;
    std::vector<std::string> history;

    [[nodiscard]] std::string format_trace() const {
        std::ostringstream ss;
        ss << "original x" << orig_idx;
        if (!name.empty() && name != ("x" + std::to_string(orig_idx))) {
            ss << " (" << name << ")";
        }
        ss << " -> " << to_string(status);
        if (status == VariableStatus::Retained) {
            ss << " (presolved col " << presolved_idx << ")";
        } else if (!rationale.empty()) {
            ss << " [" << rationale << "]";
        }
        if (has_reconstructed_val) {
            ss << " -> reconstructed x" << orig_idx << " = " << std::setprecision(6) << reconstructed_val;
        }
        return ss.str();
    }
};

/// @brief Trace record for a single constraint row.
struct ConstraintTrace {
    index_t orig_idx{0};
    std::string name;
    ConstraintStatus status{ConstraintStatus::Retained};
    index_t presolved_idx{-1};
    scalar_t reconstructed_dual{0.0};
    bool has_reconstructed_dual{false};
    std::string rationale;
    std::vector<std::string> history;

    [[nodiscard]] std::string format_trace() const {
        std::ostringstream ss;
        ss << "original c" << orig_idx;
        if (!name.empty() && name != ("c" + std::to_string(orig_idx))) {
            ss << " (" << name << ")";
        }
        ss << " -> " << to_string(status);
        if (status == ConstraintStatus::Retained) {
            ss << " (presolved row " << presolved_idx << ")";
        } else if (!rationale.empty()) {
            ss << " [" << rationale << "]";
        }
        if (has_reconstructed_dual) {
            ss << " -> reconstructed y" << orig_idx << " = " << std::setprecision(6) << reconstructed_dual;
        }
        return ss.str();
    }
};

/// @brief Structured transformation log tracking reductions from original to presolved to postsolved models.
class TransformationLog {
public:
    TransformationLog() = default;

    void init(index_t num_rows, index_t num_cols,
              const std::vector<std::string>& row_names = {},
              const std::vector<std::string>& col_names = {}) {
        vars_.clear();
        vars_.resize(num_cols);
        for (index_t j = 0; j < num_cols; ++j) {
            vars_[j].orig_idx = j;
            vars_[j].name = (j < static_cast<index_t>(col_names.size()) && !col_names[j].empty())
                            ? col_names[j] : ("x" + std::to_string(j));
            vars_[j].status = VariableStatus::Retained;
            vars_[j].presolved_idx = j;
        }

        constrs_.clear();
        constrs_.resize(num_rows);
        for (index_t i = 0; i < num_rows; ++i) {
            constrs_[i].orig_idx = i;
            constrs_[i].name = (i < static_cast<index_t>(row_names.size()) && !row_names[i].empty())
                               ? row_names[i] : ("c" + std::to_string(i));
            constrs_[i].status = ConstraintStatus::Retained;
            constrs_[i].presolved_idx = i;
        }
    }

    void record_var_fixed(index_t orig_j, scalar_t val, const std::string& reason) {
        if (orig_j >= 0 && orig_j < static_cast<index_t>(vars_.size())) {
            vars_[orig_j].status = VariableStatus::Fixed;
            vars_[orig_j].fixed_val = val;
            vars_[orig_j].presolved_idx = -1;
            vars_[orig_j].rationale = reason;
            vars_[orig_j].history.push_back("Fixed to " + std::to_string(val) + ": " + reason);
        }
    }

    void record_var_eliminated(index_t orig_j, VariableStatus status, const std::string& reason) {
        if (orig_j >= 0 && orig_j < static_cast<index_t>(vars_.size())) {
            vars_[orig_j].status = status;
            vars_[orig_j].presolved_idx = -1;
            vars_[orig_j].rationale = reason;
            vars_[orig_j].history.push_back("Eliminated (" + to_string(status) + "): " + reason);
        }
    }

    void record_var_bound_tightened(index_t orig_j, scalar_t new_lb, scalar_t new_ub, const std::string& reason) {
        if (orig_j >= 0 && orig_j < static_cast<index_t>(vars_.size())) {
            vars_[orig_j].history.push_back(
                "Bound tightened to [" + std::to_string(new_lb) + ", " + std::to_string(new_ub) + "]: " + reason);
        }
    }

    void record_var_mapping(index_t orig_j, index_t pre_j) {
        if (orig_j >= 0 && orig_j < static_cast<index_t>(vars_.size())) {
            vars_[orig_j].presolved_idx = pre_j;
            if (pre_j == -1 && vars_[orig_j].status == VariableStatus::Retained) {
                vars_[orig_j].status = VariableStatus::Fixed;
            }
        }
    }

    void record_row_eliminated(index_t orig_i, ConstraintStatus status, const std::string& reason) {
        if (orig_i >= 0 && orig_i < static_cast<index_t>(constrs_.size())) {
            constrs_[orig_i].status = status;
            constrs_[orig_i].presolved_idx = -1;
            constrs_[orig_i].rationale = reason;
            constrs_[orig_i].history.push_back("Eliminated (" + to_string(status) + "): " + reason);
        }
    }

    void record_row_mapping(index_t orig_i, index_t pre_i) {
        if (orig_i >= 0 && orig_i < static_cast<index_t>(constrs_.size())) {
            constrs_[orig_i].presolved_idx = pre_i;
            if (pre_i == -1 && constrs_[orig_i].status == ConstraintStatus::Retained) {
                constrs_[orig_i].status = ConstraintStatus::EliminatedRedundant;
            }
        }
    }

    void record_reconstructed_variable(index_t orig_j, scalar_t x_val, scalar_t s_val) {
        if (orig_j >= 0 && orig_j < static_cast<index_t>(vars_.size())) {
            vars_[orig_j].reconstructed_val = x_val;
            vars_[orig_j].reconstructed_reduced_cost = s_val;
            vars_[orig_j].has_reconstructed_val = true;
        }
    }

    void record_reconstructed_constraint(index_t orig_i, scalar_t y_val) {
        if (orig_i >= 0 && orig_i < static_cast<index_t>(constrs_.size())) {
            constrs_[orig_i].reconstructed_dual = y_val;
            constrs_[orig_i].has_reconstructed_dual = true;
        }
    }

    [[nodiscard]] const VariableTrace& variable_trace(index_t orig_j) const {
        return vars_.at(orig_j);
    }

    [[nodiscard]] const ConstraintTrace& constraint_trace(index_t orig_i) const {
        return constrs_.at(orig_i);
    }

    [[nodiscard]] std::string format_variable_trace(index_t orig_j) const {
        if (orig_j < 0 || orig_j >= static_cast<index_t>(vars_.size())) {
            return "original x" + std::to_string(orig_j) + " -> invalid index";
        }
        return vars_[orig_j].format_trace();
    }

    [[nodiscard]] std::string format_constraint_trace(index_t orig_i) const {
        if (orig_i < 0 || orig_i >= static_cast<index_t>(constrs_.size())) {
            return "original c" + std::to_string(orig_i) + " -> invalid index";
        }
        return constrs_[orig_i].format_trace();
    }

    [[nodiscard]] std::string format_full_log() const {
        std::ostringstream ss;
        ss << "=== PIPEPYE PRESOLVE TRANSFORMATION LOG ===\n";
        ss << "--- Variable Transformations (" << vars_.size() << " total) ---\n";
        for (const auto& vt : vars_) {
            ss << "  " << vt.format_trace() << "\n";
        }
        ss << "--- Constraint Transformations (" << constrs_.size() << " total) ---\n";
        for (const auto& ct : constrs_) {
            ss << "  " << ct.format_trace() << "\n";
        }
        ss << "==========================================\n";
        return ss.str();
    }

    [[nodiscard]] const std::vector<VariableTrace>& all_variables() const noexcept { return vars_; }
    [[nodiscard]] const std::vector<ConstraintTrace>& all_constraints() const noexcept { return constrs_; }

private:
    std::vector<VariableTrace> vars_;
    std::vector<ConstraintTrace> constrs_;
};

} // namespace pipepye::presolve
