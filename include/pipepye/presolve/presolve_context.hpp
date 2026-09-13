#pragma once

#include <vector>
#include <string>
#include <cmath>
#include <limits>
#include <memory>
#include <pipepye/core/types.hpp>
#include <pipepye/core/status.hpp>
#include <pipepye/model/lp_model.hpp>
#include <pipepye/presolve/presolve_types.hpp>
#include <pipepye/presolve/postsolve.hpp>

namespace pipepye::presolve {

struct SparseEntry {
    index_t index;     ///< Row index (if in column list) or Col index (if in row list)
    scalar_t value;    ///< Matrix non-zero coefficient A_{i,j}
};

/// @brief Mutable working context during presolve execution.
class PresolveContext {
public:
    explicit PresolveContext(const model::LinearProgram& lp, PresolveOptions options = {});

    [[nodiscard]] const model::LinearProgram& original_lp() const noexcept {
        return original_lp_;
    }

    [[nodiscard]] const PresolveOptions& options() const noexcept {
        return options_;
    }

    [[nodiscard]] PresolveStatus status() const noexcept {
        return status_;
    }

    void set_status(PresolveStatus status) noexcept {
        status_ = status;
    }

    [[nodiscard]] index_t num_cols() const noexcept { return num_cols_; }
    [[nodiscard]] index_t num_rows() const noexcept { return num_rows_; }

    [[nodiscard]] bool is_col_active(index_t j) const noexcept {
        return j >= 0 && j < num_cols_ && col_active_[j];
    }

    [[nodiscard]] bool is_row_active(index_t i) const noexcept {
        return i >= 0 && i < num_rows_ && row_active_[i];
    }

    [[nodiscard]] index_t row_degree(index_t i) const noexcept {
        return is_row_active(i) ? row_degrees_[i] : 0;
    }

    [[nodiscard]] index_t col_degree(index_t j) const noexcept {
        return is_col_active(j) ? col_degrees_[j] : 0;
    }

    [[nodiscard]] index_t num_active_rows() const noexcept { return active_row_count_; }
    [[nodiscard]] index_t num_active_cols() const noexcept { return active_col_count_; }
    [[nodiscard]] size_t num_active_nonzeros() const noexcept { return active_nnz_; }

    // Bounds and Objective Accessors
    [[nodiscard]] scalar_t col_lower(index_t j) const noexcept { return col_lb_[j]; }
    [[nodiscard]] scalar_t col_upper(index_t j) const noexcept { return col_ub_[j]; }
    [[nodiscard]] scalar_t row_lower(index_t i) const noexcept { return row_lb_[i]; }
    [[nodiscard]] scalar_t row_upper(index_t i) const noexcept { return row_ub_[i]; }
    [[nodiscard]] scalar_t cost(index_t j) const noexcept { return c_[j]; }
    [[nodiscard]] scalar_t obj_offset() const noexcept { return obj_offset_; }

    void set_col_lower(index_t j, scalar_t lb) noexcept { col_lb_[j] = lb; }
    void set_col_upper(index_t j, scalar_t ub) noexcept { col_ub_[j] = ub; }
    void set_row_lower(index_t i, scalar_t lb) noexcept { row_lb_[i] = lb; }
    void set_row_upper(index_t i, scalar_t ub) noexcept { row_ub_[i] = ub; }
    void set_cost(index_t j, scalar_t cost) noexcept { c_[j] = cost; }
    void add_obj_offset(scalar_t delta) noexcept { obj_offset_ += delta; }

    [[nodiscard]] const std::vector<SparseEntry>& row_entries(index_t i) const {
        return row_adj_[i];
    }

    [[nodiscard]] const std::vector<SparseEntry>& col_entries(index_t j) const {
        return col_adj_[j];
    }

    PostsolveManager& postsolve_mgr() noexcept { return postsolve_mgr_; }
    const PostsolveManager& postsolve_mgr() const noexcept { return postsolve_mgr_; }

    // Reductions and State Modifications
    void remove_row(index_t i);
    void remove_col(index_t j);
    void fix_variable(index_t j, scalar_t val);

    /// @brief Compacts the active state into a fresh, reduced LinearProgram.
    [[nodiscard]] model::LinearProgram to_presolved_lp();

private:
    const model::LinearProgram& original_lp_;
    PresolveOptions options_;
    PresolveStatus status_{PresolveStatus::Unchanged};

    index_t num_rows_{0};
    index_t num_cols_{0};
    index_t active_row_count_{0};
    index_t active_col_count_{0};
    size_t active_nnz_{0};

    std::vector<bool> row_active_;
    std::vector<bool> col_active_;

    std::vector<index_t> row_degrees_;
    std::vector<index_t> col_degrees_;

    std::vector<scalar_t> col_lb_;
    std::vector<scalar_t> col_ub_;
    std::vector<scalar_t> row_lb_;
    std::vector<scalar_t> row_ub_;
    std::vector<scalar_t> c_;
    scalar_t obj_offset_{0.0};

    // Adjacency representations
    std::vector<std::vector<SparseEntry>> row_adj_;
    std::vector<std::vector<SparseEntry>> col_adj_;

    PostsolveManager postsolve_mgr_;
};

} // namespace pipepye::presolve
