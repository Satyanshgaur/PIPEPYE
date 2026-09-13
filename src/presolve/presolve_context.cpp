#include <pipepye/presolve/presolve_context.hpp>
#include <iostream>
#include <algorithm>

namespace pipepye::presolve {

PresolveContext::PresolveContext(const model::LinearProgram& lp, PresolveOptions options)
    : original_lp_(lp), options_(options),
      num_rows_(lp.num_rows()), num_cols_(lp.num_cols()),
      active_row_count_(lp.num_rows()), active_col_count_(lp.num_cols()),
      active_nnz_(0),
      row_active_(lp.num_rows(), true),
      col_active_(lp.num_cols(), true),
      row_degrees_(lp.num_rows(), 0),
      col_degrees_(lp.num_cols(), 0),
      col_lb_(lp.col_lower),
      col_ub_(lp.col_upper),
      row_lb_(lp.row_lower),
      row_ub_(lp.row_upper),
      c_(lp.c),
      obj_offset_(lp.obj_offset) {

    row_adj_.resize(num_rows_);
    col_adj_.resize(num_cols_);
    postsolve_mgr_.set_dimensions(num_rows_, num_cols_);
    postsolve_mgr_.log().init(num_rows_, num_cols_, lp.row_names, lp.col_names);

    if (!lp.csr_row_ptr.empty()) {
        for (index_t i = 0; i < num_rows_; ++i) {
            index_t start = lp.csr_row_ptr[i];
            index_t end = lp.csr_row_ptr[i + 1];
            for (index_t p = start; p < end; ++p) {
                index_t j = lp.csr_col_ind[p];
                scalar_t val = lp.csr_values[p];
                if (std::abs(val) > 0.0) {
                    row_adj_[i].push_back({j, val});
                    col_adj_[j].push_back({i, val});
                    ++active_nnz_;
                }
            }
            row_degrees_[i] = static_cast<index_t>(row_adj_[i].size());
        }
        for (index_t j = 0; j < num_cols_; ++j) {
            col_degrees_[j] = static_cast<index_t>(col_adj_[j].size());
        }
    } else if (lp.A_coo.num_nonzeros() > 0) {
        for (const auto& tr : lp.A_coo.triplets()) {
            index_t i = tr.row;
            index_t j = tr.col;
            scalar_t val = tr.val;
            if (std::abs(val) > 0.0) {
                row_adj_[i].push_back({j, val});
                col_adj_[j].push_back({i, val});
                ++active_nnz_;
            }
        }
        for (index_t i = 0; i < num_rows_; ++i) {
            row_degrees_[i] = static_cast<index_t>(row_adj_[i].size());
        }
        for (index_t j = 0; j < num_cols_; ++j) {
            col_degrees_[j] = static_cast<index_t>(col_adj_[j].size());
        }
    }

    // Validate bounds: detect immediate infeasibility
    for (index_t j = 0; j < num_cols_; ++j) {
        if (col_lb_[j] > col_ub_[j] + options_.tolerance) {
            status_ = PresolveStatus::Infeasible;
            return;
        }
    }
    for (index_t i = 0; i < num_rows_; ++i) {
        if (row_lb_[i] > row_ub_[i] + options_.tolerance) {
            status_ = PresolveStatus::Infeasible;
            return;
        }
    }
}

void PresolveContext::remove_row(index_t i) {
    if (i < 0 || i >= num_rows_ || !row_active_[i]) return;
    row_active_[i] = false;
    --active_row_count_;

    for (const auto& entry : row_adj_[i]) {
        index_t j = entry.index;
        if (col_active_[j]) {
            if (col_degrees_[j] > 0) {
                --col_degrees_[j];
            }
            if (active_nnz_ > 0) {
                --active_nnz_;
            }
        }
    }
    row_degrees_[i] = 0;
}

void PresolveContext::remove_col(index_t j) {
    if (j < 0 || j >= num_cols_ || !col_active_[j]) return;
    col_active_[j] = false;
    --active_col_count_;

    for (const auto& entry : col_adj_[j]) {
        index_t i = entry.index;
        if (row_active_[i]) {
            if (row_degrees_[i] > 0) {
                --row_degrees_[i];
            }
            if (active_nnz_ > 0) {
                --active_nnz_;
            }
        }
    }
    col_degrees_[j] = 0;
}

void PresolveContext::fix_variable(index_t j, scalar_t val) {
    if (j < 0 || j >= num_cols_ || !col_active_[j]) return;

    col_lb_[j] = val;
    col_ub_[j] = val;
    obj_offset_ += c_[j] * val;

    for (const auto& entry : col_adj_[j]) {
        index_t i = entry.index;
        if (row_active_[i]) {
            scalar_t delta = entry.value * val;
            if (std::abs(row_lb_[i]) < options_.infinity_threshold) {
                row_lb_[i] -= delta;
            }
            if (std::abs(row_ub_[i]) < options_.infinity_threshold) {
                row_ub_[i] -= delta;
            }
        }
    }

    postsolve_mgr_.log().record_var_fixed(j, val, "Fixed to bound value");
    postsolve_mgr_.add_action(std::make_unique<FixedVariableAction>(j, val, c_[j]));
    remove_col(j);
}

model::LinearProgram PresolveContext::to_presolved_lp() {
    model::LinearProgram pre_lp;
    pre_lp.name = original_lp_.name + "_presolved";
    pre_lp.is_maximization = original_lp_.is_maximization;
    pre_lp.obj_name = original_lp_.obj_name;
    pre_lp.obj_offset = obj_offset_;

    // 1. Build contiguous mapping for active columns
    std::vector<index_t> orig_to_new_col(num_cols_, -1);
    index_t new_col_idx = 0;
    for (index_t j = 0; j < num_cols_; ++j) {
        if (col_active_[j]) {
            orig_to_new_col[j] = new_col_idx++;
            postsolve_mgr_.set_col_mapping(j, orig_to_new_col[j]);
            postsolve_mgr_.log().record_var_mapping(j, orig_to_new_col[j]);

            pre_lp.col_names.push_back(original_lp_.col_names[j]);
            pre_lp.col_name_to_idx[original_lp_.col_names[j]] = orig_to_new_col[j];
            pre_lp.c.push_back(c_[j]);
            pre_lp.col_lower.push_back(col_lb_[j]);
            pre_lp.col_upper.push_back(col_ub_[j]);
            pre_lp.var_types.push_back(original_lp_.var_types[j]);
        } else {
            postsolve_mgr_.set_col_mapping(j, -1);
            postsolve_mgr_.log().record_var_mapping(j, -1);
        }
    }

    // 2. Build contiguous mapping for active rows
    std::vector<index_t> orig_to_new_row(num_rows_, -1);
    index_t new_row_idx = 0;
    for (index_t i = 0; i < num_rows_; ++i) {
        if (row_active_[i]) {
            orig_to_new_row[i] = new_row_idx++;
            postsolve_mgr_.set_row_mapping(i, orig_to_new_row[i]);
            postsolve_mgr_.log().record_row_mapping(i, orig_to_new_row[i]);

            pre_lp.row_names.push_back(original_lp_.row_names[i]);
            pre_lp.row_name_to_idx[original_lp_.row_names[i]] = orig_to_new_row[i];
            pre_lp.row_lower.push_back(row_lb_[i]);
            pre_lp.row_upper.push_back(row_ub_[i]);
            pre_lp.row_senses.push_back(original_lp_.row_senses[i]);
        } else {
            postsolve_mgr_.set_row_mapping(i, -1);
            postsolve_mgr_.log().record_row_mapping(i, -1);
        }
    }

    // 3. Assemble compact COO matrix
    pipepye::sparse::COOMatrix coo(new_row_idx, new_col_idx);
    for (index_t i = 0; i < num_rows_; ++i) {
        if (!row_active_[i]) continue;
        index_t mapped_i = orig_to_new_row[i];
        for (const auto& entry : row_adj_[i]) {
            index_t j = entry.index;
            if (col_active_[j]) {
                index_t mapped_j = orig_to_new_col[j];
                coo.add_entry(mapped_i, mapped_j, entry.value);
            }
        }
    }
    pre_lp.A_coo = coo;

    // 4. Generate dual CSR and CSC representations
    auto csr = coo.to_csr();
    pre_lp.csr_row_ptr.assign(csr.row_ptr().begin(), csr.row_ptr().end());
    pre_lp.csr_col_ind.assign(csr.col_ind().begin(), csr.col_ind().end());
    pre_lp.csr_values.assign(csr.values().begin(), csr.values().end());

    auto csc = coo.to_csc();
    pre_lp.csc_col_ptr.assign(csc.col_ptr().begin(), csc.col_ptr().end());
    pre_lp.csc_row_ind.assign(csc.row_ind().begin(), csc.row_ind().end());
    pre_lp.csc_values.assign(csc.values().begin(), csc.values().end());

    return pre_lp;
}

} // namespace pipepye::presolve
