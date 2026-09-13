#include <pipepye/crossover/pdhg_crossover.hpp>
#include <pipepye/utils/timer.hpp>
#include <cmath>
#include <algorithm>
#include <iostream>

namespace pipepye::crossover {

CrossoverResult PDHGCrossover::run(
    const pipeline::PreparedLP& prepared_lp,
    const std::vector<scalar_t>& pdhg_x,
    const std::vector<scalar_t>& pdhg_y,
    const CrossoverConfig& config) {
    return run(prepared_lp.lp, pdhg_x, pdhg_y, config);
}

CrossoverResult PDHGCrossover::run(
    const model::LinearProgram& lp,
    const std::vector<scalar_t>& pdhg_x,
    const std::vector<scalar_t>& pdhg_y,
    const CrossoverConfig& config) {

    utils::CPUTimer total_timer;
    CrossoverResult result;

    const index_t m = lp.num_rows();
    const index_t n = lp.num_cols();
    const index_t total_vars = n + m;

    if (m == 0 || n == 0) {
        result.status = solver::TerminationStatus::OPTIMAL;
        return result;
    }

    // Compute PDHG primal objective
    scalar_t pdhg_obj = lp.obj_offset;
    for (index_t j = 0; j < n && j < static_cast<index_t>(pdhg_x.size()); ++j) {
        pdhg_obj += lp.c[j] * pdhg_x[j];
    }
    result.pdhg_objective = pdhg_obj;

    // Step 1: Active Set Detection and Variable Interiority Scoring
    utils::CPUTimer crash_timer;

    // Compute slacks s = A * x
    std::vector<scalar_t> s_val(m, 0.0);
    sparse::CSCMatrix A_csc = lp.to_csc();
    for (index_t j = 0; j < n && j < static_cast<index_t>(pdhg_x.size()); ++j) {
        auto col = A_csc.col(j);
        for (size_t k = 0; k < col.row_indices.size(); ++k) {
            s_val[col.row_indices[k]] += col.values[k] * pdhg_x[j];
        }
    }

    struct VarScore {
        index_t var;
        scalar_t interiority;
        simplex::VariableStatus status;
    };
    std::vector<VarScore> scores(total_vars);

    int active_count = 0;

    // Score structurals
    for (index_t j = 0; j < n; ++j) {
        scalar_t x_j = (j < static_cast<index_t>(pdhg_x.size())) ? pdhg_x[j] : 0.0;
        scalar_t lj = lp.col_lower[j];
        scalar_t uj = lp.col_upper[j];

        if (std::abs(lj - uj) < 1e-12) {
            scores[j] = {j, 0.0, simplex::VariableStatus::Fixed};
            active_count++;
        } else if (lj > -model::Infinity / 2 && std::abs(x_j - lj) <= config.active_tolerance * (1.0 + std::abs(lj))) {
            scores[j] = {j, 0.0, simplex::VariableStatus::AtLower};
            active_count++;
        } else if (uj < model::Infinity / 2 && std::abs(x_j - uj) <= config.active_tolerance * (1.0 + std::abs(uj))) {
            scores[j] = {j, 0.0, simplex::VariableStatus::AtUpper};
            active_count++;
        } else {
            // Interior structural variable
            scalar_t dist_l = (lj > -model::Infinity / 2) ? (x_j - lj) : model::Infinity;
            scalar_t dist_u = (uj < model::Infinity / 2) ? (uj - x_j) : model::Infinity;
            scalar_t int_dist = std::min(dist_l, dist_u);
            scores[j] = {j, int_dist, simplex::VariableStatus::Basic};
        }
    }

    // Score slacks
    for (index_t i = 0; i < m; ++i) {
        index_t sv = n + i;
        scalar_t s_i = s_val[i];
        scalar_t li = lp.row_lower[i];
        scalar_t ui = lp.row_upper[i];

        if (std::abs(li - ui) < 1e-12) {
            scores[sv] = {sv, 0.0, simplex::VariableStatus::Fixed};
            active_count++;
        } else if (li > -model::Infinity / 2 && std::abs(s_i - li) <= config.active_tolerance * (1.0 + std::abs(li))) {
            scores[sv] = {sv, 0.0, simplex::VariableStatus::AtLower};
            active_count++;
        } else if (ui < model::Infinity / 2 && std::abs(s_i - ui) <= config.active_tolerance * (1.0 + std::abs(ui))) {
            scores[sv] = {sv, 0.0, simplex::VariableStatus::AtUpper};
            active_count++;
        } else {
            // Interior slack variable
            scalar_t dist_l = (li > -model::Infinity / 2) ? (s_i - li) : model::Infinity;
            scalar_t dist_u = (ui < model::Infinity / 2) ? (ui - s_i) : model::Infinity;
            scalar_t int_dist = std::min(dist_l, dist_u);
            scores[sv] = {sv, int_dist, simplex::VariableStatus::Basic};
        }
    }
    result.active_bounds_detected = active_count;

    // Step 2: Basis Crash and Repair
    simplex::Basis crashed_basis(m, total_vars);
    std::vector<bool> row_covered(m, false);
    std::vector<index_t> selected_basis;
    selected_basis.reserve(m);

    // Collect candidate interior structurals sorted by interiority descending
    std::vector<VarScore> candidate_structurals;
    for (index_t j = 0; j < n; ++j) {
        if (scores[j].status == simplex::VariableStatus::Basic) {
            candidate_structurals.push_back(scores[j]);
        }
    }
    std::sort(candidate_structurals.begin(), candidate_structurals.end(), [](const VarScore& a, const VarScore& b) {
        return a.interiority > b.interiority;
    });

    // Greedily pick structurals that cover uncovered rows
    for (const auto& cand : candidate_structurals) {
        if (selected_basis.size() >= static_cast<size_t>(m)) break;
        auto col = A_csc.col(cand.var);
        bool covers_new_row = false;
        for (size_t k = 0; k < col.row_indices.size(); ++k) {
            index_t r = col.row_indices[k];
            if (!row_covered[r]) {
                covers_new_row = true;
                row_covered[r] = true;
            }
        }
        if (covers_new_row) {
            selected_basis.push_back(cand.var);
        }
    }

    result.structural_basic_vars = static_cast<int>(selected_basis.size());

    // Fill remaining basis positions with slack variables for uncovered rows
    for (index_t i = 0; i < m; ++i) {
        if (!row_covered[i]) {
            selected_basis.push_back(n + i);
            row_covered[i] = true;
        }
    }

    // If still under m (e.g. some structural columns covered multiple rows leaving slack slots open)
    for (index_t i = 0; i < m && selected_basis.size() < static_cast<size_t>(m); ++i) {
        index_t sv = n + i;
        if (std::find(selected_basis.begin(), selected_basis.end(), sv) == selected_basis.end()) {
            selected_basis.push_back(sv);
        }
    }

    // Truncate to exactly m if over
    if (selected_basis.size() > static_cast<size_t>(m)) {
        selected_basis.resize(m);
    }

    result.slack_basic_vars = static_cast<int>(selected_basis.size()) - result.structural_basic_vars;

    // Verify non-singularity via BasisFactorization; if singular, fall back to pure slack basis
    factorization::BasisFactorization test_fact(m);
    Status s = test_fact.factorize(A_csc, selected_basis, n, 0.1, 1e-12);
    if (!s.is_ok()) {
        // Fallback to pure slack basis
        selected_basis.clear();
        for (index_t i = 0; i < m; ++i) {
            selected_basis.push_back(n + i);
        }
        result.structural_basic_vars = 0;
        result.slack_basic_vars = m;
    }

    // Populate Basis structure
    for (index_t i = 0; i < m; ++i) {
        index_t v = selected_basis[i];
        crashed_basis.basic_vars[i] = v;
        crashed_basis.var_status[v] = simplex::VariableStatus::Basic;
        crashed_basis.var_to_basic_idx[v] = i;
    }

    for (index_t v = 0; v < total_vars; ++v) {
        if (crashed_basis.var_status[v] != simplex::VariableStatus::Basic) {
            crashed_basis.var_to_basic_idx[v] = -1;
            crashed_basis.var_status[v] = (scores[v].status != simplex::VariableStatus::Basic)
                                              ? scores[v].status
                                              : simplex::VariableStatus::AtLower;
            crashed_basis.nonbasic_vars.push_back(v);
        }
    }

    result.crash_time_ms = crash_timer.elapsed_milliseconds();
    result.initial_crashed_basis = crashed_basis;

    // Step 3: Warm-started Simplex Clean-up
    utils::CPUTimer cleanup_timer;
    simplex::DualSimplexSolver simplex_solver(config.simplex_config);
    simplex::SimplexResult sim_res = simplex_solver.solve(lp, crashed_basis);
    result.simplex_cleanup_time_ms = cleanup_timer.elapsed_milliseconds();

    result.status = sim_res.status;
    result.final_result = sim_res;
    result.final_basis = sim_res.final_basis;
    result.cleanup_pivots = sim_res.iterations;
    result.final_simplex_objective = sim_res.objective_value;
    result.objective_difference = std::abs(result.final_simplex_objective - result.pdhg_objective);
    result.total_time_ms = total_timer.elapsed_milliseconds();

    return result;
}

} // namespace pipepye::crossover
