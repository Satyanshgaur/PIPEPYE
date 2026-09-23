#include <pipepye/milp/branch_and_bound.hpp>
#include <pipepye/factorization/basis_factorization.hpp>
#include <pipepye/utils/timer.hpp>
#include <cmath>
#include <queue>
#include <stack>
#include <algorithm>
#include <iostream>

namespace pipepye::milp {

namespace {

MILPTerminationStatus map_simplex_status(solver::TerminationStatus status) {
    switch (status) {
        case solver::TerminationStatus::OPTIMAL: return MILPTerminationStatus::Optimal;
        case solver::TerminationStatus::PRIMAL_INFEASIBLE: return MILPTerminationStatus::Infeasible;
        case solver::TerminationStatus::TIME_LIMIT: return MILPTerminationStatus::TimeLimit;
        case solver::TerminationStatus::MAX_ITERATIONS: return MILPTerminationStatus::NodeLimit;
        default: return MILPTerminationStatus::NumericalFailure;
    }
}

struct BBNode {
    int id{0};
    int depth{0};
    scalar_t bound{0.0};
    scalar_t estimate{0.0};
    std::vector<scalar_t> col_lower;
    std::vector<scalar_t> col_upper;
    std::optional<simplex::Basis> basis;
    index_t branched_var{-1};
    bool is_up_branch{false};
    scalar_t branch_frac{0.0};
};

struct BestBoundComparator {
    bool operator()(const BBNode& a, const BBNode& b) const noexcept {
        return a.bound > b.bound; // Min-heap by bound
    }
};

struct BestEstimateComparator {
    bool operator()(const BBNode& a, const BBNode& b) const noexcept {
        return a.estimate > b.estimate; // Min-heap by estimate
    }
};

} // namespace

BranchAndBoundSolver::BranchAndBoundSolver(MILPConfig config)
    : config_(config) {}

Status BranchAndBoundSolver::apply_root_presolve(model::LinearProgram& lp) {
    index_t n = lp.num_cols();
    index_t m = lp.num_rows();

    // 1. Integer bound tightening: l_j = ceil(l_j), u_j = floor(u_j)
    for (index_t j = 0; j < n; ++j) {
        if (j < static_cast<index_t>(lp.var_types.size())) {
            auto vt = lp.var_types[j];
            if (vt == model::VariableType::Binary) {
                lp.col_lower[j] = std::max(0.0, std::ceil(lp.col_lower[j] - 1e-7));
                lp.col_upper[j] = std::min(1.0, std::floor(lp.col_upper[j] + 1e-7));
            } else if (vt == model::VariableType::Integer) {
                if (lp.col_lower[j] > -model::Infinity / 2.0) {
                    lp.col_lower[j] = std::ceil(lp.col_lower[j] - 1e-7);
                }
                if (lp.col_upper[j] < model::Infinity / 2.0) {
                    lp.col_upper[j] = std::floor(lp.col_upper[j] + 1e-7);
                }
            }
            if (lp.col_lower[j] > lp.col_upper[j] + 1e-6) {
                return Status::InvalidArgument("Integer bounds conflict: lower > upper");
            }
        }
    }

    // 2. Redundant singleton row analysis
    auto csr = lp.to_csr();
    for (index_t i = 0; i < m; ++i) {
        size_t row_len = csr.row_ptr()[i + 1] - csr.row_ptr()[i];
        if (row_len == 1) {
            index_t j = csr.col_ind()[csr.row_ptr()[i]];
            scalar_t a_ij = csr.values()[csr.row_ptr()[i]];
            if (std::abs(a_ij) > 1e-9) {
                scalar_t r_low = lp.row_lower[i];
                scalar_t r_up = lp.row_upper[i];

                if (a_ij > 0.0) {
                    if (r_up < model::Infinity / 2.0) {
                        scalar_t new_up = r_up / a_ij;
                        if (lp.var_types[j] != model::VariableType::Continuous) new_up = std::floor(new_up + 1e-7);
                        lp.col_upper[j] = std::min(lp.col_upper[j], new_up);
                    }
                    if (r_low > -model::Infinity / 2.0) {
                        scalar_t new_low = r_low / a_ij;
                        if (lp.var_types[j] != model::VariableType::Continuous) new_low = std::ceil(new_low - 1e-7);
                        lp.col_lower[j] = std::max(lp.col_lower[j], new_low);
                    }
                } else {
                    if (r_low > -model::Infinity / 2.0) {
                        scalar_t new_up = r_low / a_ij;
                        if (lp.var_types[j] != model::VariableType::Continuous) new_up = std::floor(new_up + 1e-7);
                        lp.col_upper[j] = std::min(lp.col_upper[j], new_up);
                    }
                    if (r_up < model::Infinity / 2.0) {
                        scalar_t new_low = r_up / a_ij;
                        if (lp.var_types[j] != model::VariableType::Continuous) new_low = std::ceil(new_low - 1e-7);
                        lp.col_lower[j] = std::max(lp.col_lower[j], new_low);
                    }
                }
                if (lp.col_lower[j] > lp.col_upper[j] + 1e-6) {
                    return Status::InvalidArgument("Singleton constraint forces infeasible bounds");
                }
            }
        }
    }

    return Status::OK();
}

int BranchAndBoundSolver::generate_gomory_cuts(
    model::LinearProgram& lp,
    const simplex::SimplexResult& root_res,
    const std::vector<index_t>& int_vars,
    int max_cuts) {
    if (max_cuts <= 0 || !root_res.is_optimal()) return 0;

    index_t m = lp.num_rows();
    index_t n = lp.num_cols();
    if (m == 0 || n == 0) return 0;

    const auto& basis = root_res.final_basis;
    if (basis.basic_vars.size() != static_cast<size_t>(m)) return 0;

    // Factorize basis matrix B to compute tableau rows
    auto A_csc = lp.to_csc();
    factorization::BasisFactorization basis_fact(m);
    Status st = basis_fact.factorize(A_csc, basis.basic_vars, n);
    if (!st.is_ok()) return 0;

    auto A_csr = lp.to_csr();
    int cuts_generated = 0;

    for (index_t i = 0; i < m && cuts_generated < max_cuts; ++i) {
        index_t var_k = basis.basic_vars[i];
        if (var_k >= n) continue; // Basic variable is slack
        if (std::find(int_vars.begin(), int_vars.end(), var_k) == int_vars.end()) continue;

        scalar_t x_val = root_res.x[var_k];
        scalar_t f0 = x_val - std::floor(x_val);
        if (f0 < 0.05 || f0 > 0.95) continue; // Sufficiently fractional

        // Solve B^T pi = e_i (BTRAN)
        std::vector<scalar_t> e_i(m, 0.0);
        e_i[i] = 1.0;
        std::vector<scalar_t> pi = basis_fact.solve_btran(e_i);

        // Standard GMI cut on nonbasic variables:
        // sum_{v in N} gamma_v * Delta_v >= f0 * (1 - f0)
        std::vector<scalar_t> C(n, 0.0);
        scalar_t cut_rhs = f0 * (1.0 - f0);

        for (index_t v : basis.nonbasic_vars) {
            auto status = basis.var_status[v];
            if (status != simplex::VariableStatus::AtLower && status != simplex::VariableStatus::AtUpper) {
                continue;
            }

            scalar_t a_bar = 0.0;
            if (v < n) {
                auto col = A_csc.col(v);
                for (size_t p = 0; p < col.row_indices.size(); ++p) {
                    a_bar += col.values[p] * pi[col.row_indices[p]];
                }
            } else {
                index_t slack_r = v - n;
                a_bar = -pi[slack_r];
            }

            scalar_t alpha = (status == simplex::VariableStatus::AtLower) ? a_bar : -a_bar;

            bool is_int = false;
            if (v < n) {
                is_int = (std::find(int_vars.begin(), int_vars.end(), v) != int_vars.end());
            }

            scalar_t gamma = 0.0;
            if (is_int) {
                scalar_t fj = alpha - std::floor(alpha);
                if (fj <= f0) {
                    gamma = fj * (1.0 - f0);
                } else {
                    gamma = (1.0 - fj) * f0;
                }
            } else {
                if (alpha >= 0.0) {
                    gamma = alpha * (1.0 - f0);
                } else {
                    gamma = -alpha * f0;
                }
            }

            if (std::abs(gamma) < 1e-7) continue;

            if (v < n) {
                if (status == simplex::VariableStatus::AtLower) {
                    C[v] += gamma;
                    cut_rhs += gamma * lp.col_lower[v];
                } else {
                    C[v] -= gamma;
                    cut_rhs -= gamma * lp.col_upper[v];
                }
            } else {
                index_t k = v - n;
                if (status == simplex::VariableStatus::AtLower) {
                    for (index_t p = A_csr.row_ptr()[k]; p < A_csr.row_ptr()[k + 1]; ++p) {
                        C[A_csr.col_ind()[p]] += gamma * A_csr.values()[p];
                    }
                    cut_rhs += gamma * lp.row_lower[k];
                } else {
                    for (index_t p = A_csr.row_ptr()[k]; p < A_csr.row_ptr()[k + 1]; ++p) {
                        C[A_csr.col_ind()[p]] -= gamma * A_csr.values()[p];
                    }
                    cut_rhs -= gamma * lp.row_upper[k];
                }
            }
        }

        scalar_t norm_sq = 0.0;
        int nz_count = 0;
        for (index_t j = 0; j < n; ++j) {
            if (std::abs(C[j]) > 1e-7) {
                norm_sq += C[j] * C[j];
                nz_count++;
            }
        }

        if (nz_count >= 1 && norm_sq > 1e-12) {
            scalar_t scale = 1.0 / std::sqrt(norm_sq);
            index_t new_row = lp.num_rows();
            std::string cut_name = "GOMORY_" + std::to_string(cuts_generated + 1);
            lp.row_names.push_back(cut_name);
            lp.row_name_to_idx[cut_name] = new_row;
            lp.row_senses.push_back(model::RowSense::GreaterEqual);
            lp.row_lower.push_back(cut_rhs * scale);
            lp.row_upper.push_back(model::Infinity);
            lp.A_coo.set_dimensions(new_row + 1, n);

            for (index_t j = 0; j < n; ++j) {
                if (std::abs(C[j]) > 1e-7) {
                    lp.A_coo.add_entry(new_row, j, C[j] * scale);
                }
            }

            cuts_generated++;
        }
    }

    if (cuts_generated > 0) {
        auto csc = lp.A_coo.to_csc();
        lp.csc_col_ptr = csc.col_ptr_vector();
        lp.csc_row_ind = csc.row_ind_vector();
        lp.csc_values = csc.values_vector();
        auto csr = lp.A_coo.to_csr();
        lp.csr_row_ptr = csr.row_ptr_vector();
        lp.csr_col_ind = csr.col_ind_vector();
        lp.csr_values = csr.values_vector();
    }

    return cuts_generated;
}

std::optional<std::vector<scalar_t>> BranchAndBoundSolver::run_simple_rounding(
    const model::LinearProgram& lp,
    const std::vector<scalar_t>& x_lp,
    const std::vector<index_t>& int_vars) {
    index_t n = lp.num_cols();
    index_t m = lp.num_rows();
    if (x_lp.size() < static_cast<size_t>(n)) return std::nullopt;

    std::vector<scalar_t> x_round = x_lp;
    for (index_t j : int_vars) {
        scalar_t rounded = std::round(x_lp[j]);
        rounded = std::clamp(rounded, lp.col_lower[j], lp.col_upper[j]);
        x_round[j] = rounded;
    }

    // Verify row feasibility
    auto csr = lp.to_csr();
    for (index_t i = 0; i < m; ++i) {
        scalar_t ax = 0.0;
        for (index_t p = csr.row_ptr()[i]; p < csr.row_ptr()[i + 1]; ++p) {
            ax += csr.values()[p] * x_round[csr.col_ind()[p]];
        }
        if (ax < lp.row_lower[i] - 1e-4 || ax > lp.row_upper[i] + 1e-4) {
            return std::nullopt; // Constraint violated
        }
    }

    return x_round;
}

std::optional<std::vector<scalar_t>> BranchAndBoundSolver::run_diving_heuristic(
    const model::LinearProgram& lp,
    const std::vector<scalar_t>& x_lp,
    const std::vector<index_t>& int_vars,
    simplex::DualSimplexSolver& simplex,
    const std::optional<simplex::Basis>& basis,
    int max_depth) {
    model::LinearProgram lp_dive = lp;
    std::vector<scalar_t> cur_x = x_lp;
    std::optional<simplex::Basis> cur_basis = basis;

    for (int step = 0; step < max_depth; ++step) {
        index_t best_j = -1;
        scalar_t min_frac_dist = 1.0;
        scalar_t target_val = 0.0;

        for (index_t j : int_vars) {
            scalar_t val = cur_x[j];
            scalar_t dist = std::abs(val - std::round(val));
            if (dist > 1e-4 && dist < min_frac_dist) {
                min_frac_dist = dist;
                best_j = j;
                target_val = std::clamp(std::round(val), lp_dive.col_lower[j], lp_dive.col_upper[j]);
            }
        }

        if (best_j == -1) {
            // All integer variables satisfied
            return cur_x;
        }

        // Fix chosen variable to rounded target
        lp_dive.col_lower[best_j] = target_val;
        lp_dive.col_upper[best_j] = target_val;

        auto res = simplex.solve(lp_dive, cur_basis);
        if (!res.is_optimal()) {
            break; // Infeasible dive branch
        }

        cur_x = res.x;
        cur_basis = res.final_basis;
    }

    // Final check on cur_x
    for (index_t j : int_vars) {
        if (std::abs(cur_x[j] - std::round(cur_x[j])) > 1e-4) {
            return std::nullopt;
        }
    }

    return cur_x;
}

MILPResult BranchAndBoundSolver::solve(const model::LinearProgram& milp_model) {
    MILPResult result;
    utils::CPUTimer timer;

    index_t n = milp_model.num_cols();
    model::LinearProgram lp = milp_model;

    // 1. Identify integer & binary variables
    std::vector<index_t> int_vars;
    for (index_t j = 0; j < n; ++j) {
        if (j < static_cast<index_t>(lp.var_types.size())) {
            auto vt = lp.var_types[j];
            if (vt == model::VariableType::Binary || vt == model::VariableType::Integer) {
                int_vars.push_back(j);
            }
        }
    }

    // 2. Root Presolve
    if (config_.enable_root_presolve && !int_vars.empty()) {
        Status presolve_status = apply_root_presolve(lp);
        if (!presolve_status.is_ok()) {
            result.status = MILPTerminationStatus::Infeasible;
            result.total_time_ms = timer.elapsed_milliseconds();
            return result;
        }
    }

    simplex::DualSimplexSolver simplex(config_.simplex_config);

    // If pure continuous LP, solve directly
    if (int_vars.empty()) {
        auto lp_res = simplex.solve(lp, std::nullopt);
        result.status = map_simplex_status(lp_res.status);
        result.best_objective = lp_res.objective_value;
        result.best_bound = lp_res.objective_value;
        result.mip_gap = 0.0;
        result.x = lp_res.x;
        result.nodes_explored = 1;
        result.nodes_created = 1;
        result.total_lp_solves = 1;
        result.total_pivots = lp_res.iterations;
        result.root_objective = lp_res.objective_value;
        result.root_relaxation_time_ms = lp_res.timing.total_time_ms;
        result.time_to_first_incumbent_ms = lp_res.timing.total_time_ms;
        result.total_time_ms = timer.elapsed_milliseconds();
        return result;
    }

    // 3. Solve Root LP Relaxation
    utils::CPUTimer root_timer;
    auto root_res = simplex.solve(lp, std::nullopt);
    result.root_relaxation_time_ms = root_timer.elapsed_milliseconds();
    result.lp_relaxation_time_ms += result.root_relaxation_time_ms;
    result.total_lp_solves++;
    result.total_pivots += root_res.iterations;
    if (config_.use_warm_start) {
        result.warm_start_pivots += root_res.iterations;
    } else {
        result.cold_start_pivots += root_res.iterations;
    }

    if (!root_res.is_optimal()) {
        result.status = map_simplex_status(root_res.status);
        result.total_time_ms = timer.elapsed_milliseconds();
        return result;
    }

    result.root_objective = root_res.objective_value;
    result.best_bound = root_res.objective_value;

    // 4. Cutting Planes at Root Node (Gomory Fractional Cuts)
    if (config_.cutting_strategy == CuttingPlaneStrategy::GomoryFractional && config_.max_root_cuts > 0) {
        utils::CPUTimer cut_timer;
        int cuts = generate_gomory_cuts(lp, root_res, int_vars, config_.max_root_cuts);
        result.cut_generation_time_ms = cut_timer.elapsed_milliseconds();
        if (cuts > 0) {
            result.root_cuts_added = cuts;
            // Re-solve with cuts
            auto cut_res = simplex.solve(lp, root_res.final_basis);
            result.total_lp_solves++;
            result.total_pivots += cut_res.iterations;
            if (cut_res.is_optimal()) {
                root_res = cut_res;
                result.root_bound_after_cuts = cut_res.objective_value;
                result.best_bound = std::max(result.best_bound, cut_res.objective_value);
            }
        }
    }

    // Helper: compute objective function value directly
    auto eval_obj = [&](const std::vector<scalar_t>& x_cand) {
        scalar_t obj = lp.obj_offset;
        for (index_t j = 0; j < n; ++j) {
            obj += lp.c[j] * x_cand[j];
        }
        return obj;
    };

    // Helper: check integer fractionality
    auto find_fractional = [&](const std::vector<scalar_t>& x) -> std::pair<index_t, scalar_t> {
        index_t best_var = -1;
        scalar_t max_frac = 0.0;
        for (index_t j : int_vars) {
            scalar_t val = x[j];
            scalar_t dist = std::abs(val - std::round(val));
            if (dist > config_.integrality_tol) {
                if (config_.branching_strategy == BranchingStrategy::FirstFractional) {
                    return {j, dist};
                }
                if (dist > max_frac) {
                    max_frac = dist;
                    best_var = j;
                }
            }
        }
        return {best_var, max_frac};
    };

    // 5. Primal Heuristics at Root
    scalar_t incumbent_obj = model::Infinity;
    std::vector<scalar_t> incumbent_x;

    auto update_incumbent = [&](const std::vector<scalar_t>& cand_x, scalar_t cand_obj, int at_node) {
        if (cand_obj < incumbent_obj - 1e-7) {
            incumbent_obj = cand_obj;
            incumbent_x = cand_x;
            double now_ms = timer.elapsed_milliseconds();
            if (result.time_to_first_incumbent_ms == 0.0) {
                result.time_to_first_incumbent_ms = now_ms;
            }

            scalar_t cur_gap = std::abs(incumbent_obj - result.best_bound) / std::max(1.0, std::abs(incumbent_obj));
            result.gap_history.push_back({now_ms, at_node, result.best_bound, incumbent_obj, cur_gap});

            if (cur_gap <= 0.10 && result.time_to_gap_10pct_ms == 0.0) result.time_to_gap_10pct_ms = now_ms;
            if (cur_gap <= 0.01 && result.time_to_gap_1pct_ms == 0.0) result.time_to_gap_1pct_ms = now_ms;
            if (cur_gap <= 0.001 && result.time_to_gap_01pct_ms == 0.0) result.time_to_gap_01pct_ms = now_ms;
            return true;
        }
        return false;
    };

    if (config_.heuristic_strategy == HeuristicStrategy::SimpleRounding || config_.heuristic_strategy == HeuristicStrategy::All) {
        utils::CPUTimer h_timer;
        auto round_sol = run_simple_rounding(lp, root_res.x, int_vars);
        result.heuristic_time_ms += h_timer.elapsed_milliseconds();
        if (round_sol.has_value()) {
            scalar_t h_obj = eval_obj(round_sol.value());
            if (update_incumbent(round_sol.value(), h_obj, 1)) {
                result.heuristic_solutions_found++;
            }
        }
    }

    if (config_.heuristic_strategy == HeuristicStrategy::Diving || config_.heuristic_strategy == HeuristicStrategy::All) {
        utils::CPUTimer h_timer;
        auto dive_sol = run_diving_heuristic(lp, root_res.x, int_vars, simplex, root_res.final_basis);
        result.heuristic_time_ms += h_timer.elapsed_milliseconds();
        if (dive_sol.has_value()) {
            scalar_t h_obj = eval_obj(dive_sol.value());
            if (update_incumbent(dive_sol.value(), h_obj, 1)) {
                result.heuristic_solutions_found++;
            }
        }
    }

    // Check if root LP was already integer optimal
    auto [root_frac_var, root_max_frac] = find_fractional(root_res.x);
    if (root_frac_var == -1) {
        result.status = MILPTerminationStatus::Optimal;
        result.best_objective = root_res.objective_value;
        result.best_bound = root_res.objective_value;
        result.mip_gap = 0.0;
        result.x = root_res.x;
        result.nodes_created = 1;
        result.nodes_explored = 1;
        result.time_to_first_incumbent_ms = result.root_relaxation_time_ms;
        result.total_time_ms = timer.elapsed_milliseconds();
        result.nodes_per_second = 1000.0 / std::max(0.001, result.total_time_ms);
        return result;
    }

    // 6. Branch-and-Bound Data Structures
    PseudoCostTable pcost_table(n);
    int node_counter = 0;

    std::priority_queue<BBNode, std::vector<BBNode>, BestBoundComparator> best_bound_queue;
    std::priority_queue<BBNode, std::vector<BBNode>, BestEstimateComparator> best_estimate_queue;
    std::stack<BBNode> dfs_stack;

    auto push_node = [&](const BBNode& node) {
        result.nodes_created++;
        if (config_.node_selection == NodeSelectionStrategy::DepthFirst) {
            dfs_stack.push(node);
        } else if (config_.node_selection == NodeSelectionStrategy::BestEstimate) {
            best_estimate_queue.push(node);
        } else {
            best_bound_queue.push(node);
        }
        int cur_size = (config_.node_selection == NodeSelectionStrategy::DepthFirst) ?
            static_cast<int>(dfs_stack.size()) :
            (config_.node_selection == NodeSelectionStrategy::BestEstimate) ?
            static_cast<int>(best_estimate_queue.size()) :
            static_cast<int>(best_bound_queue.size());
        result.peak_tree_size = std::max(result.peak_tree_size, cur_size);
    };

    auto pop_node = [&]() -> BBNode {
        BBNode n;
        if (config_.node_selection == NodeSelectionStrategy::DepthFirst) {
            n = dfs_stack.top();
            dfs_stack.pop();
        } else if (config_.node_selection == NodeSelectionStrategy::BestEstimate) {
            n = best_estimate_queue.top();
            best_estimate_queue.pop();
        } else {
            n = best_bound_queue.top();
            best_bound_queue.pop();
        }
        return n;
    };

    auto is_queue_empty = [&]() -> bool {
        if (config_.node_selection == NodeSelectionStrategy::DepthFirst) return dfs_stack.empty();
        if (config_.node_selection == NodeSelectionStrategy::BestEstimate) return best_estimate_queue.empty();
        return best_bound_queue.empty();
    };

    auto get_current_best_bound = [&]() -> scalar_t {
        if (is_queue_empty()) return incumbent_obj;
        if (config_.node_selection == NodeSelectionStrategy::BestBound) return best_bound_queue.top().bound;
        return result.best_bound;
    };

    // Create Left & Right children from root
    scalar_t root_val = root_res.x[root_frac_var];

    // Left child: x_j <= floor(val)
    BBNode left_child;
    left_child.id = ++node_counter;
    left_child.depth = 1;
    left_child.bound = root_res.objective_value;
    left_child.estimate = root_res.objective_value + (root_val - std::floor(root_val));
    left_child.col_lower = lp.col_lower;
    left_child.col_upper = lp.col_upper;
    left_child.col_upper[root_frac_var] = std::floor(root_val);
    left_child.basis = root_res.final_basis;
    left_child.branched_var = root_frac_var;
    left_child.is_up_branch = false;
    left_child.branch_frac = root_val - std::floor(root_val);
    push_node(left_child);

    // Right child: x_j >= ceil(val)
    BBNode right_child;
    right_child.id = ++node_counter;
    right_child.depth = 1;
    right_child.bound = root_res.objective_value;
    right_child.estimate = root_res.objective_value + (std::ceil(root_val) - root_val);
    right_child.col_lower = lp.col_lower;
    right_child.col_upper = lp.col_upper;
    right_child.col_lower[root_frac_var] = std::ceil(root_val);
    right_child.basis = root_res.final_basis;
    right_child.branched_var = root_frac_var;
    right_child.is_up_branch = true;
    right_child.branch_frac = std::ceil(root_val) - root_val;
    push_node(right_child);

    result.nodes_explored = 1; // Root counted

    // 7. Tree Exploration Loop
    while (!is_queue_empty()) {
        if (result.nodes_explored >= config_.max_nodes) {
            break;
        }
        if (timer.elapsed_seconds() >= config_.time_limit_sec) {
            break;
        }

        BBNode node = pop_node();

        // Pruning by bound
        if (node.bound >= incumbent_obj - 1e-6) {
            result.nodes_pruned_bound++;
            continue;
        }

        result.nodes_explored++;
        result.best_bound = std::min(incumbent_obj, get_current_best_bound());

        // Update bounds on model
        lp.col_lower = node.col_lower;
        lp.col_upper = node.col_upper;

        // Solve LP relaxation
        utils::CPUTimer node_timer;
        simplex::SimplexResult node_res;
        if (config_.use_warm_start && node.basis.has_value()) {
            node_res = simplex.solve(lp, node.basis);
            result.warm_start_pivots += node_res.iterations;
        } else {
            node_res = simplex.solve(lp, std::nullopt);
            result.cold_start_pivots += node_res.iterations;
        }
        result.lp_relaxation_time_ms += node_timer.elapsed_milliseconds();
        result.total_lp_solves++;
        result.total_pivots += node_res.iterations;

        if (!node_res.is_optimal()) {
            result.nodes_pruned_infeasible++;
            continue;
        }

        // Update pseudocost table with this branch result
        if (node.branched_var >= 0) {
            scalar_t delta = std::max(0.0, node_res.objective_value - node.bound);
            if (node.is_up_branch) {
                pcost_table.update_up(node.branched_var, delta, node.branch_frac);
            } else {
                pcost_table.update_down(node.branched_var, delta, node.branch_frac);
            }
        }

        if (node_res.objective_value >= incumbent_obj - 1e-6) {
            result.nodes_pruned_bound++;
            continue;
        }

        // Check integrality
        auto [frac_var, frac_dist] = find_fractional(node_res.x);
        if (frac_var == -1) {
            // Integer feasible solution found!
            result.nodes_pruned_integral++;
            update_incumbent(node_res.x, node_res.objective_value, result.nodes_explored);

            // Check MIP gap termination
            scalar_t cur_gap = std::abs(incumbent_obj - result.best_bound) / std::max(1.0, std::abs(incumbent_obj));
            if (cur_gap <= config_.mip_gap_tol) {
                break;
            }
            continue;
        }

        // Branching variable selection
        index_t branch_idx = frac_var;
        if (config_.branching_strategy == BranchingStrategy::PseudoCost) {
            index_t best_pc_var = -1;
            scalar_t best_score = -1.0;
            for (index_t j : int_vars) {
                scalar_t val = node_res.x[j];
                scalar_t f_down = val - std::floor(val);
                scalar_t f_up = std::ceil(val) - val;
                if (f_down > config_.integrality_tol && f_up > config_.integrality_tol) {
                    scalar_t sc = pcost_table.score(j, f_down, f_up);
                    if (sc > best_score) {
                        best_score = sc;
                        best_pc_var = j;
                    }
                }
            }
            if (best_pc_var != -1) {
                branch_idx = best_pc_var;
            }
        }

        scalar_t branch_val = node_res.x[branch_idx];

        // Left child: x_j <= floor(val)
        BBNode child_l = node;
        child_l.id = ++node_counter;
        child_l.depth = node.depth + 1;
        child_l.bound = node_res.objective_value;
        child_l.estimate = node_res.objective_value + (branch_val - std::floor(branch_val));
        child_l.col_upper[branch_idx] = std::floor(branch_val);
        child_l.basis = node_res.final_basis;
        child_l.branched_var = branch_idx;
        child_l.is_up_branch = false;
        child_l.branch_frac = branch_val - std::floor(branch_val);
        push_node(child_l);

        // Right child: x_j >= ceil(val)
        BBNode child_r = node;
        child_r.id = ++node_counter;
        child_r.depth = node.depth + 1;
        child_r.bound = node_res.objective_value;
        child_r.estimate = node_res.objective_value + (std::ceil(branch_val) - branch_val);
        child_r.col_lower[branch_idx] = std::ceil(branch_val);
        child_r.basis = node_res.final_basis;
        child_r.branched_var = branch_idx;
        child_r.is_up_branch = true;
        child_r.branch_frac = std::ceil(branch_val) - branch_val;
        push_node(child_r);
    }

    result.active_tree_size = (config_.node_selection == NodeSelectionStrategy::DepthFirst) ?
        static_cast<int>(dfs_stack.size()) :
        (config_.node_selection == NodeSelectionStrategy::BestEstimate) ?
        static_cast<int>(best_estimate_queue.size()) :
        static_cast<int>(best_bound_queue.size());

    result.total_time_ms = timer.elapsed_milliseconds();
    result.nodes_per_second = (result.total_time_ms > 0.0) ?
        (static_cast<double>(result.nodes_explored) / (result.total_time_ms / 1000.0)) : 0.0;

    result.best_objective = incumbent_obj;
    result.x = incumbent_x;

    if (is_queue_empty() && incumbent_obj < model::Infinity / 2.0) {
        result.best_bound = incumbent_obj;
        result.mip_gap = 0.0;
        result.status = MILPTerminationStatus::Optimal;
    } else if (incumbent_obj < model::Infinity / 2.0) {
        result.best_bound = get_current_best_bound();
        result.mip_gap = std::abs(result.best_objective - result.best_bound) /
                         std::max(1.0, std::abs(result.best_objective));
        result.status = (result.mip_gap <= config_.mip_gap_tol) ?
                        MILPTerminationStatus::Optimal :
                        MILPTerminationStatus::Feasible;
    } else {
        result.best_bound = get_current_best_bound();
        result.mip_gap = 1.0;
        result.status = (timer.elapsed_seconds() >= config_.time_limit_sec) ?
                        MILPTerminationStatus::TimeLimit :
                        (result.nodes_explored >= config_.max_nodes) ?
                        MILPTerminationStatus::NodeLimit :
                        MILPTerminationStatus::Infeasible;
    }

    return result;
}

std::pair<MILPResult, MILPResult> BranchAndBoundSolver::solve_warm_vs_cold(const model::LinearProgram& milp_model) {
    MILPConfig warm_cfg = config_;
    warm_cfg.use_warm_start = true;
    BranchAndBoundSolver warm_solver(warm_cfg);
    MILPResult warm_res = warm_solver.solve(milp_model);

    MILPConfig cold_cfg = config_;
    cold_cfg.use_warm_start = false;
    BranchAndBoundSolver cold_solver(cold_cfg);
    MILPResult cold_res = cold_solver.solve(milp_model);

    return {warm_res, cold_res};
}

} // namespace pipepye::milp
