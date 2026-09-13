#include <pipepye/milp/branch_and_bound.hpp>
#include <pipepye/utils/timer.hpp>
#include <queue>
#include <cmath>
#include <algorithm>

namespace pipepye::milp {

BranchAndBoundSolver::BranchAndBoundSolver(MILPConfig config)
    : config_(std::move(config)) {}

namespace {

struct BBNode {
    int id{0};
    int depth{0};
    scalar_t bound{0.0};
    std::vector<scalar_t> col_lower;
    std::vector<scalar_t> col_upper;
    std::optional<simplex::Basis> basis;

    bool operator>(const BBNode& other) const noexcept {
        return bound > other.bound;
    }
};

inline MILPTerminationStatus map_simplex_status(solver::TerminationStatus s) {
    switch (s) {
        case solver::TerminationStatus::OPTIMAL: return MILPTerminationStatus::Optimal;
        case solver::TerminationStatus::PRIMAL_INFEASIBLE: return MILPTerminationStatus::Infeasible;
        case solver::TerminationStatus::TIME_LIMIT: return MILPTerminationStatus::TimeLimit;
        case solver::TerminationStatus::MAX_ITERATIONS: return MILPTerminationStatus::NodeLimit;
        default: return MILPTerminationStatus::NumericalFailure;
    }
}

} // namespace

MILPResult BranchAndBoundSolver::solve(const model::LinearProgram& milp_model) {
    MILPResult result;
    utils::CPUTimer timer;
    timer.start();

    index_t n = milp_model.num_cols();

    // 1. Identify integer / binary variables
    std::vector<index_t> int_vars;
    for (index_t j = 0; j < n; ++j) {
        if (j < static_cast<index_t>(milp_model.var_types.size())) {
            auto vt = milp_model.var_types[j];
            if (vt == model::VariableType::Binary || vt == model::VariableType::Integer) {
                int_vars.push_back(j);
            }
        }
    }

    simplex::DualSimplexSolver simplex(config_.simplex_config);

    // If pure continuous LP, solve directly
    if (int_vars.empty()) {
        auto lp_res = simplex.solve(milp_model, std::nullopt);
        result.status = map_simplex_status(lp_res.status);
        result.best_objective = lp_res.objective_value;
        result.best_bound = lp_res.objective_value;
        result.mip_gap = 0.0;
        result.x = lp_res.x;
        result.nodes_explored = 1;
        result.total_lp_solves = 1;
        result.total_pivots = lp_res.iterations;
        result.root_objective = lp_res.objective_value;
        result.root_relaxation_time_ms = lp_res.timing.total_time_ms;
        result.time_to_first_incumbent_ms = lp_res.timing.total_time_ms;
        result.total_time_ms = timer.elapsed_milliseconds();
        return result;
    }

    // 2. Solve Root LP Relaxation
    utils::CPUTimer root_timer;
    root_timer.start();
    model::LinearProgram lp = milp_model;
    auto root_res = simplex.solve(lp, std::nullopt);
    result.root_relaxation_time_ms = root_timer.elapsed_milliseconds();
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

    // Check if root relaxation is already integral
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

    auto [root_frac_var, root_max_frac] = find_fractional(root_res.x);
    if (root_frac_var == -1) {
        // Root is optimal integer solution
        result.status = MILPTerminationStatus::Optimal;
        result.best_objective = root_res.objective_value;
        result.best_bound = root_res.objective_value;
        result.mip_gap = 0.0;
        result.x = root_res.x;
        result.nodes_explored = 1;
        result.time_to_first_incumbent_ms = result.root_relaxation_time_ms;
        result.total_time_ms = timer.elapsed_milliseconds();
        return result;
    }

    // 3. Initialize Branch-and-Bound Priority Queue (Best-Bound Search)
    std::priority_queue<BBNode, std::vector<BBNode>, std::greater<BBNode>> queue;
    int node_counter = 0;

    // Create Left & Right children from root
    scalar_t root_val = root_res.x[root_frac_var];

    // Left child: x[j] <= floor(val)
    BBNode left_child;
    left_child.id = ++node_counter;
    left_child.depth = 1;
    left_child.bound = root_res.objective_value;
    left_child.col_lower = lp.col_lower;
    left_child.col_upper = lp.col_upper;
    left_child.col_upper[root_frac_var] = std::floor(root_val);
    left_child.basis = root_res.final_basis;
    queue.push(left_child);

    // Right child: x[j] >= ceil(val)
    BBNode right_child;
    right_child.id = ++node_counter;
    right_child.depth = 1;
    right_child.bound = root_res.objective_value;
    right_child.col_lower = lp.col_lower;
    right_child.col_upper = lp.col_upper;
    right_child.col_lower[root_frac_var] = std::ceil(root_val);
    right_child.basis = root_res.final_basis;
    queue.push(right_child);

    scalar_t incumbent_obj = model::Infinity;
    std::vector<scalar_t> incumbent_x;

    result.nodes_explored = 1; // Root counted

    // 4. Branch-and-Bound Tree Exploration Loop
    while (!queue.empty()) {
        if (result.nodes_explored >= config_.max_nodes) {
            break;
        }
        if (timer.elapsed_seconds() >= config_.time_limit_sec) {
            break;
        }

        BBNode node = queue.top();
        queue.pop();

        // Pruning by bound
        if (node.bound >= incumbent_obj - 1e-6) {
            continue;
        }

        result.nodes_explored++;

        // Update bounds on model
        lp.col_lower = node.col_lower;
        lp.col_upper = node.col_upper;

        // Solve LP relaxation with or without warm start
        simplex::SimplexResult node_res;
        if (config_.use_warm_start && node.basis.has_value()) {
            node_res = simplex.solve(lp, node.basis);
            result.warm_start_pivots += node_res.iterations;
        } else {
            node_res = simplex.solve(lp, std::nullopt);
            result.cold_start_pivots += node_res.iterations;
        }

        result.total_lp_solves++;
        result.total_pivots += node_res.iterations;

        if (!node_res.is_optimal()) {
            // Infeasible node -> prune
            continue;
        }

        if (node_res.objective_value >= incumbent_obj - 1e-6) {
            // Dominated node -> prune
            continue;
        }

        // Check integrality
        auto [frac_var, frac_dist] = find_fractional(node_res.x);
        if (frac_var == -1) {
            // Integer feasible solution found!
            if (node_res.objective_value < incumbent_obj) {
                incumbent_obj = node_res.objective_value;
                incumbent_x = node_res.x;
                if (result.time_to_first_incumbent_ms == 0.0) {
                    result.time_to_first_incumbent_ms = timer.elapsed_milliseconds();
                }

                // Check MIP gap termination
                if (!queue.empty()) {
                    scalar_t current_best_bound = queue.top().bound;
                    scalar_t gap = std::abs(incumbent_obj - current_best_bound) / std::max(1.0, std::abs(incumbent_obj));
                    if (gap <= config_.mip_gap_tol) {
                        break;
                    }
                }
            }
            continue; // Integer solution reached -> prune
        }

        // Branch on frac_var
        scalar_t val = node_res.x[frac_var];

        // Left child: x[j] <= floor(val)
        BBNode child_l = node;
        child_l.id = ++node_counter;
        child_l.depth = node.depth + 1;
        child_l.bound = node_res.objective_value;
        child_l.col_upper[frac_var] = std::floor(val);
        child_l.basis = node_res.final_basis;
        queue.push(child_l);

        // Right child: x[j] >= ceil(val)
        BBNode child_r = node;
        child_r.id = ++node_counter;
        child_r.depth = node.depth + 1;
        child_r.bound = node_res.objective_value;
        child_r.col_lower[frac_var] = std::ceil(val);
        child_r.basis = node_res.final_basis;
        queue.push(child_r);
    }

    result.total_time_ms = timer.elapsed_milliseconds();
    result.best_objective = incumbent_obj;
    result.x = incumbent_x;

    if (queue.empty() && incumbent_obj < model::Infinity / 2.0) {
        result.best_bound = incumbent_obj;
        result.mip_gap = 0.0;
        result.status = MILPTerminationStatus::Optimal;
    } else if (incumbent_obj < model::Infinity / 2.0) {
        result.best_bound = queue.empty() ? incumbent_obj : queue.top().bound;
        result.mip_gap = std::abs(result.best_objective - result.best_bound) /
                         std::max(1.0, std::abs(result.best_objective));
        result.status = (result.mip_gap <= config_.mip_gap_tol) ?
                        MILPTerminationStatus::Optimal :
                        MILPTerminationStatus::Feasible;
    } else {
        result.best_bound = queue.empty() ? model::Infinity : queue.top().bound;
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
