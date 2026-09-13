#include <pipepye/workloads/case_c_refinery_scheduling.hpp>
#include <pipepye/analysis/problem_analyzer.hpp>
#include <random>
#include <sstream>
#include <iomanip>

namespace pipepye::workloads {

std::pair<model::LinearProgram, WorkloadMetadata> RefinerySchedulingGenerator::generate(
    const RefinerySchedulingParams& params) {

    model::LinearProgram lp;
    std::ostringstream name_ss;
    name_ss << "REFINERY_SCHED_" << to_string(params.scale);
    lp.name = name_ss.str();
    lp.is_maximization = false;
    lp.obj_name = "NET_COST";

    int U = params.num_units;
    int M = params.num_modes;
    int T = params.num_periods;

    std::mt19937_64 rng(params.seed);
    std::uniform_real_distribution<scalar_t> dist_oper_cost(15.0, 40.0);
    std::uniform_real_distribution<scalar_t> dist_fixed_cost(100.0, 300.0);
    std::uniform_real_distribution<scalar_t> dist_max_cap(150.0, 350.0);
    std::uniform_real_distribution<scalar_t> dist_demand(80.0, 200.0);

    // Costs and capacities
    std::vector<std::vector<scalar_t>> oper_cost(U, std::vector<scalar_t>(M));
    std::vector<std::vector<scalar_t>> fixed_cost(U, std::vector<scalar_t>(M));
    std::vector<std::vector<scalar_t>> max_cap(U, std::vector<scalar_t>(M));
    std::vector<std::vector<scalar_t>> min_cap(U, std::vector<scalar_t>(M));

    for (int u = 0; u < U; ++u) {
        for (int m = 0; m < M; ++m) {
            oper_cost[u][m] = std::round(dist_oper_cost(rng) * 10.0) / 10.0;
            fixed_cost[u][m] = std::round(dist_fixed_cost(rng));
            scalar_t cap = std::round(dist_max_cap(rng));
            max_cap[u][m] = cap;
            min_cap[u][m] = cap * 0.3;
        }
    }

    std::vector<scalar_t> final_demand(T);
    for (int t = 0; t < T; ++t) {
        final_demand[t] = std::round(dist_demand(rng));
    }

    // Variables per period t:
    // 1. Binary z_{u,m,t} (U * M * T)
    // 2. Continuous x_{u,m,t} (U * M * T)
    // 3. Continuous s_{u,t} buffer storage after unit u (U * T)
    int vars_per_t = 2 * U * M + U;
    index_t num_cols = T * vars_per_t;

    lp.col_names.resize(num_cols);
    lp.c.assign(num_cols, 0.0);
    lp.col_lower.assign(num_cols, 0.0);
    lp.col_upper.assign(num_cols, model::Infinity);
    lp.var_types.assign(num_cols, model::VariableType::Continuous);

    auto get_z_idx = [&](int t, int u, int m) -> index_t {
        return t * vars_per_t + (u * M + m);
    };
    auto get_x_idx = [&](int t, int u, int m) -> index_t {
        return t * vars_per_t + U * M + (u * M + m);
    };
    auto get_s_idx = [&](int t, int u) -> index_t {
        return t * vars_per_t + 2 * U * M + u;
    };

    index_t binary_count = 0;
    for (int t = 0; t < T; ++t) {
        for (int u = 0; u < U; ++u) {
            for (int m = 0; m < M; ++m) {
                // Binary z
                index_t z_col = get_z_idx(t, u, m);
                std::string z_name = "Z_U" + std::to_string(u) + "_M" + std::to_string(m) + "_T" + std::to_string(t);
                lp.col_names[z_col] = z_name;
                lp.col_name_to_idx[z_name] = z_col;
                lp.var_types[z_col] = model::VariableType::Binary;
                lp.col_lower[z_col] = 0.0;
                lp.col_upper[z_col] = 1.0;
                lp.c[z_col] = fixed_cost[u][m];
                binary_count++;

                // Continuous x
                index_t x_col = get_x_idx(t, u, m);
                std::string x_name = "X_U" + std::to_string(u) + "_M" + std::to_string(m) + "_T" + std::to_string(t);
                lp.col_names[x_col] = x_name;
                lp.col_name_to_idx[x_name] = x_col;
                lp.var_types[x_col] = model::VariableType::Continuous;
                lp.col_lower[x_col] = 0.0;
                lp.col_upper[x_col] = max_cap[u][m];
                // Final unit sells product for revenue
                if (u == U - 1) {
                    lp.c[x_col] = oper_cost[u][m] - 95.0; // Negative cost = revenue
                } else {
                    lp.c[x_col] = oper_cost[u][m];
                }
            }

            // Continuous buffer inventory s
            index_t s_col = get_s_idx(t, u);
            std::string s_name = "S_U" + std::to_string(u) + "_T" + std::to_string(t);
            lp.col_names[s_col] = s_name;
            lp.col_name_to_idx[s_name] = s_col;
            lp.var_types[s_col] = model::VariableType::Continuous;
            lp.col_lower[s_col] = 0.0;
            lp.col_upper[s_col] = 500.0; // max buffer capacity
            lp.c[s_col] = 2.0;           // holding cost
        }
    }

    // Constraints:
    // 1. Single mode selection: sum_m z_{u,m,t} <= 1 (U * T rows)
    // 2. Max capacity: x_{u,m,t} - max_cap * z_{u,m,t} <= 0 (U * M * T rows)
    // 3. Min throughput when active: x_{u,m,t} - min_cap * z_{u,m,t} >= 0 (U * M * T rows)
    // 4. Inter-unit mass balance:
    //    s_{u,t} - s_{u,t-1} - 0.9 * sum_m x_{u,m,t} + sum_m x_{u+1,m,t} = 0 (for u < U-1) ((U - 1) * T rows)
    // 5. Final demand: sum_m x_{U-1,m,t} >= final_demand[t] (T rows)
    index_t rows_per_t = U + 2 * U * M + (U - 1) + 1;
    index_t num_rows = T * rows_per_t;
    lp.row_names.resize(num_rows);
    lp.row_lower.assign(num_rows, 0.0);
    lp.row_upper.assign(num_rows, 0.0);
    lp.row_senses.resize(num_rows);

    std::vector<pipepye::sparse::TripletF64> triplets;
    index_t row_idx = 0;

    for (int t = 0; t < T; ++t) {
        // 1. Single mode
        for (int u = 0; u < U; ++u) {
            std::string rname = "MODE_U" + std::to_string(u) + "_T" + std::to_string(t);
            lp.row_names[row_idx] = rname;
            lp.row_name_to_idx[rname] = row_idx;
            lp.row_lower[row_idx] = -model::Infinity;
            lp.row_upper[row_idx] = 1.0;
            lp.row_senses[row_idx] = model::RowSense::LessEqual;

            for (int m = 0; m < M; ++m) {
                triplets.push_back({row_idx, get_z_idx(t, u, m), 1.0});
            }
            row_idx++;
        }

        // 2. Max capacity & 3. Min throughput
        for (int u = 0; u < U; ++u) {
            for (int m = 0; m < M; ++m) {
                // Max: x - max_cap * z <= 0
                std::string rname_max = "CAPMAX_U" + std::to_string(u) + "_M" + std::to_string(m) + "_T" + std::to_string(t);
                lp.row_names[row_idx] = rname_max;
                lp.row_name_to_idx[rname_max] = row_idx;
                lp.row_lower[row_idx] = -model::Infinity;
                lp.row_upper[row_idx] = 0.0;
                lp.row_senses[row_idx] = model::RowSense::LessEqual;

                triplets.push_back({row_idx, get_x_idx(t, u, m), 1.0});
                triplets.push_back({row_idx, get_z_idx(t, u, m), -max_cap[u][m]});
                row_idx++;

                // Min: x - min_cap * z >= 0
                std::string rname_min = "CAPMIN_U" + std::to_string(u) + "_M" + std::to_string(m) + "_T" + std::to_string(t);
                lp.row_names[row_idx] = rname_min;
                lp.row_name_to_idx[rname_min] = row_idx;
                lp.row_lower[row_idx] = 0.0;
                lp.row_upper[row_idx] = model::Infinity;
                lp.row_senses[row_idx] = model::RowSense::GreaterEqual;

                triplets.push_back({row_idx, get_x_idx(t, u, m), 1.0});
                triplets.push_back({row_idx, get_z_idx(t, u, m), -min_cap[u][m]});
                row_idx++;
            }
        }

        // 4. Inter-unit buffer mass balance
        for (int u = 0; u < U - 1; ++u) {
            std::string rname = "MBAL_U" + std::to_string(u) + "_T" + std::to_string(t);
            lp.row_names[row_idx] = rname;
            lp.row_name_to_idx[rname] = row_idx;
            lp.row_lower[row_idx] = 0.0;
            lp.row_upper[row_idx] = 0.0;
            lp.row_senses[row_idx] = model::RowSense::Equality;

            // + s_{u,t}
            triplets.push_back({row_idx, get_s_idx(t, u), 1.0});
            // - s_{u,t-1}
            if (t > 0) {
                triplets.push_back({row_idx, get_s_idx(t - 1, u), -1.0});
            }
            // - 0.9 * sum_m x_{u,m,t}
            for (int m = 0; m < M; ++m) {
                triplets.push_back({row_idx, get_x_idx(t, u, m), -0.9});
            }
            // + sum_m x_{u+1,m,t}
            for (int m = 0; m < M; ++m) {
                triplets.push_back({row_idx, get_x_idx(t, u + 1, m), 1.0});
            }
            row_idx++;
        }

        // 5. Final demand
        {
            std::string rname = "DEMAND_T" + std::to_string(t);
            lp.row_names[row_idx] = rname;
            lp.row_name_to_idx[rname] = row_idx;
            lp.row_lower[row_idx] = final_demand[t];
            lp.row_upper[row_idx] = model::Infinity;
            lp.row_senses[row_idx] = model::RowSense::GreaterEqual;

            for (int m = 0; m < M; ++m) {
                triplets.push_back({row_idx, get_x_idx(t, U - 1, m), 1.0});
            }
            row_idx++;
        }
    }

    // Assemble matrices
    lp.A_coo = pipepye::sparse::COOMatrix(num_rows, num_cols, std::move(triplets));
    lp.A_coo.sort(pipepye::sparse::StorageOrder::ColMajor);
    lp.A_coo.sum_duplicates();

    auto csc = lp.A_coo.to_csc();
    auto csr = lp.A_coo.to_csr();

    lp.csc_col_ptr = csc.col_ptr_vector();
    lp.csc_row_ind = csc.row_ind_vector();
    lp.csc_values = csc.values_vector();

    lp.csr_row_ptr = csr.row_ptr_vector();
    lp.csr_col_ind = csr.col_ind_vector();
    lp.csr_values = csr.values_vector();

    analysis::ProblemStats stats = analysis::ProblemAnalyzer::analyze(lp);

    WorkloadMetadata meta;
    meta.problem_name = "Refinery_Scheduling";
    meta.instance_id = lp.name;
    meta.random_seed = params.seed;
    meta.scale = params.scale;
    meta.num_continuous_vars = num_cols - binary_count;
    meta.num_integer_vars = 0;
    meta.num_binary_vars = binary_count;
    meta.min_rhs = 0.0;
    meta.max_rhs = 500.0;
    meta.min_bound = 0.0;
    meta.max_bound = 500.0;
    meta.expected_solver = "BranchAndBound";
    meta.expected_backend = "CPU";
    meta.prediction_rationale =
        "Discrete unit commitment and operating modes impose combinatorial search; warm-started Dual Simplex provides crucial pivot efficiency during branch-and-bound exploration.";
    meta.populate_from_stats(stats);

    return {std::move(lp), std::move(meta)};
}

} // namespace pipepye::workloads
