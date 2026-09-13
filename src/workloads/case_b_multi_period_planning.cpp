#include <pipepye/workloads/case_b_multi_period_planning.hpp>
#include <pipepye/analysis/problem_analyzer.hpp>
#include <random>
#include <sstream>
#include <iomanip>

namespace pipepye::workloads {

std::pair<model::LinearProgram, WorkloadMetadata> MultiPeriodPlanningGenerator::generate(
    const MultiPeriodPlanningParams& params) {

    model::LinearProgram lp;
    std::ostringstream name_ss;
    name_ss << "PLANNING_T" << params.num_periods << "_" << to_string(params.scale);
    lp.name = name_ss.str();
    lp.is_maximization = false;
    lp.obj_name = "COST";

    int T = params.num_periods;
    int P = params.num_products;
    int M = params.num_machines;

    std::mt19937_64 rng(params.seed);
    std::uniform_real_distribution<scalar_t> dist_prod_cost(10.0, 50.0);
    std::uniform_real_distribution<scalar_t> dist_hold_cost(1.0, 5.0);
    std::uniform_real_distribution<scalar_t> dist_demand(50.0, 200.0);
    std::uniform_real_distribution<scalar_t> dist_machine_time(0.5, 3.0);
    std::uniform_real_distribution<scalar_t> dist_capacity(200.0, 500.0);

    // Costs
    std::vector<std::vector<scalar_t>> prod_cost(T, std::vector<scalar_t>(P));
    std::vector<std::vector<scalar_t>> hold_cost(T, std::vector<scalar_t>(P));
    std::vector<std::vector<scalar_t>> demand(T, std::vector<scalar_t>(P));
    for (int t = 0; t < T; ++t) {
        for (int p = 0; p < P; ++p) {
            prod_cost[t][p] = std::round(dist_prod_cost(rng) * 10.0) / 10.0;
            hold_cost[t][p] = std::round(dist_hold_cost(rng) * 10.0) / 10.0;
            demand[t][p] = std::round(dist_demand(rng));
        }
    }

    // Machine resource consumption matrix A_{m, p}
    std::vector<std::vector<scalar_t>> mach_usage(M, std::vector<scalar_t>(P));
    for (int m = 0; m < M; ++m) {
        for (int p = 0; p < P; ++p) {
            mach_usage[m][p] = std::round(dist_machine_time(rng) * 10.0) / 10.0;
        }
    }

    // Machine capacity per period (ample headroom to allow production and inventory accumulation)
    std::vector<std::vector<scalar_t>> mach_cap(T, std::vector<scalar_t>(M));
    for (int t = 0; t < T; ++t) {
        for (int m = 0; m < M; ++m) {
            mach_cap[t][m] = std::round(dist_capacity(rng) * P * 3.0);
        }
    }

    // Warehouse storage limit
    scalar_t max_storage = P * 500.0;

    // Variables:
    // Period t:
    //   x_{p,t} for p in 0..P-1 (index: t * (2*P) + p)
    //   s_{p,t} for p in 0..P-1 (index: t * (2*P) + P + p)
    index_t num_cols = T * (2 * P);
    lp.col_names.resize(num_cols);
    lp.c.assign(num_cols, 0.0);
    lp.col_lower.assign(num_cols, 0.0);
    lp.col_upper.assign(num_cols, model::Infinity);
    lp.var_types.assign(num_cols, model::VariableType::Continuous);

    auto get_x_idx = [&](int t, int p) -> index_t { return t * (2 * P) + p; };
    auto get_s_idx = [&](int t, int p) -> index_t { return t * (2 * P) + P + p; };

    for (int t = 0; t < T; ++t) {
        for (int p = 0; p < P; ++p) {
            index_t x_col = get_x_idx(t, p);
            std::string x_name = "X_P" + std::to_string(p) + "_T" + std::to_string(t);
            lp.col_names[x_col] = x_name;
            lp.col_name_to_idx[x_name] = x_col;
            lp.c[x_col] = prod_cost[t][p];

            index_t s_col = get_s_idx(t, p);
            std::string s_name = "S_P" + std::to_string(p) + "_T" + std::to_string(t);
            lp.col_names[s_col] = s_name;
            lp.col_name_to_idx[s_name] = s_col;
            lp.c[s_col] = hold_cost[t][p];
        }
    }

    // Constraints:
    // Per period t:
    // 1. Inventory balance: x_{p,t} + s_{p,t-1} - s_{p,t} = demand_{p,t} (P rows)
    // 2. Machine capacity: sum_p mach_usage_{m,p} * x_{p,t} <= mach_cap_{t,m} (M rows)
    // 3. Warehouse storage: sum_p s_{p,t} <= max_storage (1 row)
    index_t rows_per_t = P + M + 1;
    index_t num_rows = T * rows_per_t;
    lp.row_names.resize(num_rows);
    lp.row_lower.assign(num_rows, 0.0);
    lp.row_upper.assign(num_rows, 0.0);
    lp.row_senses.resize(num_rows);

    std::vector<pipepye::sparse::TripletF64> triplets;
    index_t row_idx = 0;

    for (int t = 0; t < T; ++t) {
        // 1. Inventory balance
        for (int p = 0; p < P; ++p) {
            std::string rname = "BAL_P" + std::to_string(p) + "_T" + std::to_string(t);
            lp.row_names[row_idx] = rname;
            lp.row_name_to_idx[rname] = row_idx;
            lp.row_lower[row_idx] = demand[t][p];
            lp.row_upper[row_idx] = demand[t][p];
            lp.row_senses[row_idx] = model::RowSense::Equality;

            // + x_{p,t}
            triplets.push_back({row_idx, get_x_idx(t, p), 1.0});
            // + s_{p,t-1} if t > 0
            if (t > 0) {
                triplets.push_back({row_idx, get_s_idx(t - 1, p), 1.0});
            }
            // - s_{p,t}
            triplets.push_back({row_idx, get_s_idx(t, p), -1.0});

            row_idx++;
        }

        // 2. Machine capacity
        for (int m = 0; m < M; ++m) {
            std::string rname = "CAP_M" + std::to_string(m) + "_T" + std::to_string(t);
            lp.row_names[row_idx] = rname;
            lp.row_name_to_idx[rname] = row_idx;
            lp.row_lower[row_idx] = -model::Infinity;
            lp.row_upper[row_idx] = mach_cap[t][m];
            lp.row_senses[row_idx] = model::RowSense::LessEqual;

            for (int p = 0; p < P; ++p) {
                triplets.push_back({row_idx, get_x_idx(t, p), mach_usage[m][p]});
            }
            row_idx++;
        }

        // 3. Storage capacity
        {
            std::string rname = "STORAGE_T" + std::to_string(t);
            lp.row_names[row_idx] = rname;
            lp.row_name_to_idx[rname] = row_idx;
            lp.row_lower[row_idx] = -model::Infinity;
            lp.row_upper[row_idx] = max_storage;
            lp.row_senses[row_idx] = model::RowSense::LessEqual;

            for (int p = 0; p < P; ++p) {
                triplets.push_back({row_idx, get_s_idx(t, p), 1.0});
            }
            row_idx++;
        }
    }

    // Assemble sparse matrices
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
    meta.problem_name = "Multi_Period_Planning";
    meta.instance_id = lp.name;
    meta.random_seed = params.seed;
    meta.scale = params.scale;
    meta.num_continuous_vars = num_cols;
    meta.num_integer_vars = 0;
    meta.num_binary_vars = 0;
    meta.min_rhs = 0.0;
    meta.max_rhs = max_storage;
    meta.min_bound = 0.0;
    meta.max_bound = model::Infinity;

    if (params.scale == InstanceScale::Large || params.scale == InstanceScale::Stress || params.num_periods >= 50) {
        meta.expected_solver = "PDHG";
        meta.expected_backend = "GPU";
        meta.prediction_rationale =
            "Large multi-period block-angular staircase structure with high NNZ maps exceptionally well to GPU memory bandwidth and data-parallel SpMV.";
    } else {
        meta.expected_solver = "DualSimplex";
        meta.expected_backend = "CPU";
        meta.prediction_rationale =
            "Small to moderate period models stay within CPU cache boundaries where low-overhead Simplex pivots outperform iterative first-order GPU solvers.";
    }
    meta.populate_from_stats(stats);

    return {std::move(lp), std::move(meta)};
}

} // namespace pipepye::workloads
