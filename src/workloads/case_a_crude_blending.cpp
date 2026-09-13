#include <pipepye/workloads/case_a_crude_blending.hpp>
#include <pipepye/analysis/problem_analyzer.hpp>
#include <random>
#include <sstream>
#include <iomanip>

namespace pipepye::workloads {

std::pair<model::LinearProgram, WorkloadMetadata> CrudeBlendingGenerator::generate(
    const CrudeBlendingParams& params) {

    model::LinearProgram lp;
    std::ostringstream name_ss;
    name_ss << "BLENDING_" << to_string(params.scale);
    lp.name = name_ss.str();
    lp.is_maximization = false; // PipePye internally stores minimization
    lp.obj_name = "PROFIT";

    int I = params.num_crudes;
    int J = params.num_products;
    int K = params.num_qualities;

    std::mt19937_64 rng(params.seed);
    std::uniform_real_distribution<scalar_t> dist_crude_cost(45.0, 85.0);
    std::uniform_real_distribution<scalar_t> dist_crude_avail(5000.0, 25000.0);
    std::uniform_real_distribution<scalar_t> dist_prod_price(90.0, 135.0);
    std::uniform_real_distribution<scalar_t> dist_prod_demand(3000.0, 15000.0);

    // Crude properties
    std::vector<scalar_t> crude_cost(I);
    std::vector<scalar_t> crude_avail(I);
    for (int i = 0; i < I; ++i) {
        crude_cost[i] = std::round(dist_crude_cost(rng) * 10.0) / 10.0;
        crude_avail[i] = std::round(dist_crude_avail(rng));
    }

    // Product properties
    std::vector<scalar_t> prod_price(J);
    std::vector<scalar_t> prod_demand_min(J);
    std::vector<scalar_t> prod_demand_max(J);
    for (int j = 0; j < J; ++j) {
        prod_price[j] = std::round(dist_prod_price(rng) * 10.0) / 10.0;
        scalar_t base_demand = std::round(dist_prod_demand(rng));
        prod_demand_min[j] = 0.0; // Demand minimum is 0, maximum is market capacity
        prod_demand_max[j] = base_demand;
    }

    // Quality specifications
    // q_crude[i][k], q_min[j][k], q_max[j][k]
    std::vector<std::vector<scalar_t>> q_crude(I, std::vector<scalar_t>(K));
    std::vector<std::vector<scalar_t>> q_min(J, std::vector<scalar_t>(K));
    std::vector<std::vector<scalar_t>> q_max(J, std::vector<scalar_t>(K));

    for (int k = 0; k < K; ++k) {
        std::uniform_real_distribution<scalar_t> dist_q(70.0 + k * 10.0, 95.0 + k * 10.0);
        for (int i = 0; i < I; ++i) {
            q_crude[i][k] = std::round(dist_q(rng) * 10.0) / 10.0;
        }
        // Ensure crude 0 is low and crude I-1 is high to span specifications
        q_crude[0][k] = 65.0 + k * 10.0;
        q_crude[I - 1][k] = 100.0 + k * 10.0;

        for (int j = 0; j < J; ++j) {
            scalar_t mid_q = 80.0 + k * 10.0 + j * 2.0;
            q_min[j][k] = mid_q - 8.0;
            q_max[j][k] = mid_q + 8.0;
        }
    }

    // Variable indexing:
    // x_{i,j} for crude i in product j: index = i * J + j (total I * J)
    // y_j for total product j: index = I * J + j (total J)
    index_t num_cols = I * J + J;
    lp.col_names.resize(num_cols);
    lp.c.assign(num_cols, 0.0);
    lp.col_lower.assign(num_cols, 0.0);
    lp.col_upper.assign(num_cols, model::Infinity);
    lp.var_types.assign(num_cols, model::VariableType::Continuous);

    for (int i = 0; i < I; ++i) {
        for (int j = 0; j < J; ++j) {
            index_t col_idx = i * J + j;
            std::ostringstream ss;
            ss << "X_C" << i << "_P" << j;
            lp.col_names[col_idx] = ss.str();
            lp.col_name_to_idx[ss.str()] = col_idx;
            // Cost of crude: min (cost * x - price * y)
            lp.c[col_idx] = crude_cost[i];
        }
    }

    for (int j = 0; j < J; ++j) {
        index_t col_idx = I * J + j;
        std::ostringstream ss;
        ss << "Y_P" << j;
        lp.col_names[col_idx] = ss.str();
        lp.col_name_to_idx[ss.str()] = col_idx;
        // Revenue from product (minimization -> negative cost)
        lp.c[col_idx] = -prod_price[j];
        lp.col_lower[col_idx] = prod_demand_min[j];
        lp.col_upper[col_idx] = prod_demand_max[j];
    }

    // Constraints:
    // 1. Component Balance: sum_i x_{i,j} - y_j = 0  (J rows)
    // 2. Crude Availability: sum_j x_{i,j} <= crude_avail_i (I rows)
    // 3. Quality Min: sum_i q_{i,k} x_{i,j} - q_min_{j,k} y_j >= 0 (J * K rows)
    // 4. Quality Max: sum_i q_{i,k} x_{i,j} - q_max_{j,k} y_j <= 0 (J * K rows)
    index_t num_rows = J + I + 2 * J * K;
    lp.row_names.resize(num_rows);
    lp.row_lower.assign(num_rows, 0.0);
    lp.row_upper.assign(num_rows, 0.0);
    lp.row_senses.resize(num_rows);

    std::vector<pipepye::sparse::TripletF64> triplets;
    index_t row_idx = 0;

    // 1. Balance
    for (int j = 0; j < J; ++j) {
        std::string rname = "BAL_P" + std::to_string(j);
        lp.row_names[row_idx] = rname;
        lp.row_name_to_idx[rname] = row_idx;
        lp.row_lower[row_idx] = 0.0;
        lp.row_upper[row_idx] = 0.0;
        lp.row_senses[row_idx] = model::RowSense::Equality;

        for (int i = 0; i < I; ++i) {
            triplets.push_back({row_idx, i * J + j, 1.0});
        }
        triplets.push_back({row_idx, I * J + j, -1.0});
        row_idx++;
    }

    // 2. Availability
    for (int i = 0; i < I; ++i) {
        std::string rname = "AVAIL_C" + std::to_string(i);
        lp.row_names[row_idx] = rname;
        lp.row_name_to_idx[rname] = row_idx;
        lp.row_lower[row_idx] = -model::Infinity;
        lp.row_upper[row_idx] = crude_avail[i];
        lp.row_senses[row_idx] = model::RowSense::LessEqual;

        for (int j = 0; j < J; ++j) {
            triplets.push_back({row_idx, i * J + j, 1.0});
        }
        row_idx++;
    }

    // 3. Quality Min
    for (int j = 0; j < J; ++j) {
        for (int k = 0; k < K; ++k) {
            std::string rname = "QMIN_P" + std::to_string(j) + "_Q" + std::to_string(k);
            lp.row_names[row_idx] = rname;
            lp.row_name_to_idx[rname] = row_idx;
            lp.row_lower[row_idx] = 0.0;
            lp.row_upper[row_idx] = model::Infinity;
            lp.row_senses[row_idx] = model::RowSense::GreaterEqual;

            for (int i = 0; i < I; ++i) {
                triplets.push_back({row_idx, i * J + j, q_crude[i][k]});
            }
            triplets.push_back({row_idx, I * J + j, -q_min[j][k]});
            row_idx++;
        }
    }

    // 4. Quality Max
    for (int j = 0; j < J; ++j) {
        for (int k = 0; k < K; ++k) {
            std::string rname = "QMAX_P" + std::to_string(j) + "_Q" + std::to_string(k);
            lp.row_names[row_idx] = rname;
            lp.row_name_to_idx[rname] = row_idx;
            lp.row_lower[row_idx] = -model::Infinity;
            lp.row_upper[row_idx] = 0.0;
            lp.row_senses[row_idx] = model::RowSense::LessEqual;

            for (int i = 0; i < I; ++i) {
                triplets.push_back({row_idx, i * J + j, q_crude[i][k]});
            }
            triplets.push_back({row_idx, I * J + j, -q_max[j][k]});
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

    // Analyze problem structure
    analysis::ProblemStats stats = analysis::ProblemAnalyzer::analyze(lp);

    WorkloadMetadata meta;
    meta.problem_name = "Crude_Blending";
    meta.instance_id = lp.name;
    meta.random_seed = params.seed;
    meta.scale = params.scale;
    meta.num_continuous_vars = num_cols;
    meta.num_integer_vars = 0;
    meta.num_binary_vars = 0;
    meta.min_rhs = 0.0;
    meta.max_rhs = 25000.0;
    meta.min_bound = 0.0;
    meta.max_bound = 30000.0;
    meta.expected_solver = "DualSimplex";
    meta.expected_backend = "CPU";
    meta.prediction_rationale =
        "Dense quality rows and compact dimensions favor sparse Dual Simplex with fast basis factorization over iterative first-order GPU solvers.";
    meta.populate_from_stats(stats);

    return {std::move(lp), std::move(meta)};
}

} // namespace pipepye::workloads
