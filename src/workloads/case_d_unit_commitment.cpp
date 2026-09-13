#include <pipepye/workloads/case_d_unit_commitment.hpp>
#include <pipepye/analysis/problem_analyzer.hpp>
#include <random>
#include <sstream>
#include <iomanip>

namespace pipepye::workloads {

std::pair<model::LinearProgram, WorkloadMetadata> UnitCommitmentGenerator::generate(
    const UnitCommitmentParams& params) {

    model::LinearProgram lp;
    std::ostringstream name_ss;
    name_ss << "UNIT_COMMIT_" << to_string(params.scale);
    lp.name = name_ss.str();
    lp.is_maximization = false;
    lp.obj_name = "TOTAL_COST";

    int G = params.num_generators;
    int T = params.num_periods;

    std::mt19937_64 rng(params.seed);
    std::uniform_real_distribution<scalar_t> dist_pmin(20.0, 80.0);
    std::uniform_real_distribution<scalar_t> dist_pmax(120.0, 300.0);
    std::uniform_real_distribution<scalar_t> dist_lin_cost(20.0, 60.0);
    std::uniform_real_distribution<scalar_t> dist_nl_cost(100.0, 400.0);
    std::uniform_real_distribution<scalar_t> dist_ramp(40.0, 100.0);

    std::vector<scalar_t> pmin(G), pmax(G), c_lin(G), c_nl(G), ramp_lim(G);
    for (int g = 0; g < G; ++g) {
        pmin[g] = std::round(dist_pmin(rng));
        pmax[g] = std::round(dist_pmax(rng));
        c_lin[g] = std::round(dist_lin_cost(rng) * 10.0) / 10.0;
        c_nl[g] = std::round(dist_nl_cost(rng));
        ramp_lim[g] = std::round(dist_ramp(rng));
    }

    // System load profile with diurnal fluctuation
    std::vector<scalar_t> demand(T);
    std::vector<scalar_t> reserve(T);
    scalar_t base_demand = G * 75.0;
    for (int t = 0; t < T; ++t) {
        double diurnal = 1.0 + 0.35 * std::sin(2.0 * M_PI * (t % 24) / 24.0);
        demand[t] = std::round(base_demand * diurnal);
        reserve[t] = std::round(demand[t] * 0.15); // 15% spinning reserve
    }

    // Variables per period t:
    // 1. Binary u_{g,t} in {0, 1}: commitment status
    // 2. Continuous p_{g,t} >= 0: dispatch output (MW)
    int vars_per_t = 2 * G;
    index_t num_cols = T * vars_per_t;

    lp.col_names.resize(num_cols);
    lp.c.assign(num_cols, 0.0);
    lp.col_lower.assign(num_cols, 0.0);
    lp.col_upper.assign(num_cols, model::Infinity);
    lp.var_types.assign(num_cols, model::VariableType::Continuous);

    auto get_u_idx = [&](int t, int g) -> index_t { return t * vars_per_t + g; };
    auto get_p_idx = [&](int t, int g) -> index_t { return t * vars_per_t + G + g; };

    index_t binary_count = 0;
    for (int t = 0; t < T; ++t) {
        for (int g = 0; g < G; ++g) {
            // Binary commitment u
            index_t u_col = get_u_idx(t, g);
            std::string u_name = "U_G" + std::to_string(g) + "_T" + std::to_string(t);
            lp.col_names[u_col] = u_name;
            lp.col_name_to_idx[u_name] = u_col;
            lp.var_types[u_col] = model::VariableType::Binary;
            lp.col_lower[u_col] = 0.0;
            lp.col_upper[u_col] = 1.0;
            lp.c[u_col] = c_nl[g];
            binary_count++;

            // Continuous power dispatch p
            index_t p_col = get_p_idx(t, g);
            std::string p_name = "P_G" + std::to_string(g) + "_T" + std::to_string(t);
            lp.col_names[p_col] = p_name;
            lp.col_name_to_idx[p_name] = p_col;
            lp.var_types[p_col] = model::VariableType::Continuous;
            lp.col_lower[p_col] = 0.0;
            lp.col_upper[p_col] = pmax[g];
            lp.c[p_col] = c_lin[g];
        }
    }

    // Constraints per period t:
    // 1. Power balance: sum_g p_{g,t} = demand[t] (T rows)
    // 2. Spinning reserve: sum_g pmax_g * u_{g,t} >= demand[t] + reserve[t] (T rows)
    // 3. Max generation: p_{g,t} - pmax_g * u_{g,t} <= 0 (G * T rows)
    // 4. Min generation: p_{g,t} - pmin_g * u_{g,t} >= 0 (G * T rows)
    // 5. Ramp-up (t > 0): p_{g,t} - p_{g,t-1} <= ramp_lim_g (G * (T - 1) rows)
    // 6. Ramp-down (t > 0): p_{g,t-1} - p_{g,t} <= ramp_lim_g (G * (T - 1) rows)
    index_t num_rows = 2 * T + 2 * G * T + 2 * G * (T > 1 ? T - 1 : 0);
    lp.row_names.resize(num_rows);
    lp.row_lower.assign(num_rows, 0.0);
    lp.row_upper.assign(num_rows, 0.0);
    lp.row_senses.resize(num_rows);

    std::vector<pipepye::sparse::TripletF64> triplets;
    index_t row_idx = 0;

    for (int t = 0; t < T; ++t) {
        // 1. Power balance
        {
            std::string rname = "DEMAND_T" + std::to_string(t);
            lp.row_names[row_idx] = rname;
            lp.row_name_to_idx[rname] = row_idx;
            lp.row_lower[row_idx] = demand[t];
            lp.row_upper[row_idx] = demand[t];
            lp.row_senses[row_idx] = model::RowSense::Equality;

            for (int g = 0; g < G; ++g) {
                triplets.push_back({row_idx, get_p_idx(t, g), 1.0});
            }
            row_idx++;
        }

        // 2. Spinning reserve
        {
            std::string rname = "RESERVE_T" + std::to_string(t);
            lp.row_names[row_idx] = rname;
            lp.row_name_to_idx[rname] = row_idx;
            lp.row_lower[row_idx] = demand[t] + reserve[t];
            lp.row_upper[row_idx] = model::Infinity;
            lp.row_senses[row_idx] = model::RowSense::GreaterEqual;

            for (int g = 0; g < G; ++g) {
                triplets.push_back({row_idx, get_u_idx(t, g), pmax[g]});
            }
            row_idx++;
        }

        // 3. Max generation & 4. Min generation
        for (int g = 0; g < G; ++g) {
            // Max: p - pmax * u <= 0
            std::string rname_max = "PMAX_G" + std::to_string(g) + "_T" + std::to_string(t);
            lp.row_names[row_idx] = rname_max;
            lp.row_name_to_idx[rname_max] = row_idx;
            lp.row_lower[row_idx] = -model::Infinity;
            lp.row_upper[row_idx] = 0.0;
            lp.row_senses[row_idx] = model::RowSense::LessEqual;

            triplets.push_back({row_idx, get_p_idx(t, g), 1.0});
            triplets.push_back({row_idx, get_u_idx(t, g), -pmax[g]});
            row_idx++;

            // Min: p - pmin * u >= 0
            std::string rname_min = "PMIN_G" + std::to_string(g) + "_T" + std::to_string(t);
            lp.row_names[row_idx] = rname_min;
            lp.row_name_to_idx[rname_min] = row_idx;
            lp.row_lower[row_idx] = 0.0;
            lp.row_upper[row_idx] = model::Infinity;
            lp.row_senses[row_idx] = model::RowSense::GreaterEqual;

            triplets.push_back({row_idx, get_p_idx(t, g), 1.0});
            triplets.push_back({row_idx, get_u_idx(t, g), -pmin[g]});
            row_idx++;
        }

        // 5. Ramp limits (t > 0)
        if (t > 0) {
            for (int g = 0; g < G; ++g) {
                // Ramp-up: p_t - p_{t-1} <= ramp_lim
                std::string rname_ru = "RAMPUP_G" + std::to_string(g) + "_T" + std::to_string(t);
                lp.row_names[row_idx] = rname_ru;
                lp.row_name_to_idx[rname_ru] = row_idx;
                lp.row_lower[row_idx] = -model::Infinity;
                lp.row_upper[row_idx] = ramp_lim[g];
                lp.row_senses[row_idx] = model::RowSense::LessEqual;

                triplets.push_back({row_idx, get_p_idx(t, g), 1.0});
                triplets.push_back({row_idx, get_p_idx(t - 1, g), -1.0});
                row_idx++;

                // Ramp-down: p_{t-1} - p_t <= ramp_lim
                std::string rname_rd = "RAMPDN_G" + std::to_string(g) + "_T" + std::to_string(t);
                lp.row_names[row_idx] = rname_rd;
                lp.row_name_to_idx[rname_rd] = row_idx;
                lp.row_lower[row_idx] = -model::Infinity;
                lp.row_upper[row_idx] = ramp_lim[g];
                lp.row_senses[row_idx] = model::RowSense::LessEqual;

                triplets.push_back({row_idx, get_p_idx(t - 1, g), 1.0});
                triplets.push_back({row_idx, get_p_idx(t, g), -1.0});
                row_idx++;
            }
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
    meta.problem_name = "Unit_Commitment";
    meta.instance_id = lp.name;
    meta.random_seed = params.seed;
    meta.scale = params.scale;
    meta.num_continuous_vars = num_cols - binary_count;
    meta.num_integer_vars = 0;
    meta.num_binary_vars = binary_count;
    meta.min_rhs = 0.0;
    meta.max_rhs = base_demand * 1.5;
    meta.min_bound = 0.0;
    meta.max_bound = 300.0;
    meta.expected_solver = "BranchAndBound";
    meta.expected_backend = "CPU";
    meta.prediction_rationale =
        "Time-indexed generator commitment requires discrete branching; continuous dispatch subproblems are efficiently resolved via warm-started Dual Simplex.";
    meta.populate_from_stats(stats);

    return {std::move(lp), std::move(meta)};
}

} // namespace pipepye::workloads
