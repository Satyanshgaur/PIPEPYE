#include <pipepye/milp/branch_and_bound.hpp>
#include <pipepye/workloads/case_c_refinery_scheduling.hpp>
#include <pipepye/workloads/case_d_unit_commitment.hpp>
#include <pipepye/utils/timer.hpp>
#include <random>

#include <iostream>
#include <fstream>
#include <filesystem>
#include <vector>
#include <string>
#include <sstream>
#include <iomanip>

namespace fs = std::filesystem;
using namespace pipepye;
using namespace pipepye::model;
using namespace pipepye::milp;
using namespace pipepye::workloads;

struct BenchmarkInstance {
    std::string name;
    std::string scale;
    LinearProgram lp;
};

struct BenchmarkRecord {
    std::string instance_name;
    std::string scale;
    index_t rows{0};
    index_t cols{0};
    size_t nnz{0};
    index_t num_binaries{0};
    index_t num_integers{0};

    std::string config_name;
    std::string status;
    scalar_t best_obj{0.0};
    scalar_t best_bound{0.0};
    scalar_t gap_pct{0.0};

    int nodes_explored{0};
    int nodes_created{0};
    int peak_tree_size{0};
    double nodes_per_sec{0.0};

    int total_lp_solves{0};
    int total_pivots{0};
    int warm_pivots{0};
    int cold_pivots{0};
    double pivot_reduction_pct{0.0};

    int cuts_added{0};
    int heuristic_sols{0};

    double root_relaxation_ms{0.0};
    double cut_time_ms{0.0};
    double heuristic_time_ms{0.0};
    double total_time_ms{0.0};

    double time_to_first_incumbent_ms{0.0};
    double time_to_gap_10pct_ms{0.0};
    double time_to_gap_1pct_ms{0.0};
};

LinearProgram make_knapsack_benchmark(int n_vars, uint64_t seed) {
    LinearProgram lp;
    lp.name = "KNAPSACK_" + std::to_string(n_vars);
    lp.is_maximization = false;
    lp.row_names = {"CAP"};
    lp.row_name_to_idx["CAP"] = 0;
    lp.row_lower = {-Infinity};
    lp.row_senses = {RowSense::LessEqual};

    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int> weight_dist(2, 10);
    std::uniform_int_distribution<int> value_dist(5, 25);

    scalar_t total_weight = 0.0;
    std::vector<scalar_t> weights(n_vars);
    std::vector<scalar_t> values(n_vars);
    for (int j = 0; j < n_vars; ++j) {
        weights[j] = static_cast<scalar_t>(weight_dist(rng));
        values[j] = static_cast<scalar_t>(value_dist(rng));
        total_weight += weights[j];
    }
    scalar_t capacity = std::floor(total_weight * 0.45);
    lp.row_upper = {capacity};

    lp.A_coo = sparse::COOMatrix(1, n_vars);
    for (int j = 0; j < n_vars; ++j) {
        std::string col = "X" + std::to_string(j + 1);
        lp.col_names.push_back(col);
        lp.col_name_to_idx[col] = j;
        lp.c.push_back(-values[j]); // Minimization of negative value
        lp.col_lower.push_back(0.0);
        lp.col_upper.push_back(1.0);
        lp.var_types.push_back(VariableType::Binary);
        lp.A_coo.add_entry(0, j, weights[j]);
    }

    auto csc = lp.A_coo.to_csc();
    lp.csc_col_ptr = csc.col_ptr_vector();
    lp.csc_row_ind = csc.row_ind_vector();
    lp.csc_values = csc.values_vector();

    auto csr = lp.A_coo.to_csr();
    lp.csr_row_ptr = csr.row_ptr_vector();
    lp.csr_col_ind = csr.col_ind_vector();
    lp.csr_values = csr.values_vector();

    return lp;
}

int main(int argc, char** argv) {
    std::cout << "===============================================================================\n"
              << "          PipePye Phase 7: Branch-and-Bound MILP Benchmark Harness             \n"
              << "===============================================================================\n"
              << "Ablation Matrix:\n"
              << "  1. Baseline Cold-Start (DFS, Cold Simplex, No Cuts, No Heuristics)\n"
              << "  2. Warm-Start Only (BestBound, Warm Simplex PFI)\n"
              << "  3. Warm-Start + Root Presolve\n"
              << "  4. Warm-Start + Presolve + Gomory Cuts\n"
              << "  5. Full Pipeline (Warm + Presolve + Cuts + All Primal Heuristics)\n"
              << "  6. Pseudocost Branching (Full Pipeline + PseudoCost Selection)\n"
              << "-------------------------------------------------------------------------------\n\n";

    fs::create_directories("reports");

    // 1. Build test instance ladder
    std::vector<BenchmarkInstance> instances;

    // A. Knapsack
    instances.push_back({"Knapsack_5", "Toy", make_knapsack_benchmark(5, 101)});
    instances.push_back({"Knapsack_10", "Small", make_knapsack_benchmark(10, 202)});

    // B. Case C: Refinery Scheduling
    {
        auto [lp_toy, meta_toy] = RefinerySchedulingGenerator::generate(
            RefinerySchedulingParams::ForScale(InstanceScale::Toy));
        instances.push_back({"RefineryScheduling", "Toy", lp_toy});

        auto [lp_small, meta_small] = RefinerySchedulingGenerator::generate(
            RefinerySchedulingParams::ForScale(InstanceScale::Small));
        instances.push_back({"RefineryScheduling", "Small", lp_small});
    }

    // C. Case D: Unit Commitment
    {
        auto [lp_toy, meta_toy] = UnitCommitmentGenerator::generate(
            UnitCommitmentParams::ForScale(InstanceScale::Toy));
        instances.push_back({"UnitCommitment", "Toy", lp_toy});

        auto [lp_small, meta_small] = UnitCommitmentGenerator::generate(
            UnitCommitmentParams::ForScale(InstanceScale::Small));
        instances.push_back({"UnitCommitment", "Small", lp_small});
    }

    // 2. Define Ablation Configurations
    struct ConfigEntry {
        std::string name;
        MILPConfig config;
    };

    std::vector<ConfigEntry> configs;

    // Config 1: Baseline Cold-Start DFS
    {
        MILPConfig c;
        c.node_selection = NodeSelectionStrategy::DepthFirst;
        c.use_warm_start = false;
        c.enable_root_presolve = false;
        c.cutting_strategy = CuttingPlaneStrategy::None;
        c.heuristic_strategy = HeuristicStrategy::None;
        c.branching_strategy = BranchingStrategy::MostFractional;
        c.max_nodes = 400;
        c.time_limit_sec = 10.0;
        configs.push_back({"1_Baseline_ColdDFS", c});
    }

    // Config 2: Warm-Start Only BestBound
    {
        MILPConfig c;
        c.node_selection = NodeSelectionStrategy::BestBound;
        c.use_warm_start = true;
        c.enable_root_presolve = false;
        c.cutting_strategy = CuttingPlaneStrategy::None;
        c.heuristic_strategy = HeuristicStrategy::None;
        c.branching_strategy = BranchingStrategy::MostFractional;
        c.max_nodes = 400;
        c.time_limit_sec = 10.0;
        configs.push_back({"2_WarmStart_BestBound", c});
    }

    // Config 3: Warm-Start + Root Presolve
    {
        MILPConfig c;
        c.node_selection = NodeSelectionStrategy::BestBound;
        c.use_warm_start = true;
        c.enable_root_presolve = true;
        c.cutting_strategy = CuttingPlaneStrategy::None;
        c.heuristic_strategy = HeuristicStrategy::None;
        c.branching_strategy = BranchingStrategy::MostFractional;
        c.max_nodes = 400;
        c.time_limit_sec = 10.0;
        configs.push_back({"3_Warm_Presolve", c});
    }

    // Config 4: Warm-Start + Presolve + Gomory Cuts
    {
        MILPConfig c;
        c.node_selection = NodeSelectionStrategy::BestBound;
        c.use_warm_start = true;
        c.enable_root_presolve = true;
        c.cutting_strategy = CuttingPlaneStrategy::GomoryFractional;
        c.max_root_cuts = 15;
        c.heuristic_strategy = HeuristicStrategy::None;
        c.branching_strategy = BranchingStrategy::MostFractional;
        c.max_nodes = 400;
        c.time_limit_sec = 10.0;
        configs.push_back({"4_Warm_Presolve_Cuts", c});
    }

    // Config 5: Full Pipeline with Primal Heuristics
    {
        MILPConfig c;
        c.node_selection = NodeSelectionStrategy::BestBound;
        c.use_warm_start = true;
        c.enable_root_presolve = true;
        c.cutting_strategy = CuttingPlaneStrategy::GomoryFractional;
        c.max_root_cuts = 15;
        c.heuristic_strategy = HeuristicStrategy::All;
        c.branching_strategy = BranchingStrategy::MostFractional;
        c.max_nodes = 400;
        c.time_limit_sec = 10.0;
        configs.push_back({"5_FullPipeline_AllHeuristics", c});
    }

    // Config 6: Pseudocost Branching
    {
        MILPConfig c;
        c.node_selection = NodeSelectionStrategy::BestBound;
        c.use_warm_start = true;
        c.enable_root_presolve = true;
        c.cutting_strategy = CuttingPlaneStrategy::GomoryFractional;
        c.max_root_cuts = 15;
        c.heuristic_strategy = HeuristicStrategy::All;
        c.branching_strategy = BranchingStrategy::PseudoCost;
        c.max_nodes = 400;
        c.time_limit_sec = 10.0;
        configs.push_back({"6_FullPipeline_PseudoCost", c});
    }

    std::vector<BenchmarkRecord> records;

    // 3. Run Benchmark Matrix
    for (const auto& inst : instances) {
        index_t n_bin = 0, n_int = 0;
        for (auto vt : inst.lp.var_types) {
            if (vt == VariableType::Binary) n_bin++;
            else if (vt == VariableType::Integer) n_int++;
        }

        std::cout << "\n>>> Benchmarking Instance: " << inst.name << " [" << inst.scale << "]"
                  << " (" << inst.lp.num_rows() << " rows, " << inst.lp.num_cols() << " cols, "
                  << inst.lp.num_nonzeros() << " nnz, " << n_bin << " binary, " << n_int << " int)\n";

        for (const auto& cfg_entry : configs) {
            BranchAndBoundSolver solver(cfg_entry.config);
            MILPResult res = solver.solve(inst.lp);

            BenchmarkRecord rec;
            rec.instance_name = inst.name;
            rec.scale = inst.scale;
            rec.rows = inst.lp.num_rows();
            rec.cols = inst.lp.num_cols();
            rec.nnz = inst.lp.num_nonzeros();
            rec.num_binaries = n_bin;
            rec.num_integers = n_int;

            rec.config_name = cfg_entry.name;
            rec.status = to_string(res.status);
            rec.best_obj = res.best_objective;
            rec.best_bound = res.best_bound;
            rec.gap_pct = res.mip_gap * 100.0;

            rec.nodes_explored = res.nodes_explored;
            rec.nodes_created = res.nodes_created;
            rec.peak_tree_size = res.peak_tree_size;
            rec.nodes_per_sec = res.nodes_per_second;

            rec.total_lp_solves = res.total_lp_solves;
            rec.total_pivots = res.total_pivots;
            rec.warm_pivots = res.warm_start_pivots;
            rec.cold_pivots = res.cold_start_pivots;
            rec.pivot_reduction_pct = res.pivot_reduction_ratio() * 100.0;

            rec.cuts_added = res.root_cuts_added;
            rec.heuristic_sols = res.heuristic_solutions_found;

            rec.root_relaxation_ms = res.root_relaxation_time_ms;
            rec.cut_time_ms = res.cut_generation_time_ms;
            rec.heuristic_time_ms = res.heuristic_time_ms;
            rec.total_time_ms = res.total_time_ms;

            rec.time_to_first_incumbent_ms = res.time_to_first_incumbent_ms;
            rec.time_to_gap_10pct_ms = res.time_to_gap_10pct_ms;
            rec.time_to_gap_1pct_ms = res.time_to_gap_1pct_ms;

            records.push_back(rec);

            std::cout << "  [" << std::setw(28) << std::left << cfg_entry.name << "] "
                      << "Status=" << std::setw(9) << rec.status << " "
                      << "Obj=" << std::scientific << std::setprecision(4) << rec.best_obj << " "
                      << "Gap=" << std::fixed << std::setprecision(2) << rec.gap_pct << "% "
                      << "Nodes=" << std::setw(4) << rec.nodes_explored << " "
                      << "Pivots=" << std::setw(5) << rec.total_pivots << " "
                      << "Cuts=" << rec.cuts_added << " "
                      << "Heur=" << rec.heuristic_sols << " "
                      << "Time=" << std::setprecision(2) << rec.total_time_ms << "ms"
                      << "\n";
        }
    }

    // 4. Export CSV
    std::string csv_path = "reports/milp_benchmark.csv";
    std::ofstream csv(csv_path);
    if (csv.is_open()) {
        csv << "instance_name,scale,rows,cols,nnz,binaries,integers,config_name,status,"
            << "best_objective,best_bound,gap_pct,nodes_explored,nodes_created,peak_tree_size,"
            << "nodes_per_sec,total_lp_solves,total_pivots,cuts_added,heuristic_sols,"
            << "root_relaxation_ms,cut_time_ms,heuristic_time_ms,total_time_ms,"
            << "time_to_first_incumbent_ms,time_to_gap_10pct_ms\n";

        for (const auto& r : records) {
            csv << r.instance_name << ","
                << r.scale << ","
                << r.rows << ","
                << r.cols << ","
                << r.nnz << ","
                << r.num_binaries << ","
                << r.num_integers << ","
                << r.config_name << ","
                << r.status << ","
                << std::scientific << std::setprecision(8) << r.best_obj << ","
                << r.best_bound << ","
                << std::fixed << std::setprecision(4) << r.gap_pct << ","
                << r.nodes_explored << ","
                << r.nodes_created << ","
                << r.peak_tree_size << ","
                << std::setprecision(2) << r.nodes_per_sec << ","
                << r.total_lp_solves << ","
                << r.total_pivots << ","
                << r.cuts_added << ","
                << r.heuristic_sols << ","
                << r.root_relaxation_ms << ","
                << r.cut_time_ms << ","
                << r.heuristic_time_ms << ","
                << r.total_time_ms << ","
                << r.time_to_first_incumbent_ms << ","
                << r.time_to_gap_10pct_ms << "\n";
        }
        std::cout << "\n[CSV Export] Successfully written to: " << csv_path << "\n";
    }

    // 5. Export JSON
    std::string json_path = "reports/milp_benchmark.json";
    std::ofstream js(json_path);
    if (js.is_open()) {
        js << "[\n";
        for (size_t i = 0; i < records.size(); ++i) {
            const auto& r = records[i];
            js << "  {\n"
               << "    \"instance_name\": \"" << r.instance_name << "\",\n"
               << "    \"scale\": \"" << r.scale << "\",\n"
               << "    \"rows\": " << r.rows << ",\n"
               << "    \"cols\": " << r.cols << ",\n"
               << "    \"nnz\": " << r.nnz << ",\n"
               << "    \"binaries\": " << r.num_binaries << ",\n"
               << "    \"integers\": " << r.num_integers << ",\n"
               << "    \"config_name\": \"" << r.config_name << "\",\n"
               << "    \"status\": \"" << r.status << "\",\n"
               << "    \"best_objective\": " << std::scientific << std::setprecision(8) << r.best_obj << ",\n"
               << "    \"best_bound\": " << r.best_bound << ",\n"
               << "    \"gap_pct\": " << std::fixed << std::setprecision(4) << r.gap_pct << ",\n"
               << "    \"nodes_explored\": " << r.nodes_explored << ",\n"
               << "    \"nodes_created\": " << r.nodes_created << ",\n"
               << "    \"peak_tree_size\": " << r.peak_tree_size << ",\n"
               << "    \"nodes_per_sec\": " << std::setprecision(2) << r.nodes_per_sec << ",\n"
               << "    \"total_lp_solves\": " << r.total_lp_solves << ",\n"
               << "    \"total_pivots\": " << r.total_pivots << ",\n"
               << "    \"cuts_added\": " << r.cuts_added << ",\n"
               << "    \"heuristic_solutions\": " << r.heuristic_sols << ",\n"
               << "    \"root_relaxation_ms\": " << r.root_relaxation_ms << ",\n"
               << "    \"cut_time_ms\": " << r.cut_time_ms << ",\n"
               << "    \"heuristic_time_ms\": " << r.heuristic_time_ms << ",\n"
               << "    \"total_time_ms\": " << r.total_time_ms << ",\n"
               << "    \"time_to_first_incumbent_ms\": " << r.time_to_first_incumbent_ms << ",\n"
               << "    \"time_to_gap_10pct_ms\": " << r.time_to_gap_10pct_ms << "\n"
               << "  }" << (i + 1 < records.size() ? "," : "") << "\n";
        }
        js << "]\n";
        std::cout << "[JSON Export] Successfully written to: " << json_path << "\n";
    }

    return 0;
}
