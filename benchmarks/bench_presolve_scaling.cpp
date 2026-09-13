#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <string>
#include <filesystem>
#include <pipepye/model/mps_parser.hpp>
#include <pipepye/model/lp_model.hpp>
#include <pipepye/pipeline/model_pipeline.hpp>
#include <pipepye/sparse/matrix_generator.hpp>
#include <pipepye/utils/timer.hpp>

using namespace pipepye;
using namespace pipepye::model;
using namespace pipepye::pipeline;
using namespace pipepye::analysis;
using namespace pipepye::sparse;

namespace {

struct ExperimentResult {
    std::string model_name;
    std::string category;
    index_t raw_rows{0};
    index_t raw_cols{0};
    size_t raw_nnz{0};
    double raw_density{0.0};
    double raw_dynamic_range{0.0};
    double raw_row_norm_ratio{0.0};
    double raw_col_norm_ratio{0.0};
    double raw_spectral_norm{0.0};

    index_t prep_rows{0};
    index_t prep_cols{0};
    size_t prep_nnz{0};
    double prep_density{0.0};
    double prep_dynamic_range{0.0};
    double prep_row_norm_ratio{0.0};
    double prep_col_norm_ratio{0.0};
    double prep_spectral_norm{0.0};

    index_t eliminated_rows{0};
    index_t eliminated_cols{0};
    size_t eliminated_nnz{0};
    index_t fixed_vars{0};
    index_t bounds_tightened{0};
    index_t redundant_rows{0};
    int presolve_passes{0};
    double presolve_ms{0.0};

    int scaling_iterations{0};
    double scaling_ms{0.0};
    double total_pipeline_ms{0.0};
};

std::string find_mps(const std::string& filename) {
    std::vector<std::string> dirs = {
        "tests/data/mps/netlib",
        "../tests/data/mps/netlib",
        "../../tests/data/mps/netlib",
        "/home/satyansh/pipepye/tests/data/mps/netlib"
    };
    for (const auto& d : dirs) {
        std::filesystem::path p = std::filesystem::path(d) / filename;
        if (std::filesystem::exists(p)) return p.string();
    }
    return "";
}

LinearProgram make_synthetic_lp(const std::string& name, const std::string& pattern,
                                index_t rows, index_t cols, double density,
                                scalar_t min_v, scalar_t max_v, uint32_t seed = 42) {
    COOMatrix coo(rows, cols);
    if (pattern == "banded") {
        coo = MatrixGenerator::generate_banded(rows, cols, 15, 15, min_v, max_v, seed);
    } else if (pattern == "block") {
        coo = MatrixGenerator::generate_block_diagonal(8, rows / 8, cols / 8, 0.05, 0.0005, min_v, max_v, seed);
    } else if (pattern == "staircase") {
        coo = MatrixGenerator::generate_staircase(10, rows / 10, cols / 10, 0.04, min_v, max_v, seed);
    } else if (pattern == "irregular") {
        size_t nnz = static_cast<size_t>(rows * cols * density);
        coo = MatrixGenerator::generate_irregular(rows, cols, nnz, 0.05, 0.50, min_v, max_v, seed);
    } else if (pattern == "ill_conditioned") {
        coo = MatrixGenerator::generate_random(rows, cols, density, 1e-6, 1e6, seed);
    } else {
        coo = MatrixGenerator::generate_random(rows, cols, density, min_v, max_v, seed);
    }

    index_t m = coo.num_rows();
    index_t n = coo.num_cols();

    LinearProgram lp;
    lp.name = name;
    lp.c.assign(n, 1.0);
    lp.col_lower.assign(n, 0.0);
    lp.col_upper.assign(n, 100.0);
    lp.row_lower.assign(m, 0.0);
    lp.row_upper.assign(m, 100.0);
    lp.row_senses.assign(m, RowSense::LessEqual);
    lp.var_types.assign(n, VariableType::Continuous);
    for (index_t j = 0; j < n; ++j) lp.col_names.push_back("x" + std::to_string(j));
    for (index_t i = 0; i < m; ++i) lp.row_names.push_back("c" + std::to_string(i));

    lp.A_coo = coo;
    auto csr = coo.to_csr();
    lp.csr_row_ptr.assign(csr.row_ptr().begin(), csr.row_ptr().end());
    lp.csr_col_ind.assign(csr.col_ind().begin(), csr.col_ind().end());
    lp.csr_values.assign(csr.values().begin(), csr.values().end());

    auto csc = coo.to_csc();
    lp.csc_col_ptr.assign(csc.col_ptr().begin(), csc.col_ptr().end());
    lp.csc_row_ind.assign(csc.row_ind().begin(), csc.row_ind().end());
    lp.csc_values.assign(csc.values().begin(), csc.values().end());

    return lp;
}

LinearProgram make_degenerate_lp() {
    LinearProgram lp;
    lp.name = "synth_degenerate";
    index_t rows = 120;
    index_t cols = 150;
    lp.c.assign(cols, 1.0);
    lp.col_lower.assign(cols, 0.0);
    lp.col_upper.assign(cols, 100.0);
    lp.row_lower.assign(rows, 0.0);
    lp.row_upper.assign(rows, 50.0);
    lp.row_senses.assign(rows, RowSense::LessEqual);
    lp.var_types.assign(cols, VariableType::Continuous);
    for (index_t j = 0; j < cols; ++j) lp.col_names.push_back("x" + std::to_string(j));
    for (index_t i = 0; i < rows; ++i) lp.row_names.push_back("c" + std::to_string(i));

    // Create fixed variables (10 vars)
    for (index_t j = 0; j < 10; ++j) {
        lp.col_lower[j] = 3.5;
        lp.col_upper[j] = 3.5;
    }

    COOMatrix coo(rows, cols);
    // Add sparse entries for active rows 20..119
    for (index_t i = 20; i < rows; ++i) {
        coo.add_entry(i, i % cols, 2.0);
        coo.add_entry(i, (i + 5) % cols, -1.5);
    }
    // Rows 0..9 are empty rows with bounds containing 0
    // Rows 10..19 are redundant duplicates of row 20
    for (index_t i = 10; i < 20; ++i) {
        coo.add_entry(i, 20 % cols, 2.0);
        coo.add_entry(i, (20 + 5) % cols, -1.5);
    }

    lp.A_coo = coo;
    auto csr = coo.to_csr();
    lp.csr_row_ptr.assign(csr.row_ptr().begin(), csr.row_ptr().end());
    lp.csr_col_ind.assign(csr.col_ind().begin(), csr.col_ind().end());
    lp.csr_values.assign(csr.values().begin(), csr.values().end());

    auto csc = coo.to_csc();
    lp.csc_col_ptr.assign(csc.col_ptr().begin(), csc.col_ptr().end());
    lp.csc_row_ind.assign(csc.row_ind().begin(), csc.row_ind().end());
    lp.csc_values.assign(csc.values().begin(), csc.values().end());

    return lp;
}

ExperimentResult run_experiment(const std::string& category, const LinearProgram& lp) {
    ExperimentResult res;
    res.model_name = lp.name;
    res.category = category;

    // Path A: Raw Characterization
    ProblemStats raw_stats = ProblemAnalyzer::analyze(lp);
    res.raw_rows = raw_stats.num_rows;
    res.raw_cols = raw_stats.num_cols;
    res.raw_nnz = raw_stats.num_nonzeros;
    res.raw_density = raw_stats.density;
    res.raw_dynamic_range = raw_stats.dynamic_range;
    res.raw_row_norm_ratio = raw_stats.conditioning_proxy.row_norm_ratio_proxy;
    res.raw_col_norm_ratio = raw_stats.conditioning_proxy.col_norm_ratio_proxy;
    res.raw_spectral_norm = raw_stats.conditioning_proxy.spectral_norm_estimate;

    // Path B: Presolve + Scaling Pipeline
    PipelineConfig cfg;
    cfg.enable_presolve = true;
    cfg.enable_scaling = true;
    cfg.compute_characterization = true;

    utils::CPUTimer timer;
    timer.start();
    auto prep_res = ModelPipeline::prepare(lp, cfg);
    timer.stop();
    res.total_pipeline_ms = timer.elapsed_milliseconds();

    if (prep_res.is_ok()) {
        const auto& prep = prep_res.value();
        res.prep_rows = prep.problem_stats.num_rows;
        res.prep_cols = prep.problem_stats.num_cols;
        res.prep_nnz = prep.problem_stats.num_nonzeros;
        res.prep_density = prep.problem_stats.density;
        res.prep_dynamic_range = prep.problem_stats.dynamic_range;
        res.prep_row_norm_ratio = prep.problem_stats.conditioning_proxy.row_norm_ratio_proxy;
        res.prep_col_norm_ratio = prep.problem_stats.conditioning_proxy.col_norm_ratio_proxy;
        res.prep_spectral_norm = prep.problem_stats.conditioning_proxy.spectral_norm_estimate;

        res.eliminated_rows = prep.presolve_stats.eliminated_rows();
        res.eliminated_cols = prep.presolve_stats.eliminated_cols();
        res.eliminated_nnz = prep.presolve_stats.eliminated_nonzeros();
        res.fixed_vars = prep.presolve_stats.total_variables_fixed();
        res.bounds_tightened = prep.presolve_stats.total_bounds_tightened();
        res.redundant_rows = prep.presolve_stats.total_redundant_rows();
        res.presolve_passes = prep.presolve_stats.total_passes_executed;
        res.presolve_ms = prep.presolve_stats.total_elapsed_ms;

        res.scaling_iterations = prep.scaling.iterations_performed;
        res.scaling_ms = prep.scaling.elapsed_ms;
    }

    return res;
}

} // namespace

int main() {
    std::cout << "================================================================================\n";
    std::cout << "          PIPEPYE AUTOMATED BEFORE / AFTER BENCHMARK RUNNER                    \n";
    std::cout << "================================================================================\n";

    std::vector<std::pair<std::string, LinearProgram>> suite;

    // 1. Netlib Standard Benchmarks
    std::vector<std::pair<std::string, std::string>> netlib_files = {
        {"afiro.mps", "Netlib_Easy"},
        {"adlittle.mps", "Netlib_Standard"},
        {"beaconfd.mps", "Netlib_LargeSparse"},
        {"blend.mps", "Netlib_Standard"},
        {"bandm.mps", "Netlib_Banded"}
    };

    for (const auto& [fname, cat] : netlib_files) {
        std::string p = find_mps(fname);
        if (!p.empty()) {
            LinearProgram lp;
            if (MPSParser::parse_file(p, lp).is_ok()) {
                suite.push_back({cat, std::move(lp)});
            }
        }
    }

    // 2. Synthetic Benchmarks (Ill-conditioned, Degenerate, Structural)
    suite.push_back({"Synthetic_IllConditioned",
                     make_synthetic_lp("synth_ill_cond", "ill_conditioned", 1000, 1000, 0.01, 1e-6, 1e6, 42)});
    suite.push_back({"Synthetic_Degenerate", make_degenerate_lp()});
    suite.push_back({"Synthetic_Banded",
                     make_synthetic_lp("synth_banded", "banded", 2000, 2000, 0.01, -5.0, 5.0, 43)});
    suite.push_back({"Synthetic_BlockDiagonal",
                     make_synthetic_lp("synth_block_diag", "block", 2000, 2000, 0.01, -5.0, 5.0, 44)});
    suite.push_back({"Synthetic_Staircase",
                     make_synthetic_lp("synth_staircase", "staircase", 2000, 2000, 0.01, -5.0, 5.0, 45)});
    suite.push_back({"Synthetic_IrregularHub",
                     make_synthetic_lp("synth_irregular", "irregular", 2000, 2000, 0.005, -5.0, 5.0, 46)});

    std::vector<ExperimentResult> results;
    for (const auto& [cat, lp] : suite) {
        std::cout << "Running before/after characterization: [" << cat << "] " << lp.name << "..." << std::flush;
        auto res = run_experiment(cat, lp);
        results.push_back(res);
        std::cout << " Done (" << std::fixed << std::setprecision(2) << res.total_pipeline_ms << " ms)\n";
    }

    // Print Formatted Summary Table
    std::cout << "\n========================================================================================================================\n";
    std::cout << std::left << std::setw(20) << "Model"
              << std::setw(15) << "Raw Size"
              << std::setw(15) << "Prep Size"
              << std::setw(12) << "Row Red%"
              << std::setw(12) << "Col Red%"
              << std::setw(15) << "Raw Range"
              << std::setw(15) << "Prep Range"
              << std::setw(12) << "Ruiz Iter"
              << std::setw(10) << "Time (ms)" << "\n";
    std::cout << "------------------------------------------------------------------------------------------------------------------------\n";

    for (const auto& r : results) {
        std::ostringstream s_raw_sz, s_prep_sz, s_raw_rng, s_prep_rng;
        s_raw_sz << r.raw_rows << "x" << r.raw_cols;
        s_prep_sz << r.prep_rows << "x" << r.prep_cols;
        s_raw_rng << std::scientific << std::setprecision(1) << r.raw_dynamic_range;
        s_prep_rng << std::scientific << std::setprecision(1) << r.prep_dynamic_range;

        double r_red = r.raw_rows > 0 ? (1.0 - static_cast<double>(r.prep_rows) / r.raw_rows) * 100.0 : 0.0;
        double c_red = r.raw_cols > 0 ? (1.0 - static_cast<double>(r.prep_cols) / r.raw_cols) * 100.0 : 0.0;

        std::cout << std::left << std::setw(20) << r.model_name
                  << std::setw(15) << s_raw_sz.str()
                  << std::setw(15) << s_prep_sz.str()
                  << std::setw(12) << std::fixed << std::setprecision(1) << r_red
                  << std::setw(12) << std::fixed << std::setprecision(1) << c_red
                  << std::setw(15) << s_raw_rng.str()
                  << std::setw(15) << s_prep_rng.str()
                  << std::setw(12) << r.scaling_iterations
                  << std::setw(10) << std::fixed << std::setprecision(2) << r.total_pipeline_ms << "\n";
    }
    std::cout << "========================================================================================================================\n";

    // Write Machine-Readable JSON and CSV reports
    std::filesystem::create_directories("reports");

    // CSV
    std::ofstream csv("reports/presolve_scaling_benchmark.csv");
    if (csv.is_open()) {
        csv << "model_name,category,raw_rows,raw_cols,raw_nnz,raw_density,raw_dynamic_range,"
            << "raw_row_norm_ratio,raw_col_norm_ratio,raw_spectral_norm,"
            << "prep_rows,prep_cols,prep_nnz,prep_density,prep_dynamic_range,"
            << "prep_row_norm_ratio,prep_col_norm_ratio,prep_spectral_norm,"
            << "eliminated_rows,eliminated_cols,eliminated_nnz,fixed_vars,bounds_tightened,"
            << "redundant_rows,presolve_passes,presolve_ms,scaling_iterations,scaling_ms,total_pipeline_ms\n";

        for (const auto& r : results) {
            csv << r.model_name << "," << r.category << ","
                << r.raw_rows << "," << r.raw_cols << "," << r.raw_nnz << "," << r.raw_density << "," << r.raw_dynamic_range << ","
                << r.raw_row_norm_ratio << "," << r.raw_col_norm_ratio << "," << r.raw_spectral_norm << ","
                << r.prep_rows << "," << r.prep_cols << "," << r.prep_nnz << "," << r.prep_density << "," << r.prep_dynamic_range << ","
                << r.prep_row_norm_ratio << "," << r.prep_col_norm_ratio << "," << r.prep_spectral_norm << ","
                << r.eliminated_rows << "," << r.eliminated_cols << "," << r.eliminated_nnz << ","
                << r.fixed_vars << "," << r.bounds_tightened << "," << r.redundant_rows << ","
                << r.presolve_passes << "," << r.presolve_ms << ","
                << r.scaling_iterations << "," << r.scaling_ms << "," << r.total_pipeline_ms << "\n";
        }
        std::cout << "\nWrote CSV results to: reports/presolve_scaling_benchmark.csv\n";
    }

    // JSON
    std::ofstream json("reports/presolve_scaling_benchmark.json");
    if (json.is_open()) {
        json << "[\n";
        for (size_t i = 0; i < results.size(); ++i) {
            const auto& r = results[i];
            json << "  {\n"
                 << "    \"model_name\": \"" << r.model_name << "\",\n"
                 << "    \"category\": \"" << r.category << "\",\n"
                 << "    \"raw\": {\"rows\": " << r.raw_rows << ", \"cols\": " << r.raw_cols
                 << ", \"nnz\": " << r.raw_nnz << ", \"dynamic_range\": " << r.raw_dynamic_range
                 << ", \"row_norm_ratio\": " << r.raw_row_norm_ratio << ", \"col_norm_ratio\": " << r.raw_col_norm_ratio
                 << ", \"spectral_norm\": " << r.raw_spectral_norm << "},\n"
                 << "    \"prepared\": {\"rows\": " << r.prep_rows << ", \"cols\": " << r.prep_cols
                 << ", \"nnz\": " << r.prep_nnz << ", \"dynamic_range\": " << r.prep_dynamic_range
                 << ", \"row_norm_ratio\": " << r.prep_row_norm_ratio << ", \"col_norm_ratio\": " << r.prep_col_norm_ratio
                 << ", \"spectral_norm\": " << r.prep_spectral_norm << "},\n"
                 << "    \"presolve\": {\"eliminated_rows\": " << r.eliminated_rows
                 << ", \"eliminated_cols\": " << r.eliminated_cols << ", \"eliminated_nnz\": " << r.eliminated_nnz
                 << ", \"fixed_vars\": " << r.fixed_vars << ", \"bounds_tightened\": " << r.bounds_tightened
                 << ", \"redundant_rows\": " << r.redundant_rows << ", \"passes\": " << r.presolve_passes
                 << ", \"time_ms\": " << r.presolve_ms << "},\n"
                 << "    \"scaling\": {\"iterations\": " << r.scaling_iterations << ", \"time_ms\": " << r.scaling_ms << "},\n"
                 << "    \"total_pipeline_ms\": " << r.total_pipeline_ms << "\n"
                 << "  }" << (i + 1 < results.size() ? "," : "") << "\n";
        }
        json << "]\n";
        std::cout << "Wrote JSON results to: reports/presolve_scaling_benchmark.json\n";
    }

    return 0;
}
