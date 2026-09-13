#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <iomanip>
#include <filesystem>
#include <pipepye/model/mps_parser.hpp>
#include <pipepye/model/lp_model.hpp>
#include <pipepye/analysis/problem_analyzer.hpp>
#include <pipepye/pipeline/model_pipeline.hpp>
#include <pipepye/sparse/matrix_generator.hpp>

using namespace pipepye;
using namespace pipepye::model;
using namespace pipepye::analysis;
using namespace pipepye::pipeline;
using namespace pipepye::sparse;

namespace {

void print_help(const char* prog) {
    std::cout << "PipePye Unified Model Inspection & Characterization Tool\n\n"
              << "Usage:\n"
              << "  " << prog << " <model.mps> [options]\n"
              << "  " << prog << " --synth <type> [options]\n\n"
              << "Options:\n"
              << "  --one-line           Print only the canonical single-line summary banner\n"
              << "  --full               Print full multi-section analytical report (default)\n"
              << "  --before-after       Run presolve + scaling pipeline and report before/after comparison\n"
              << "  --json <file>        Save machine-readable characterization JSON to file\n"
              << "  --synth <pattern>    Generate synthetic matrix (random, banded, block, staircase, irregular)\n"
              << "  --rows <N>           Synthetic rows (default: 5000)\n"
              << "  --cols <N>           Synthetic columns (default: 5000)\n"
              << "  --density <D>        Synthetic density (default: 0.005)\n"
              << "  --help, -h           Show this help message\n";
}

LinearProgram make_synthetic_lp(const std::string& pattern, index_t rows, index_t cols, double density) {
    COOMatrix coo(rows, cols);
    if (pattern == "banded") {
        coo = MatrixGenerator::generate_banded(rows, cols, 10, 10, -5.0, 5.0);
    } else if (pattern == "block") {
        coo = MatrixGenerator::generate_block_diagonal(10, rows / 10, cols / 10, 0.05, 0.0005);
    } else if (pattern == "staircase") {
        coo = MatrixGenerator::generate_staircase(10, rows / 10, cols / 10, 0.05);
    } else if (pattern == "irregular") {
        size_t nnz = static_cast<size_t>(rows * cols * density);
        coo = MatrixGenerator::generate_irregular(rows, cols, nnz);
    } else {
        coo = MatrixGenerator::generate_random(rows, cols, density, -10.0, 10.0);
    }

    index_t m = coo.num_rows();
    index_t n = coo.num_cols();

    LinearProgram lp;
    lp.name = "synthetic_" + pattern;
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

std::string escape_json(const std::string& s) {
    std::string out;
    for (char c : s) {
        if (c == '"') out += "\\\"";
        else if (c == '\\') out += "\\\\";
        else if (c == '\n') out += "\\n";
        else out += c;
    }
    return out;
}

std::string stats_to_json(const std::string& name, const ProblemStats& s) {
    std::ostringstream ss;
    ss << "{\n"
       << "  \"model_name\": \"" << escape_json(name) << "\",\n"
       << "  \"summary_banner\": \"" << escape_json(s.format_one_line_summary()) << "\",\n"
       << "  \"num_rows\": " << s.num_rows << ",\n"
       << "  \"num_cols\": " << s.num_cols << ",\n"
       << "  \"num_nonzeros\": " << s.num_nonzeros << ",\n"
       << "  \"density\": " << s.density << ",\n"
       << "  \"min_abs_coeff\": " << s.min_abs_coeff << ",\n"
       << "  \"max_abs_coeff\": " << s.max_abs_coeff << ",\n"
       << "  \"dynamic_range\": " << s.dynamic_range << ",\n"
       << "  \"dynamic_range_orders\": " << s.dynamic_range_orders << ",\n"
       << "  \"row_norm_ratio_proxy\": " << s.conditioning_proxy.row_norm_ratio_proxy << ",\n"
       << "  \"col_norm_ratio_proxy\": " << s.conditioning_proxy.col_norm_ratio_proxy << ",\n"
       << "  \"spectral_norm_estimate\": " << s.conditioning_proxy.spectral_norm_estimate << ",\n"
       << "  \"spectral_cond_proxy\": " << s.conditioning_proxy.spectral_conditioning_proxy << ",\n"
       << "  \"row_imbalance_ratio\": " << s.row_length_imbalance << ",\n"
       << "  \"row_imbalance_rating\": \"" << s.row_imbalance_rating() << "\",\n"
       << "  \"row_gini_coefficient\": " << s.row_length_gini << ",\n"
       << "  \"staircase_score\": " << s.staircase_score << ",\n"
       << "  \"normalized_bandwidth\": " << s.normalized_bandwidth << ",\n"
       << "  \"connected_components\": " << s.num_connected_components << ",\n"
       << "  \"estimated_host_ram_mb\": " << s.estimated_host_ram_mb << ",\n"
       << "  \"estimated_gpu_vram_mb\": " << s.estimated_gpu_vram_mb << ",\n"
       << "  \"recommended_engine\": \"" << to_string(s.recommended_engine) << "\",\n"
       << "  \"recommendation_reason\": \"" << escape_json(s.recommendation_reason) << "\"\n"
       << "}";
    return ss.str();
}

} // namespace

int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_help(argv[0]);
        return 1;
    }

    std::string mps_file;
    std::string synth_type;
    index_t synth_rows = 5000;
    index_t synth_cols = 5000;
    double synth_density = 0.005;
    bool one_line_only = false;
    bool before_after = false;
    std::string json_output_file;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_help(argv[0]);
            return 0;
        } else if (arg == "--one-line") {
            one_line_only = true;
        } else if (arg == "--full") {
            one_line_only = false;
        } else if (arg == "--before-after") {
            before_after = true;
        } else if (arg == "--json" && i + 1 < argc) {
            json_output_file = argv[++i];
        } else if (arg == "--synth" && i + 1 < argc) {
            synth_type = argv[++i];
        } else if (arg == "--rows" && i + 1 < argc) {
            synth_rows = std::stoll(argv[++i]);
        } else if (arg == "--cols" && i + 1 < argc) {
            synth_cols = std::stoll(argv[++i]);
        } else if (arg == "--density" && i + 1 < argc) {
            synth_density = std::stod(argv[++i]);
        } else if (arg[0] != '-') {
            mps_file = arg;
        }
    }

    LinearProgram lp;
    if (!mps_file.empty()) {
        auto status = MPSParser::parse_file(mps_file, lp);
        if (!status.is_ok()) {
            std::cerr << "Error parsing MPS file: " << status.to_string() << "\n";
            return 1;
        }
    } else if (!synth_type.empty()) {
        lp = make_synthetic_lp(synth_type, synth_rows, synth_cols, synth_density);
    } else {
        std::cerr << "Error: Neither an MPS file nor --synth was specified.\n";
        print_help(argv[0]);
        return 1;
    }

    ProblemStats raw_stats = ProblemAnalyzer::analyze(lp);

    if (one_line_only && !before_after) {
        std::cout << raw_stats.format_one_line_summary() << "\n";
        return 0;
    }

    std::cout << raw_stats.format_report() << "\n";

    PreparedLP prep;
    if (before_after) {
        PipelineConfig cfg;
        cfg.enable_presolve = true;
        cfg.enable_scaling = true;
        cfg.compute_characterization = true;

        auto prep_res = ModelPipeline::prepare(lp, cfg);
        if (!prep_res.is_ok()) {
            std::cerr << "Preparation pipeline error: " << prep_res.status().to_string() << "\n";
            return 1;
        }
        prep = prep_res.value();

        std::cout << "\n================================================================================\n";
        std::cout << "                 BEFORE / AFTER PIPELINE COMPARISON REPORT                      \n";
        std::cout << "================================================================================\n";
        std::cout << std::left << std::setw(28) << "Metric"
                  << std::setw(25) << "Before (Raw)"
                  << std::setw(25) << "After (Presolve + Ruiz)" << "\n";
        std::cout << "--------------------------------------------------------------------------------\n";

        auto print_row_str = [](const std::string& name, const std::string& b, const std::string& a) {
            std::cout << std::left << std::setw(28) << name
                      << std::setw(25) << b
                      << std::setw(25) << a << "\n";
        };
        auto print_row_idx = [](const std::string& name, index_t b, index_t a) {
            std::ostringstream sb, sa;
            sb << b;
            sa << a << " (" << (b > a ? "-" : "+") << std::abs(b - a) << ")";
            std::cout << std::left << std::setw(28) << name
                      << std::setw(25) << sb.str()
                      << std::setw(25) << sa.str() << "\n";
        };
        auto print_row_sci = [](const std::string& name, double b, double a) {
            std::ostringstream sb, sa;
            sb << std::scientific << std::setprecision(2) << b;
            sa << std::scientific << std::setprecision(2) << a;
            std::cout << std::left << std::setw(28) << name
                      << std::setw(25) << sb.str()
                      << std::setw(25) << sa.str() << "\n";
        };

        print_row_idx("Variables (Cols)", raw_stats.num_cols, prep.problem_stats.num_cols);
        print_row_idx("Constraints (Rows)", raw_stats.num_rows, prep.problem_stats.num_rows);
        print_row_idx("Nonzeros (NNZ)", static_cast<index_t>(raw_stats.num_nonzeros),
                      static_cast<index_t>(prep.problem_stats.num_nonzeros));

        std::ostringstream db, da;
        db << std::fixed << std::setprecision(4) << (raw_stats.density * 100.0) << "%";
        da << std::fixed << std::setprecision(4) << (prep.problem_stats.density * 100.0) << "%";
        print_row_str("Sparsity Density", db.str(), da.str());

        print_row_sci("Min Abs Coefficient", raw_stats.min_abs_coeff, prep.problem_stats.min_abs_coeff);
        print_row_sci("Max Abs Coefficient", raw_stats.max_abs_coeff, prep.problem_stats.max_abs_coeff);
        print_row_sci("Dynamic Range Proxy", raw_stats.dynamic_range, prep.problem_stats.dynamic_range);
        print_row_sci("Row Norm Ratio Proxy", raw_stats.conditioning_proxy.row_norm_ratio_proxy,
                      prep.problem_stats.conditioning_proxy.row_norm_ratio_proxy);
        print_row_sci("Col Norm Ratio Proxy", raw_stats.conditioning_proxy.col_norm_ratio_proxy,
                      prep.problem_stats.conditioning_proxy.col_norm_ratio_proxy);
        print_row_sci("Spectral Norm Estimate", raw_stats.conditioning_proxy.spectral_norm_estimate,
                      prep.problem_stats.conditioning_proxy.spectral_norm_estimate);
        print_row_str("Row Imbalance Rating", raw_stats.row_imbalance_rating(),
                      prep.problem_stats.row_imbalance_rating());
        print_row_str("Recommended Engine", to_string(raw_stats.recommended_engine),
                      to_string(prep.problem_stats.recommended_engine));
        print_row_str("Estimated VRAM", raw_stats.format_memory(raw_stats.estimated_gpu_vram_mb),
                      prep.problem_stats.format_memory(prep.problem_stats.estimated_gpu_vram_mb));

        std::cout << "--------------------------------------------------------------------------------\n";
        std::cout << "SUMMARY AFTER PIPELINE:\n";
        std::cout << "  " << prep.problem_stats.format_one_line_summary() << "\n";
        std::cout << "================================================================================\n";
    }

    if (!json_output_file.empty()) {
        std::ofstream ofs(json_output_file);
        if (ofs.is_open()) {
            if (before_after) {
                ofs << "{\n"
                    << "  \"raw\": " << stats_to_json(lp.name, raw_stats) << ",\n"
                    << "  \"prepared\": " << stats_to_json(lp.name + "_prep", prep.problem_stats) << "\n"
                    << "}\n";
            } else {
                ofs << stats_to_json(lp.name, raw_stats) << "\n";
            }
            std::cout << "Wrote JSON report to: " << json_output_file << "\n";
        }
    }

    return 0;
}
