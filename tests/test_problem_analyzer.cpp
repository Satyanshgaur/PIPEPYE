#include <gtest/gtest.h>
#include <pipepye/analysis/problem_analyzer.hpp>
#include <pipepye/sparse/matrix_generator.hpp>
#include <pipepye/model/mps_parser.hpp>
#include <cmath>
#include <filesystem>

using namespace pipepye;
using namespace pipepye::analysis;
using namespace pipepye::sparse;
using namespace pipepye::model;

namespace {

LinearProgram lp_from_coo(const COOMatrix& coo) {
    LinearProgram lp;
    lp.name = "synth_lp";
    index_t m = coo.num_rows();
    index_t n = coo.num_cols();

    lp.c.assign(n, 1.0);
    lp.col_lower.assign(n, 0.0);
    lp.col_upper.assign(n, 10.0);
    lp.row_lower.assign(m, -1e20);
    lp.row_upper.assign(m, 10.0);

    for (index_t j = 0; j < n; ++j) {
        lp.col_names.push_back("x" + std::to_string(j));
        lp.col_name_to_idx["x" + std::to_string(j)] = j;
        lp.var_types.push_back(VariableType::Continuous);
    }
    for (index_t i = 0; i < m; ++i) {
        lp.row_names.push_back("c" + std::to_string(i));
        lp.row_name_to_idx["c" + std::to_string(i)] = i;
        lp.row_senses.push_back(RowSense::Ranged);
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

std::string find_netlib_file(const std::string& filename) {
    std::vector<std::string> search_dirs = {
        "tests/data/mps/netlib",
        "../tests/data/mps/netlib",
        "../../tests/data/mps/netlib",
        "/home/satyansh/pipepye/tests/data/mps/netlib"
    };
    for (const auto& dir : search_dirs) {
        std::filesystem::path p = std::filesystem::path(dir) / filename;
        if (std::filesystem::exists(p)) return p.string();
    }
    return "";
}

} // anonymous namespace

// =============================================================================
// Test 1: Uniform Random Matrix Characterization
// =============================================================================
TEST(ProblemAnalyzerTest, UniformRandomMatrixProperties) {
    auto coo = MatrixGenerator::generate_random(100, 100, 0.05, 1.0, 5.0, 42);
    auto lp = lp_from_coo(coo);

    auto stats = ProblemAnalyzer::analyze(lp);
    EXPECT_EQ(stats.num_rows, 100);
    EXPECT_EQ(stats.num_cols, 100);
    EXPECT_GT(stats.num_nonzeros, 400);
    EXPECT_NEAR(stats.density, 0.05, 0.015);

    // Uniform random should have low Gini coefficient
    EXPECT_LT(stats.row_length_gini, 0.30);
    EXPECT_LT(stats.row_length_imbalance, 3.0);

    // Formatter check
    std::string rep = stats.format_report();
    EXPECT_NE(rep.find("PIPEPYE PROBLEM CHARACTERIZATION REPORT"), std::string::npos);
    EXPECT_NE(rep.find("Row Gini Coefficient:"), std::string::npos);
}

// =============================================================================
// Test 2: Banded Matrix Bandwidth Detection
// =============================================================================
TEST(ProblemAnalyzerTest, BandedMatrixBandwidth) {
    index_t k = 5; // Half-bandwidth
    auto coo = MatrixGenerator::generate_banded(100, 100, k, k, 1.0, 5.0, 42);
    auto lp = lp_from_coo(coo);

    auto stats = ProblemAnalyzer::analyze(lp);
    EXPECT_EQ(stats.num_rows, 100);
    EXPECT_EQ(stats.num_cols, 100);
    EXPECT_LE(stats.half_bandwidth, k);
    EXPECT_LE(stats.normalized_bandwidth, 0.06);
}

// =============================================================================
// Test 3: Block Diagonal Matrix Connected Components
// =============================================================================
TEST(ProblemAnalyzerTest, BlockDiagonalConnectedComponents) {
    int num_blocks = 4;
    // 4 blocks of 25x25 with zero coupling density
    auto coo = MatrixGenerator::generate_block_diagonal(num_blocks, 25, 25, 0.2, 0.0, 1.0, 5.0, 42);
    auto lp = lp_from_coo(coo);

    auto stats = ProblemAnalyzer::analyze(lp);
    EXPECT_EQ(stats.num_rows, 100);
    EXPECT_EQ(stats.num_cols, 100);
    // Bipartite graph should have at least 4 connected components
    EXPECT_GE(stats.num_connected_components, num_blocks);
}

// =============================================================================
// Test 4: Staircase Matrix Progression Correlation
// =============================================================================
TEST(ProblemAnalyzerTest, StaircaseMatrixProgressionScore) {
    // 10 stages of 10x10 blocks
    auto coo = MatrixGenerator::generate_staircase(10, 10, 10, 0.2, 1.0, 5.0, 42);
    auto lp = lp_from_coo(coo);

    auto stats = ProblemAnalyzer::analyze(lp);
    // Row median column indices should have strong positive correlation with row indices
    EXPECT_GT(stats.staircase_score, 0.85);
}

// =============================================================================
// Test 5: Irregular Power-Law Hub Matrix Characterization
// =============================================================================
TEST(ProblemAnalyzerTest, IrregularHubMatrixImbalanceAndEngineRecommendation) {
    // 2000 rows, 2000 cols, 40000 NNZ with 5% hub rows holding 50% of nonzeros (> 30000 NNZ triggers GPU)
    auto coo = MatrixGenerator::generate_irregular(2000, 2000, 40000, 0.05, 0.50, 1.0, 5.0, 42);
    auto lp = lp_from_coo(coo);

    auto stats = ProblemAnalyzer::analyze(lp);
    EXPECT_GT(stats.row_length_imbalance, 4.0);
    EXPECT_GE(stats.row_length_gini, 0.35);

    // Large NNZ with high Gini must recommend GPU_MergePath based on Phase 1 findings
    EXPECT_EQ(stats.recommended_engine, RecommendedEngine::GPU_MergePath);
    EXPECT_NE(stats.recommendation_reason.find("Merge-Path"), std::string::npos);
}

// =============================================================================
// Test 6: Netlib AFIRO MPS Structural Analysis
// =============================================================================
TEST(ProblemAnalyzerTest, NetlibAFIROStructuralAnalysis) {
    std::string path = find_netlib_file("afiro.mps");
    if (path.empty()) GTEST_SKIP() << "Netlib afiro.mps not found";

    LinearProgram lp;
    ASSERT_TRUE(MPSParser::parse_file(path, lp).is_ok());

    auto stats = ProblemAnalyzer::analyze(lp);
    EXPECT_EQ(stats.num_rows, 27);
    EXPECT_EQ(stats.num_cols, 32);
    EXPECT_EQ(stats.num_nonzeros, 83);
    EXPECT_GT(stats.num_equality_rows, 0);
    EXPECT_EQ(stats.num_bounded_below_vars, 32);

    // AFIRO NNZ = 83 < 15,000 => must recommend CPU_SingleThread
    EXPECT_EQ(stats.recommended_engine, RecommendedEngine::CPU_SingleThread);
    EXPECT_NE(stats.recommendation_reason.find("CPU"), std::string::npos);
}
