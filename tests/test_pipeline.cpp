#include <gtest/gtest.h>
#include <pipepye/pipeline/model_pipeline.hpp>
#include <pipepye/model/mps_parser.hpp>
#include <pipepye/sparse/matrix_generator.hpp>
#include <filesystem>
#include <cmath>

using namespace pipepye;
using namespace pipepye::pipeline;
using namespace pipepye::model;
using namespace pipepye::presolve;
using namespace pipepye::scaling;
using namespace pipepye::analysis;
using namespace pipepye::sparse;

namespace {

std::string find_netlib_file(const std::string& filename) {
    std::vector<std::string> search_dirs = {
        "tests/data/mps/netlib",
        "../tests/data/mps/netlib",
        "../../tests/data/mps/netlib",
        "/home/satyansh/pipepye/tests/data/mps/netlib"
    };
    for (const auto& dir : search_dirs) {
        std::filesystem::path p = std::filesystem::path(dir) / filename;
        if (std::filesystem::exists(p)) {
            return p.string();
        }
    }
    return "";
}

LinearProgram make_small_solvable_lp() {
    // min 2 x0 + 3 x1
    // s.t.
    // row0: x0 + x1 >= 4
    // row1: 2 x0 + x1 == 6 (forced: x0 + 4 = 6 => x0 = 2, x1 = 2)
    // row2: x2 = 5 (fixed variable)
    // bounds: x0 >= 0, x1 >= 0, 5 <= x2 <= 5
    LinearProgram lp;
    lp.name = "small_solvable";
    lp.c = {2.0, 3.0, 10.0};
    lp.col_lower = {0.0, 0.0, 5.0};
    lp.col_upper = {100.0, 100.0, 5.0};
    lp.col_names = {"x0", "x1", "x2"};
    lp.col_name_to_idx = {{"x0", 0}, {"x1", 1}, {"x2", 2}};
    lp.var_types.assign(3, VariableType::Continuous);

    lp.row_lower = {4.0, 6.0};
    lp.row_upper = {1e20, 6.0};
    lp.row_names = {"r0", "r1"};
    lp.row_name_to_idx = {{"r0", 0}, {"r1", 1}};
    lp.row_senses.push_back(RowSense::GreaterEqual);
    lp.row_senses.push_back(RowSense::Equality);

    COOMatrix coo(2, 3);
    coo.add_entry(0, 0, 1.0);
    coo.add_entry(0, 1, 1.0);
    coo.add_entry(1, 0, 2.0);
    coo.add_entry(1, 1, 1.0);
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

} // namespace

TEST(ConditioningProxyTest, MagnitudeAndSpectralProxies) {
    // Generate an ill-conditioned synthetic matrix with known dynamic range
    COOMatrix coo(50, 50);
    for (index_t i = 0; i < 50; ++i) {
        coo.add_entry(i, i, (i % 2 == 0) ? 1e-5 : 1e5);
        if (i + 1 < 50) {
            coo.add_entry(i, i + 1, 1.0);
        }
    }
    LinearProgram lp;
    lp.name = "ill_conditioned";
    lp.c.assign(50, 1.0);
    lp.col_lower.assign(50, 0.0);
    lp.col_upper.assign(50, 10.0);
    lp.row_lower.assign(50, 0.0);
    lp.row_upper.assign(50, 10.0);
    lp.row_senses.assign(50, RowSense::LessEqual);
    lp.var_types.assign(50, VariableType::Continuous);
    for (index_t j = 0; j < 50; ++j) {
        lp.col_names.push_back("x" + std::to_string(j));
    }
    for (index_t i = 0; i < 50; ++i) {
        lp.row_names.push_back("r" + std::to_string(i));
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

    ProblemStats stats = ProblemAnalyzer::analyze(lp);

    EXPECT_GT(stats.dynamic_range, 1e8);
    EXPECT_GE(stats.dynamic_range_orders, 8.0);
    EXPECT_GT(stats.conditioning_proxy.magnitude_range_proxy, 1e8);
    EXPECT_GT(stats.conditioning_proxy.spectral_norm_estimate, 0.0);
    EXPECT_GT(stats.conditioning_proxy.spectral_conditioning_proxy, 1.0);

    std::string summary = stats.format_one_line_summary();
    EXPECT_NE(summary.find("variables"), std::string::npos);
    EXPECT_NE(summary.find("constraints"), std::string::npos);
    EXPECT_NE(summary.find("density"), std::string::npos);
    EXPECT_NE(summary.find("coefficient range"), std::string::npos);
    EXPECT_NE(summary.find("row imbalance"), std::string::npos);
    EXPECT_NE(summary.find("estimated VRAM"), std::string::npos);
}

TEST(PipelineTypesTest, StringParsingAndRoundTrip) {
    auto r_raw = parse_pipeline_mode("RAW");
    ASSERT_TRUE(r_raw.is_ok());
    EXPECT_EQ(r_raw.value(), PipelineMode::RAW);
    EXPECT_EQ(to_string(PipelineMode::RAW), "RAW");

    auto r_raw_lower = parse_pipeline_mode("raw");
    ASSERT_TRUE(r_raw_lower.is_ok());
    EXPECT_EQ(r_raw_lower.value(), PipelineMode::RAW);

    auto r_p = parse_pipeline_mode("PRESOLVE_ONLY");
    ASSERT_TRUE(r_p.is_ok());
    EXPECT_EQ(r_p.value(), PipelineMode::PRESOLVE_ONLY);
    EXPECT_EQ(to_string(PipelineMode::PRESOLVE_ONLY), "PRESOLVE_ONLY");

    auto r_p_alias = parse_pipeline_mode("presolve");
    ASSERT_TRUE(r_p_alias.is_ok());
    EXPECT_EQ(r_p_alias.value(), PipelineMode::PRESOLVE_ONLY);

    auto r_s = parse_pipeline_mode("SCALING_ONLY");
    ASSERT_TRUE(r_s.is_ok());
    EXPECT_EQ(r_s.value(), PipelineMode::SCALING_ONLY);
    EXPECT_EQ(to_string(PipelineMode::SCALING_ONLY), "SCALING_ONLY");

    auto r_s_alias = parse_pipeline_mode("scaling");
    ASSERT_TRUE(r_s_alias.is_ok());
    EXPECT_EQ(r_s_alias.value(), PipelineMode::SCALING_ONLY);

    auto r_full = parse_pipeline_mode("PRESOLVE_AND_SCALING");
    ASSERT_TRUE(r_full.is_ok());
    EXPECT_EQ(r_full.value(), PipelineMode::PRESOLVE_AND_SCALING);
    EXPECT_EQ(to_string(PipelineMode::PRESOLVE_AND_SCALING), "PRESOLVE_AND_SCALING");

    auto r_full_alias = parse_pipeline_mode("full");
    ASSERT_TRUE(r_full_alias.is_ok());
    EXPECT_EQ(r_full_alias.value(), PipelineMode::PRESOLVE_AND_SCALING);

    auto r_bad = parse_pipeline_mode("UNKNOWN_MODE_XYZ");
    EXPECT_FALSE(r_bad.is_ok());
}

TEST(ModelPipelineTest, AblationModeRaw) {
    LinearProgram orig_lp = make_small_solvable_lp();

    PipelineConfig config = PipelineConfig::Raw();
    EXPECT_FALSE(config.presolve_enabled());
    EXPECT_FALSE(config.scaling_enabled());
    EXPECT_EQ(config.mode, PipelineMode::RAW);

    auto prep_res = ModelPipeline::prepare(orig_lp, config);
    ASSERT_TRUE(prep_res.is_ok());

    const auto& prepared = prep_res.value();
    EXPECT_EQ(prepared.mode_applied, PipelineMode::RAW);
    EXPECT_FALSE(prepared.recovery_map.was_presolved());
    EXPECT_FALSE(prepared.recovery_map.was_scaled());

    // Dimensions must be untouched
    EXPECT_EQ(prepared.lp.num_cols(), orig_lp.num_cols());
    EXPECT_EQ(prepared.lp.num_rows(), orig_lp.num_rows());
    EXPECT_EQ(prepared.lp.num_nonzeros(), orig_lp.num_nonzeros());

    // Solution recovery is pure identity pass-through
    PrimalDualSolution solver_sol;
    solver_sol.x = {2.0, 2.0, 5.0};
    solver_sol.y = {0.0, 0.0};
    solver_sol.s = {0.0, 0.0, 0.0};
    solver_sol.is_feasible = true;

    auto rec_res = prepared.recover_solution(solver_sol, orig_lp);
    ASSERT_TRUE(rec_res.is_ok());
    const auto& rec = rec_res.value();
    ASSERT_EQ(rec.x.size(), 3);
    EXPECT_DOUBLE_EQ(rec.x[0], 2.0);
    EXPECT_DOUBLE_EQ(rec.x[1], 2.0);
    EXPECT_DOUBLE_EQ(rec.x[2], 5.0);
    EXPECT_DOUBLE_EQ(rec.objective_value, 60.0);
}

TEST(ModelPipelineTest, AblationModePresolveOnly) {
    LinearProgram orig_lp = make_small_solvable_lp();

    PipelineConfig config = PipelineConfig::PresolveOnly();
    EXPECT_TRUE(config.presolve_enabled());
    EXPECT_FALSE(config.scaling_enabled());
    EXPECT_EQ(config.mode, PipelineMode::PRESOLVE_ONLY);

    auto prep_res = ModelPipeline::prepare(orig_lp, config);
    ASSERT_TRUE(prep_res.is_ok());

    const auto& prepared = prep_res.value();
    EXPECT_EQ(prepared.mode_applied, PipelineMode::PRESOLVE_ONLY);
    EXPECT_TRUE(prepared.recovery_map.was_presolved());
    EXPECT_FALSE(prepared.recovery_map.was_scaled());

    // Fixed variable x2 eliminated
    EXPECT_EQ(prepared.lp.num_cols(), 2);
    EXPECT_EQ(prepared.presolve_stats.total_variables_fixed(), 1);

    // Unscaled solver coordinates
    PrimalDualSolution solver_sol;
    solver_sol.x = {2.0, 2.0};
    solver_sol.y = {0.0, 0.0};
    solver_sol.s = {0.0, 0.0};
    solver_sol.is_feasible = true;

    auto rec_res = prepared.recover_solution(solver_sol, orig_lp);
    ASSERT_TRUE(rec_res.is_ok());
    const auto& rec = rec_res.value();
    ASSERT_EQ(rec.x.size(), 3);
    EXPECT_NEAR(rec.x[0], 2.0, 1e-6);
    EXPECT_NEAR(rec.x[1], 2.0, 1e-6);
    EXPECT_NEAR(rec.x[2], 5.0, 1e-6); // Restored fixed variable
    EXPECT_NEAR(rec.objective_value, 60.0, 1e-6);
}

TEST(ModelPipelineTest, AblationModeScalingOnly) {
    LinearProgram orig_lp = make_small_solvable_lp();

    PipelineConfig config = PipelineConfig::ScalingOnly();
    EXPECT_FALSE(config.presolve_enabled());
    EXPECT_TRUE(config.scaling_enabled());
    EXPECT_EQ(config.mode, PipelineMode::SCALING_ONLY);

    auto prep_res = ModelPipeline::prepare(orig_lp, config);
    ASSERT_TRUE(prep_res.is_ok());

    const auto& prepared = prep_res.value();
    EXPECT_EQ(prepared.mode_applied, PipelineMode::SCALING_ONLY);
    EXPECT_FALSE(prepared.recovery_map.was_presolved());
    EXPECT_TRUE(prepared.recovery_map.was_scaled());

    // Dimensions unchanged because presolve was skipped
    EXPECT_EQ(prepared.lp.num_cols(), 3);

    // Scaling applied
    ASSERT_EQ(prepared.scaling.col_scale_C.size(), 3);
    ASSERT_EQ(prepared.scaling.row_scale_R.size(), 2);

    // Scaled solver solution: x_prep = x / C[j]
    PrimalDualSolution solver_sol;
    solver_sol.x = {2.0 / prepared.scaling.col_scale_C[0],
                    2.0 / prepared.scaling.col_scale_C[1],
                    5.0 / prepared.scaling.col_scale_C[2]};
    solver_sol.y = {0.0, 0.0};
    solver_sol.s = {0.0, 0.0, 0.0};
    solver_sol.is_feasible = true;

    auto rec_res = prepared.recover_solution(solver_sol, orig_lp);
    ASSERT_TRUE(rec_res.is_ok());
    const auto& rec = rec_res.value();
    ASSERT_EQ(rec.x.size(), 3);
    EXPECT_NEAR(rec.x[0], 2.0, 1e-5);
    EXPECT_NEAR(rec.x[1], 2.0, 1e-5);
    EXPECT_NEAR(rec.x[2], 5.0, 1e-5);
    EXPECT_NEAR(rec.objective_value, 60.0, 1e-5);
}

TEST(ModelPipelineTest, FullPreparationAndSolutionRecovery) {
    LinearProgram lp = make_small_solvable_lp();

    PipelineConfig config = PipelineConfig::PresolveAndScaling();
    config.compute_characterization = true;

    auto prep_res = ModelPipeline::prepare(lp, config);
    ASSERT_TRUE(prep_res.is_ok());

    const auto& prepared = prep_res.value();
    EXPECT_EQ(prepared.mode_applied, PipelineMode::PRESOLVE_AND_SCALING);
    // Fixed variable x2 should be eliminated by presolve
    EXPECT_EQ(prepared.lp.num_cols(), 2);
    EXPECT_EQ(prepared.presolve_stats.total_variables_fixed(), 1);
    EXPECT_TRUE(prepared.recovery_map.was_presolved());
    EXPECT_TRUE(prepared.recovery_map.was_scaled());

    // Prepared LP should have problem stats computed
    EXPECT_EQ(prepared.problem_stats.num_cols, 2);
    EXPECT_GT(prepared.problem_stats.density, 0.0);

    // Summary formatting test
    std::string summary = prepared.format_pipeline_summary();
    EXPECT_NE(summary.find("PRESOLVE_AND_SCALING"), std::string::npos);
    EXPECT_NE(summary.find("REDUCED"), std::string::npos);

    // Simulate downstream solver solution on the prepared LP
    PrimalDualSolution solver_sol;
    solver_sol.x.resize(prepared.lp.num_cols());
    for (index_t j = 0; j < prepared.lp.num_cols(); ++j) {
        scalar_t unscaled_val = 2.0;
        scalar_t c_scale = prepared.scaling.col_scale_C.empty() ? 1.0 : prepared.scaling.col_scale_C[j];
        solver_sol.x[j] = unscaled_val / c_scale;
    }
    solver_sol.y.assign(prepared.lp.num_rows(), 0.0);
    solver_sol.s.assign(prepared.lp.num_cols(), 0.0);
    solver_sol.is_feasible = true;

    // Single-call recovery back to original formulation
    auto rec_res = prepared.recover_solution(solver_sol, lp);
    ASSERT_TRUE(rec_res.is_ok());

    const auto& rec = rec_res.value();
    // Reconstructed solution must have original dimension 3
    ASSERT_EQ(rec.x.size(), 3);
    EXPECT_NEAR(rec.x[0], 2.0, 1e-4);
    EXPECT_NEAR(rec.x[1], 2.0, 1e-4);
    EXPECT_NEAR(rec.x[2], 5.0, 1e-4); // Fixed variable restored to 5.0!

    // Objective: 2*2 + 3*2 + 10*5 = 4 + 6 + 50 = 60
    EXPECT_NEAR(rec.objective_value, 60.0, 1e-4);
}

TEST(ModelPipelineTest, ReproducibilityBitIdenticalOutputs) {
    LinearProgram lp = make_small_solvable_lp();

    PipelineConfig config = PipelineConfig::PresolveAndScaling();
    config.random_seed = 12345;

    auto res1 = ModelPipeline::prepare(lp, config);
    auto res2 = ModelPipeline::prepare(lp, config);

    ASSERT_TRUE(res1.is_ok());
    ASSERT_TRUE(res2.is_ok());

    const auto& p1 = res1.value();
    const auto& p2 = res2.value();

    // Check bit-identical CSR matrices
    EXPECT_EQ(p1.lp.csr_row_ptr, p2.lp.csr_row_ptr);
    EXPECT_EQ(p1.lp.csr_col_ind, p2.lp.csr_col_ind);
    EXPECT_EQ(p1.lp.csr_values, p2.lp.csr_values);

    // Check bit-identical vectors
    EXPECT_EQ(p1.lp.c, p2.lp.c);
    EXPECT_EQ(p1.lp.col_lower, p2.lp.col_lower);
    EXPECT_EQ(p1.lp.col_upper, p2.lp.col_upper);
    EXPECT_EQ(p1.lp.row_lower, p2.lp.row_lower);
    EXPECT_EQ(p1.lp.row_upper, p2.lp.row_upper);
    EXPECT_EQ(p1.lp.obj_offset, p2.lp.obj_offset);

    // Check bit-identical scaling factors
    EXPECT_EQ(p1.scaling.col_scale_C, p2.scaling.col_scale_C);
    EXPECT_EQ(p1.scaling.row_scale_R, p2.scaling.row_scale_R);

    // Check presolve statistics
    EXPECT_EQ(p1.presolve_stats.eliminated_rows(), p2.presolve_stats.eliminated_rows());
    EXPECT_EQ(p1.presolve_stats.eliminated_cols(), p2.presolve_stats.eliminated_cols());
    EXPECT_EQ(p1.presolve_stats.eliminated_nonzeros(), p2.presolve_stats.eliminated_nonzeros());
}

TEST(ModelPipelineTest, NetlibAfiroPipeline) {
    std::string filepath = find_netlib_file("afiro.mps");
    if (filepath.empty()) {
        GTEST_SKIP() << "afiro.mps not found";
    }

    LinearProgram orig_lp;
    auto parse_res = MPSParser::parse_file(filepath, orig_lp);
    ASSERT_TRUE(parse_res.is_ok());

    PipelineConfig config = PipelineConfig::PresolveAndScaling();
    config.compute_characterization = true;

    auto prep_res = ModelPipeline::prepare(orig_lp, config);
    ASSERT_TRUE(prep_res.is_ok());

    const auto& prep = prep_res.value();
    EXPECT_LE(prep.lp.num_cols(), orig_lp.num_cols());
    EXPECT_LE(prep.lp.num_rows(), orig_lp.num_rows());
    EXPECT_GT(prep.problem_stats.num_nonzeros, 0);
    EXPECT_FALSE(prep.problem_stats.format_one_line_summary().empty());

    // Scaling diagnostics should show well-controlled ranges
    EXPECT_LT(prep.scaling.diag_after.dynamic_range, prep.scaling.diag_before.dynamic_range + 1e-5);
}

TEST(ModelPipelineTest, InfeasibleModelDetection) {
    LinearProgram lp;
    lp.name = "contradictory_bounds";
    lp.c = {1.0};
    lp.col_lower = {10.0};
    lp.col_upper = {5.0}; // lb > ub => Infeasible
    lp.col_names = {"x0"};
    lp.col_name_to_idx = {{"x0", 0}};
    lp.var_types = {VariableType::Continuous};

    PipelineConfig config = PipelineConfig::PresolveAndScaling();

    auto prep_res = ModelPipeline::prepare(lp, config);
    ASSERT_TRUE(prep_res.is_ok());
    EXPECT_TRUE(prep_res.value().is_infeasible);
}
