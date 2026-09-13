#include <gtest/gtest.h>
#include <pipepye/scaling/equilibrator.hpp>
#include <pipepye/model/mps_parser.hpp>
#include <cmath>
#include <filesystem>

using namespace pipepye;
using namespace pipepye::model;
using namespace pipepye::scaling;

namespace {

LinearProgram create_ill_conditioned_lp() {
    // 3x3 system with widely varying coefficient magnitudes (10^-4 to 10^4)
    LinearProgram lp;
    lp.name = "ill_conditioned";
    lp.c = {1e-3, 2.0, 1e4};
    lp.col_lower = {0.0, 0.0, 0.0};
    lp.col_upper = {10.0, 10.0, 10.0};
    lp.row_lower = {-1e20, -1e20, -1e20};
    lp.row_upper = {1e-2, 5.0, 1e6};

    for (index_t j = 0; j < 3; ++j) {
        lp.col_names.push_back("x" + std::to_string(j));
        lp.col_name_to_idx["x" + std::to_string(j)] = j;
        lp.var_types.push_back(VariableType::Continuous);
    }
    for (index_t i = 0; i < 3; ++i) {
        lp.row_names.push_back("c" + std::to_string(i));
        lp.row_name_to_idx["c" + std::to_string(i)] = i;
        lp.row_senses.push_back(RowSense::Ranged);
    }

    sparse::COOMatrix coo(3, 3);
    coo.add_entry(0, 0, 1e-4);
    coo.add_entry(0, 1, 2e-3);
    coo.add_entry(1, 0, 1.0);
    coo.add_entry(1, 1, 5.0);
    coo.add_entry(1, 2, 2.0);
    coo.add_entry(2, 1, 1e3);
    coo.add_entry(2, 2, 1e4);
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
// Test 1: Ruiz Equilibration Convergence & Dynamic Range Reduction
// =============================================================================
TEST(ScalingTest, RuizEquilibrationBalancesRowAndColNorms) {
    auto lp = create_ill_conditioned_lp();

    Equilibrator eq;
    auto res = eq.scale(lp);
    ASSERT_TRUE(res.is_ok());

    auto scaled = res.value();
    EXPECT_GT(scaled.iterations_performed, 0);

    // Dynamic range should shrink drastically
    EXPECT_GT(scaled.diag_before.dynamic_range, 1e7);
    EXPECT_LT(scaled.diag_after.dynamic_range, 100.0);

    // Row infinity norms should be close to 1.0
    EXPECT_NEAR(scaled.diag_after.row_linf_stats.max_norm, 1.0, 0.05);
    EXPECT_NEAR(scaled.diag_after.row_linf_stats.min_norm, 1.0, 0.05);

    // Column infinity norms should be close to 1.0
    EXPECT_NEAR(scaled.diag_after.col_linf_stats.max_norm, 1.0, 0.05);
    EXPECT_NEAR(scaled.diag_after.col_linf_stats.min_norm, 1.0, 0.05);
}

// =============================================================================
// Test 2: Scaling Diagnostics Telemetry
// =============================================================================
TEST(ScalingTest, DiagnosticsBeforeAndAfterReporting) {
    auto lp = create_ill_conditioned_lp();

    auto diag = Equilibrator::compute_diagnostics(lp);
    EXPECT_NEAR(diag.min_abs_coeff, 1e-4, 1e-8);
    EXPECT_NEAR(diag.max_abs_coeff, 1e4, 1e-2);
    EXPECT_GT(diag.dynamic_range_orders, 7.0);
    EXPECT_GT(diag.row_conditioning_proxy, 100.0);

    std::string report = diag.format_report("ORIGINAL MODEL DIAGNOSTICS");
    EXPECT_NE(report.find("Coeff Min / Max:"), std::string::npos);
    EXPECT_NE(report.find("Dynamic Range:"), std::string::npos);
    EXPECT_NE(report.find("Conditioning Proxies:"), std::string::npos);
}

// =============================================================================
// Test 3: Model Reversibility (Unscale Model Returns Original)
// =============================================================================
TEST(ScalingTest, ModelUnscalingReversibility) {
    auto lp = create_ill_conditioned_lp();

    Equilibrator eq;
    auto res = eq.scale(lp);
    ASSERT_TRUE(res.is_ok());
    auto scaled = res.value();

    auto unscaled = scaled.unscale_model();

    // Verify matrix values match within high precision
    ASSERT_EQ(unscaled.csr_values.size(), lp.csr_values.size());
    for (size_t k = 0; k < lp.csr_values.size(); ++k) {
        EXPECT_NEAR(unscaled.csr_values[k], lp.csr_values[k], 1e-9);
    }

    // Verify objective
    for (index_t j = 0; j < lp.num_cols(); ++j) {
        EXPECT_NEAR(unscaled.c[j], lp.c[j], 1e-9);
    }

    // Verify bounds
    for (index_t i = 0; i < lp.num_rows(); ++i) {
        if (lp.row_upper[i] < 1e19) {
            EXPECT_NEAR(unscaled.row_upper[i], lp.row_upper[i], 1e-9);
        }
    }
}

// =============================================================================
// Test 4: Solution Scaling and Unscaling Reversibility
// =============================================================================
TEST(ScalingTest, SolutionUnscalingAndObjectiveParity) {
    auto lp = create_ill_conditioned_lp();

    Equilibrator eq;
    auto res = eq.scale(lp);
    ASSERT_TRUE(res.is_ok());
    auto scaled = res.value();

    // Fabricate a scaled solution
    presolve::PrimalDualSolution scaled_sol;
    scaled_sol.x = {0.5, 0.8, 0.2};
    scaled_sol.y = {0.1, 0.05, 0.01};
    scaled_sol.s = {0.2, 0.4, 0.1};
    scaled_sol.is_feasible = true;

    // Unscale solution
    auto unscaled_sol = scaled.unscale_solution(scaled_sol);
    EXPECT_TRUE(unscaled_sol.is_feasible);

    // Round-trip: scale back
    auto rescale = scaled.scale_solution(unscaled_sol);
    for (size_t j = 0; j < 3; ++j) {
        EXPECT_NEAR(rescale.x[j], scaled_sol.x[j], 1e-10);
        EXPECT_NEAR(rescale.y[j], scaled_sol.y[j], 1e-10);
        EXPECT_NEAR(rescale.s[j], scaled_sol.s[j], 1e-10);
    }

    // Check objective parity: c^T x == c'^T x'
    scalar_t unscaled_obj = 0.0;
    for (index_t j = 0; j < 3; ++j) unscaled_obj += lp.c[j] * unscaled_sol.x[j];

    scalar_t scaled_obj = 0.0;
    for (index_t j = 0; j < 3; ++j) scaled_obj += scaled.lp.c[j] * scaled_sol.x[j];

    EXPECT_NEAR(unscaled_obj, scaled_obj, 1e-8);
}

// =============================================================================
// Test 5: Pock-Chambolle Scaling Strategy
// =============================================================================
TEST(ScalingTest, PockChambollePreconditioning) {
    auto lp = create_ill_conditioned_lp();

    ScalingOptions opt;
    opt.method = ScalingMethod::PockChambolle;
    opt.alpha = 1.0;

    Equilibrator eq(opt);
    auto res = eq.scale(lp);
    ASSERT_TRUE(res.is_ok());
    auto scaled = res.value();

    EXPECT_EQ(scaled.iterations_performed, 1);
    EXPECT_LT(scaled.diag_after.dynamic_range, scaled.diag_before.dynamic_range);

    auto unscaled = scaled.unscale_model();
    for (size_t k = 0; k < lp.csr_values.size(); ++k) {
        EXPECT_NEAR(unscaled.csr_values[k], lp.csr_values[k], 1e-9);
    }
}

// =============================================================================
// Test 6: Netlib AFIRO Real-World Scaling Evaluation
// =============================================================================
TEST(ScalingTest, NetlibAFIROScalingDiagnostics) {
    std::string path = find_netlib_file("afiro.mps");
    if (path.empty()) GTEST_SKIP() << "Netlib afiro.mps not found";

    LinearProgram lp;
    ASSERT_TRUE(MPSParser::parse_file(path, lp).is_ok());

    Equilibrator eq;
    auto res = eq.scale(lp);
    ASSERT_TRUE(res.is_ok());
    auto scaled = res.value();

    EXPECT_LE(scaled.diag_after.row_conditioning_proxy, scaled.diag_before.row_conditioning_proxy + 1e-5);
    EXPECT_GT(scaled.row_scale_R.size(), 0);
    EXPECT_GT(scaled.col_scale_C.size(), 0);
}
