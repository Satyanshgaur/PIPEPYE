#include <gtest/gtest.h>
#include <pipepye/simplex/dual_simplex.hpp>
#include <pipepye/solver/solution_verifier.hpp>
#include <pipepye/model/mps_parser.hpp>
#include <pipepye/pipeline/model_pipeline.hpp>
#include <filesystem>
#include <cmath>

using namespace pipepye;
using namespace pipepye::simplex;

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

} // namespace

class DualSimplexTest : public ::testing::Test {
protected:
    static constexpr scalar_t kTol = 1e-5;
};

TEST_F(DualSimplexTest, UnconstrainedBounded1DLP) {
    // min 3 x  s.t. 2 <= x <= 5
    model::LinearProgram lp;
    lp.name = "bounded_1d";
    lp.c = {3.0};
    lp.col_lower = {2.0};
    lp.col_upper = {5.0};
    lp.col_names = {"x0"};
    lp.csr_row_ptr = {0};
    lp.csc_col_ptr = {0, 0};

    DualSimplexSolver solver(SimplexConfig::Fast());
    SimplexResult res = solver.solve(lp);

    EXPECT_TRUE(res.is_optimal());
    ASSERT_EQ(res.x.size(), 1u);
    EXPECT_NEAR(res.x[0], 2.0, kTol);
    EXPECT_NEAR(res.objective_value, 6.0, kTol);
}

TEST_F(DualSimplexTest, SingleEqualityConstraint) {
    // min 2 x0 + 5  s.t. x0 = 3, 0 <= x0 <= 10
    model::LinearProgram lp;
    lp.name = "eq_1d";
    lp.c = {2.0};
    lp.obj_offset = 5.0;
    lp.col_lower = {0.0};
    lp.col_upper = {10.0};
    lp.col_names = {"x0"};
    lp.row_names = {"c0"};
    lp.row_lower = {3.0};
    lp.row_upper = {3.0};

    // A = [1.0]
    lp.csr_row_ptr = {0, 1};
    lp.csr_col_ind = {0};
    lp.csr_values = {1.0};
    lp.csc_col_ptr = {0, 1};
    lp.csc_row_ind = {0};
    lp.csc_values = {1.0};

    DualSimplexSolver solver(SimplexConfig::Fast());
    SimplexResult res = solver.solve(lp);

    EXPECT_TRUE(res.is_optimal());
    ASSERT_EQ(res.x.size(), 1u);
    EXPECT_NEAR(res.x[0], 3.0, kTol);
    EXPECT_NEAR(res.objective_value, 11.0, kTol);
}

TEST_F(DualSimplexTest, TwoVariableLinearProgramWithVertexOptimum) {
    // min -2 x0 - x1
    // s.t.
    // row 0: x0 + x1 <= 4    (-inf <= x0 + x1 <= 4)
    // row 1: x0 <= 3         (-inf <= x0 <= 3)
    // 0 <= x0 <= 5, 0 <= x1 <= 5
    // Optimal vertex: x0 = 3, x1 = 1 => obj = -2(3) - 1 = -7.0
    model::LinearProgram lp;
    lp.name = "vertex_2d";
    lp.c = {-2.0, -1.0};
    lp.col_lower = {0.0, 0.0};
    lp.col_upper = {5.0, 5.0};
    lp.col_names = {"x0", "x1"};
    lp.row_names = {"r0", "r1"};
    lp.row_lower = {-model::Infinity, -model::Infinity};
    lp.row_upper = {4.0, 3.0};

    // A = [ 1, 1 ]
    //     [ 1, 0 ]
    lp.csr_row_ptr = {0, 2, 3};
    lp.csr_col_ind = {0, 1, 0};
    lp.csr_values = {1.0, 1.0, 1.0};

    lp.csc_col_ptr = {0, 2, 3};
    lp.csc_row_ind = {0, 1, 0};
    lp.csc_values = {1.0, 1.0, 1.0};

    DualSimplexSolver solver(SimplexConfig::Fast());
    SimplexResult res = solver.solve(lp);

    EXPECT_TRUE(res.is_optimal());
    ASSERT_EQ(res.x.size(), 2u);
    EXPECT_NEAR(res.x[0], 3.0, kTol);
    EXPECT_NEAR(res.x[1], 1.0, kTol);
    EXPECT_NEAR(res.objective_value, -7.0, kTol);

    auto ver = solver::SolutionVerifier::verify(lp, res.x, res.y, res.objective_value, 1e-4);
    EXPECT_TRUE(ver.is_valid());
    EXPECT_TRUE(ver.is_feasible);
    EXPECT_TRUE(ver.is_objective_consistent);
}

TEST_F(DualSimplexTest, InfeasibleLPDetected) {
    // min x  s.t. x <= 2 and x >= 5 (contradictory)
    model::LinearProgram lp;
    lp.name = "infeasible";
    lp.c = {1.0};
    lp.col_lower = {0.0};
    lp.col_upper = {10.0};
    lp.row_names = {"r0", "r1"};
    lp.row_lower = {-model::Infinity, 5.0};
    lp.row_upper = {2.0, model::Infinity};

    lp.csr_row_ptr = {0, 1, 2};
    lp.csr_col_ind = {0, 0};
    lp.csr_values = {1.0, 1.0};

    lp.csc_col_ptr = {0, 2};
    lp.csc_row_ind = {0, 1};
    lp.csc_values = {1.0, 1.0};

    DualSimplexSolver solver(SimplexConfig::Fast());
    SimplexResult res = solver.solve(lp);

    EXPECT_TRUE(res.status == solver::TerminationStatus::PRIMAL_INFEASIBLE);
}

TEST_F(DualSimplexTest, PricingStrategyAblationDantzigVsDevex) {
    model::LinearProgram lp;
    lp.name = "pricing_ablation";
    lp.c = {-3.0, -2.0};
    lp.col_lower = {0.0, 0.0};
    lp.col_upper = {10.0, 10.0};
    lp.row_names = {"r0", "r1"};
    lp.row_lower = {-model::Infinity, -model::Infinity};
    lp.row_upper = {6.0, 4.0};

    lp.csr_row_ptr = {0, 2, 3};
    lp.csr_col_ind = {0, 1, 0};
    lp.csr_values = {2.0, 1.0, 1.0};

    lp.csc_col_ptr = {0, 2, 3};
    lp.csc_row_ind = {0, 1, 0};
    lp.csc_values = {2.0, 1.0, 1.0};

    SimplexConfig cfg_dantzig = SimplexConfig::Fast();
    cfg_dantzig.pricing = PricingStrategy::Dantzig;
    DualSimplexSolver solver_dantzig(cfg_dantzig);
    SimplexResult res_dantzig = solver_dantzig.solve(lp);

    SimplexConfig cfg_devex = SimplexConfig::Fast();
    cfg_devex.pricing = PricingStrategy::Devex;
    DualSimplexSolver solver_devex(cfg_devex);
    SimplexResult res_devex = solver_devex.solve(lp);

    EXPECT_TRUE(res_dantzig.is_optimal());
    EXPECT_TRUE(res_devex.is_optimal());
    EXPECT_NEAR(res_dantzig.objective_value, res_devex.objective_value, kTol);
}

TEST_F(DualSimplexTest, NetlibAFIRODirectSolveAndVerification) {
    std::string path = find_netlib_file("afiro.mps");
    if (path.empty()) GTEST_SKIP() << "Netlib afiro.mps not found";

    model::LinearProgram lp;
    ASSERT_TRUE(model::MPSParser::parse_file(path, lp).is_ok());

    DualSimplexSolver solver(SimplexConfig::Fast());
    SimplexResult res = solver.solve(lp);

    EXPECT_TRUE(res.is_optimal());
    // Known exact optimal objective for AFIRO: -464.753142857
    EXPECT_NEAR(res.objective_value, -464.753142857, 1e-4);

    auto ver = solver::SolutionVerifier::verify(lp, res.x, res.y, res.objective_value, 1e-4);
    EXPECT_TRUE(ver.is_valid());
    EXPECT_TRUE(ver.is_feasible);
    EXPECT_TRUE(ver.is_objective_consistent);
}

TEST_F(DualSimplexTest, NetlibBLENDDirectSolveAndVerification) {
    std::string path = find_netlib_file("blend.mps");
    if (path.empty()) GTEST_SKIP() << "Netlib blend.mps not found";

    model::LinearProgram lp;
    ASSERT_TRUE(model::MPSParser::parse_file(path, lp).is_ok());

    DualSimplexSolver solver(SimplexConfig::Fast());
    SimplexResult res = solver.solve(lp);

    EXPECT_TRUE(res.is_optimal());
    // Known exact optimal objective for BLEND: -30.8121498
    EXPECT_NEAR(res.objective_value, -30.8121498, 1e-3);

    auto ver = solver::SolutionVerifier::verify(lp, res.x, res.y, res.objective_value, 1e-3);
    EXPECT_TRUE(ver.is_valid());
    EXPECT_TRUE(ver.is_feasible);
}

TEST_F(DualSimplexTest, EndToEndSolvePipelineIntegrationAFIRO) {
    std::string path = find_netlib_file("afiro.mps");
    if (path.empty()) GTEST_SKIP() << "Netlib afiro.mps not found";

    model::LinearProgram orig_lp;
    ASSERT_TRUE(model::MPSParser::parse_file(path, orig_lp).is_ok());

    pipeline::PipelineConfig pipe_cfg = pipeline::PipelineConfig::PresolveAndScaling();
    SimplexConfig sim_cfg = SimplexConfig::Fast();

    auto [res, ver] = DualSimplexSolver::solve_end_to_end(orig_lp, pipe_cfg, sim_cfg, 1e-4);

    EXPECT_TRUE(res.is_optimal());
    EXPECT_NEAR(res.objective_value, -464.753142857, 1e-4);
    EXPECT_TRUE(ver.is_valid());
    EXPECT_TRUE(ver.is_feasible);
    EXPECT_TRUE(ver.is_objective_consistent);
}
