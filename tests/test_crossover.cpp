#include <gtest/gtest.h>
#include <pipepye/crossover/pdhg_crossover.hpp>
#include <pipepye/solver/pdhg_solver.hpp>
#include <pipepye/model/mps_parser.hpp>
#include <filesystem>
#include <cmath>

using namespace pipepye;
using namespace pipepye::crossover;

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

class CrossoverTest : public ::testing::Test {
protected:
    static constexpr scalar_t kTol = 1e-4;
};

TEST_F(CrossoverTest, TwoVariableLPCrossoverFromApproximatePoint) {
    // min -2 x0 - x1
    // s.t. x0 + x1 <= 4, x0 <= 3, x >= 0
    // Exact optimum: (3, 1), obj = -7.0
    model::LinearProgram lp;
    lp.name = "crossover_2d";
    lp.c = {-2.0, -1.0};
    lp.col_lower = {0.0, 0.0};
    lp.col_upper = {5.0, 5.0};
    lp.col_names = {"x0", "x1"};
    lp.row_names = {"r0", "r1"};
    lp.row_lower = {-model::Infinity, -model::Infinity};
    lp.row_upper = {4.0, 3.0};

    lp.csr_row_ptr = {0, 2, 3};
    lp.csr_col_ind = {0, 1, 0};
    lp.csr_values = {1.0, 1.0, 1.0};

    lp.csc_col_ptr = {0, 2, 3};
    lp.csc_row_ind = {0, 1, 0};
    lp.csc_values = {1.0, 1.0, 1.0};

    // Perturbed approximate PDHG solution: x ~ [2.99, 1.01]^T
    std::vector<scalar_t> pdhg_x = {2.99, 1.01};
    std::vector<scalar_t> pdhg_y = {-1.0, -1.0};

    CrossoverConfig cfg;
    cfg.active_tolerance = 0.05; // 5% tolerance recognizes active bounds

    CrossoverResult res = PDHGCrossover::run(lp, pdhg_x, pdhg_y, cfg);

    EXPECT_TRUE(res.is_optimal());
    EXPECT_NEAR(res.final_simplex_objective, -7.0, kTol);
    ASSERT_EQ(res.final_result.x.size(), 2u);
    EXPECT_NEAR(res.final_result.x[0], 3.0, kTol);
    EXPECT_NEAR(res.final_result.x[1], 1.0, kTol);
    EXPECT_TRUE(res.initial_crashed_basis.is_valid());
}

TEST_F(CrossoverTest, NetlibAFIROPDHGToSimplexCrossover) {
    std::string path = find_netlib_file("afiro.mps");
    if (path.empty()) GTEST_SKIP() << "Netlib afiro.mps not found";

    model::LinearProgram lp;
    ASSERT_TRUE(model::MPSParser::parse_file(path, lp).is_ok());

    // 1. Run PDHG CPU at moderate precision
    solver::SolverConfig pdhg_cfg = solver::SolverConfig::AdaptiveRestartCPU();
    pdhg_cfg.primal_tol = 1e-4;
    pdhg_cfg.dual_tol = 1e-4;
    pdhg_cfg.max_iterations = 20000;

    solver::SolverResult pdhg_res = solver::PDHGSolver::solve(lp, pdhg_cfg);
    ASSERT_TRUE(pdhg_res.is_converged());

    // 2. Run Crossover to snap to vertex accuracy
    CrossoverConfig xover_cfg;
    xover_cfg.active_tolerance = 1e-3;

    CrossoverResult xover_res = PDHGCrossover::run(lp, pdhg_res.x, pdhg_res.y, xover_cfg);

    EXPECT_TRUE(xover_res.is_optimal());
    // Should snap to exact AFIRO objective: -464.753142857
    EXPECT_NEAR(xover_res.final_simplex_objective, -464.753142857, 1e-4);

    // Verify final solution
    auto ver = solver::SolutionVerifier::verify(
        lp,
        xover_res.final_result.x,
        xover_res.final_result.y,
        xover_res.final_simplex_objective,
        1e-4);
    EXPECT_TRUE(ver.is_valid());
    EXPECT_TRUE(ver.is_feasible);
    EXPECT_TRUE(ver.is_objective_consistent);
}
