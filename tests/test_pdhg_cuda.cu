#include <gtest/gtest.h>
#include <pipepye/solver/pdhg_solver.hpp>
#include <pipepye/solver/pdhg_solver_cpu.hpp>
#include <pipepye/solver/pdhg_solver_cuda.hpp>
#include <pipepye/solver/solution_verifier.hpp>
#include <pipepye/model/mps_parser.hpp>
#include <pipepye/pipeline/model_pipeline.hpp>
#include <cmath>

using namespace pipepye;
using namespace pipepye::solver;

class PDHGCUDATest : public ::testing::Test {
protected:
    void SetUp() override {
        // Ensure CUDA solver is explicitly initialized / registered
        init_cuda_solver();
    }
};

TEST_F(PDHGCUDATest, SingleEqualityConstraintCUDA) {
    // min 2*x0 + 3*x1
    // s.t. x0 + x1 = 4
    //      x0, x1 >= 0
    // Optimum: x0 = 4, x1 = 0, obj = 8.0
    model::LinearProgram lp;
    lp.name = "single_eq_cuda";
    lp.c = {2.0, 3.0};
    lp.col_lower = {0.0, 0.0};
    lp.col_upper = {1e15, 1e15};
    lp.col_names = {"x0", "x1"};
    lp.row_names = {"c0"};
    lp.row_lower = {4.0};
    lp.row_upper = {4.0};

    lp.csr_row_ptr = {0, 2};
    lp.csr_col_ind = {0, 1};
    lp.csr_values = {1.0, 1.0};

    SolverConfig config = SolverConfig::BaselineCUDA();
    config.primal_tol = 1e-4;
    config.dual_tol = 1e-4;
    config.max_iterations = 20000;

    CudaPDPOptimizer optimizer(config);
    SolverResult res = optimizer.solve(lp);

    EXPECT_TRUE(res.is_converged());
    EXPECT_NEAR(res.primal_objective, 8.0, 5e-3);
    EXPECT_NEAR(res.x[0], 4.0, 1e-2);
    EXPECT_NEAR(res.x[1], 0.0, 1e-2);
    EXPECT_NEAR(res.x[0] + res.x[1], 4.0, 5e-3);

    // Verify Timing Decomposition
    EXPECT_GT(res.timing.h2d_transfer_ms, 0.0);
    EXPECT_GT(res.timing.pure_solve_ms, 0.0);
    EXPECT_GT(res.timing.d2h_transfer_ms, 0.0);
    EXPECT_GT(res.timing.total_time_ms, 0.0);
}

TEST_F(PDHGCUDATest, MultipleConstraintsParityAgainstCPU) {
    // min -3*x0 - 5*x1
    // s.t.   x0 +   x1 <= 4
    //        x0 + 3*x1 <= 6
    //        x0, x1 >= 0
    // Optimum at intersection x0 = 3, x1 = 1, obj = -14.0
    model::LinearProgram lp;
    lp.name = "multi_2d_parity";
    lp.c = {-3.0, -5.0};
    lp.col_lower = {0.0, 0.0};
    lp.col_upper = {10.0, 10.0};
    lp.col_names = {"x0", "x1"};
    lp.row_names = {"c0", "c1"};
    lp.row_lower = {-1e15, -1e15};
    lp.row_upper = {4.0, 6.0};

    lp.csr_row_ptr = {0, 2, 4};
    lp.csr_col_ind = {0, 1, 0, 1};
    lp.csr_values = {1.0, 1.0, 1.0, 3.0};

    SolverConfig cfg_cpu = SolverConfig::BaselineCPU();
    cfg_cpu.primal_tol = 1e-4;
    cfg_cpu.dual_tol = 1e-4;
    cfg_cpu.max_iterations = 30000;
    SolverResult res_cpu = CPUPDPOptimizer(cfg_cpu).solve(lp);

    SolverConfig cfg_cuda = SolverConfig::BaselineCUDA();
    cfg_cuda.primal_tol = 1e-4;
    cfg_cuda.dual_tol = 1e-4;
    cfg_cuda.max_iterations = 30000;
    SolverResult res_cuda = CudaPDPOptimizer(cfg_cuda).solve(lp);

    EXPECT_TRUE(res_cpu.is_converged());
    EXPECT_TRUE(res_cuda.is_converged());

    // Parity checks
    EXPECT_NEAR(res_cpu.primal_objective, res_cuda.primal_objective, 1e-2);
    ASSERT_EQ(res_cpu.x.size(), res_cuda.x.size());
    for (size_t j = 0; j < res_cpu.x.size(); ++j) {
        EXPECT_NEAR(res_cpu.x[j], res_cuda.x[j], 2e-2);
    }

    VerificationResult ver = SolutionVerifier::verify(lp, res_cuda.x, res_cuda.y, res_cuda.primal_objective, 1e-2);
    EXPECT_TRUE(ver.is_valid());
    EXPECT_TRUE(ver.is_feasible);
}

#include <filesystem>

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

TEST_F(PDHGCUDATest, NetlibAFIROSolveCUDA) {
    std::string filepath = find_netlib_file("afiro.mps");
    ASSERT_FALSE(filepath.empty());

    model::LinearProgram orig_lp;
    auto parse_status = model::MPSParser::parse_file(filepath, orig_lp);
    ASSERT_TRUE(parse_status.is_ok());

    pipeline::PipelineConfig pipe_cfg = pipeline::PipelineConfig::PresolveAndScaling();
    SolverConfig solver_cfg = SolverConfig::AdaptiveRestartCUDA();
    solver_cfg.primal_tol = 1e-5;
    solver_cfg.dual_tol = 1e-5;
    solver_cfg.max_iterations = 40000;

    auto [res, ver] = PDHGSolver::solve_end_to_end(orig_lp, pipe_cfg, solver_cfg, 0.05);

    std::cout << "PDHG CUDA AFIRO solve completed in " << res.iterations << " iterations. Status: " << static_cast<int>(res.status) << "\n";
    std::cout << ver.format_report() << "\n";

    EXPECT_TRUE(res.is_converged());
    EXPECT_NEAR(res.primal_objective, -464.753, 2.0);
    EXPECT_TRUE(ver.is_valid());
    EXPECT_TRUE(ver.is_feasible);
    EXPECT_TRUE(ver.is_objective_consistent);

    EXPECT_GT(res.timing.h2d_transfer_ms, 0.0);
    EXPECT_GT(res.timing.pure_solve_ms, 0.0);
    EXPECT_GT(res.timing.d2h_transfer_ms, 0.0);
    EXPECT_GT(res.timing.postsolve_ms, 0.0);
    EXPECT_GT(res.timing.verification_ms, 0.0);
    EXPECT_GT(res.timing.total_time_ms, 0.0);
}
