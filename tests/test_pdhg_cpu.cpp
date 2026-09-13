#include <gtest/gtest.h>
#include <pipepye/solver/pdhg_solver.hpp>
#include <pipepye/solver/pdhg_solver_cpu.hpp>
#include <pipepye/solver/solution_verifier.hpp>
#include <pipepye/model/mps_parser.hpp>
#include <pipepye/pipeline/model_pipeline.hpp>
#include <cmath>

using namespace pipepye;
using namespace pipepye::solver;

class PDHGCPUTest : public ::testing::Test {
protected:
    static constexpr scalar_t kTol = 1e-3;
};

TEST_F(PDHGCPUTest, UnconstrainedBounded1DLP) {
    // min x  s.t. 2 <= x <= 5
    model::LinearProgram lp;
    lp.name = "bounded_1d";
    lp.c = {1.0};
    lp.col_lower = {2.0};
    lp.col_upper = {5.0};
    lp.col_names = {"x0"};
    lp.csr_row_ptr = {0};

    SolverConfig config = SolverConfig::BaselineCPU();
    config.primal_tol = 1e-4;
    config.dual_tol = 1e-4;

    CPUPDPOptimizer optimizer(config);
    SolverResult res = optimizer.solve(lp);

    EXPECT_TRUE(res.is_converged());
    ASSERT_EQ(res.x.size(), 1u);
    EXPECT_NEAR(res.x[0], 2.0, 1e-5);
    EXPECT_NEAR(res.primal_objective, 2.0, 1e-5);
}

TEST_F(PDHGCPUTest, SingleEqualityConstraint) {
    // min 2*x0 + 3*x1
    // s.t. x0 + x1 = 4
    //      x0, x1 >= 0
    // Optimum: x0 = 4, x1 = 0, obj = 8.0
    model::LinearProgram lp;
    lp.name = "single_eq";
    lp.c = {2.0, 3.0};
    lp.col_lower = {0.0, 0.0};
    lp.col_upper = {1e15, 1e15};
    lp.col_names = {"x0", "x1"};
    lp.row_names = {"c0"};
    lp.row_lower = {4.0};
    lp.row_upper = {4.0};

    // A = [1.0, 1.0]
    lp.csr_row_ptr = {0, 2};
    lp.csr_col_ind = {0, 1};
    lp.csr_values = {1.0, 1.0};

    SolverConfig config = SolverConfig::BaselineCPU();
    config.primal_tol = 1e-4;
    config.dual_tol = 1e-4;
    config.max_iterations = 20000;

    CPUPDPOptimizer optimizer(config);
    SolverResult res = optimizer.solve(lp);

    EXPECT_TRUE(res.is_converged());
    EXPECT_NEAR(res.primal_objective, 8.0, 5e-3);
    EXPECT_NEAR(res.x[0], 4.0, 1e-2);
    EXPECT_NEAR(res.x[1], 0.0, 1e-2);
    EXPECT_NEAR(res.x[0] + res.x[1], 4.0, 5e-3);
}

TEST_F(PDHGCPUTest, SingleInequalityLessThan) {
    // min -x0 - 2*x1
    // s.t. x0 + x1 <= 3
    //      0 <= x0, x1 <= 10
    // Optimum: x0 = 0, x1 = 3, obj = -6.0
    model::LinearProgram lp;
    lp.name = "single_ineq_le";
    lp.c = {-1.0, -2.0};
    lp.col_lower = {0.0, 0.0};
    lp.col_upper = {10.0, 10.0};
    lp.col_names = {"x0", "x1"};
    lp.row_names = {"c0"};
    lp.row_lower = {-1e15};
    lp.row_upper = {3.0};

    lp.csr_row_ptr = {0, 2};
    lp.csr_col_ind = {0, 1};
    lp.csr_values = {1.0, 1.0};

    SolverConfig config = SolverConfig::BaselineCPU();
    config.primal_tol = 1e-4;
    config.dual_tol = 1e-4;
    config.max_iterations = 20000;

    CPUPDPOptimizer optimizer(config);
    SolverResult res = optimizer.solve(lp);

    EXPECT_TRUE(res.is_converged());
    EXPECT_NEAR(res.primal_objective, -6.0, 5e-3);
    EXPECT_NEAR(res.x[0], 0.0, 1e-2);
    EXPECT_NEAR(res.x[1], 3.0, 1e-2);
}

TEST_F(PDHGCPUTest, SingleInequalityGreaterThan) {
    // min x0 + x1
    // s.t. 2*x0 + x1 >= 4
    //      x0, x1 >= 0
    // Optimum: x0 = 2, x1 = 0, obj = 2.0
    model::LinearProgram lp;
    lp.name = "single_ineq_ge";
    lp.c = {1.0, 1.0};
    lp.col_lower = {0.0, 0.0};
    lp.col_upper = {10.0, 10.0};
    lp.col_names = {"x0", "x1"};
    lp.row_names = {"c0"};
    lp.row_lower = {4.0};
    lp.row_upper = {1e15};

    lp.csr_row_ptr = {0, 2};
    lp.csr_col_ind = {0, 1};
    lp.csr_values = {2.0, 1.0};

    SolverConfig config = SolverConfig::BaselineCPU();
    config.primal_tol = 1e-4;
    config.dual_tol = 1e-4;
    config.max_iterations = 20000;

    CPUPDPOptimizer optimizer(config);
    SolverResult res = optimizer.solve(lp);

    EXPECT_TRUE(res.is_converged());
    EXPECT_NEAR(res.primal_objective, 2.0, 5e-3);
    EXPECT_NEAR(res.x[0], 2.0, 1e-2);
    EXPECT_NEAR(res.x[1], 0.0, 1e-2);
}

TEST_F(PDHGCPUTest, MultipleConstraints2DLP) {
    // min -3*x0 - 5*x1
    // s.t.   x0 +   x1 <= 4
    //        x0 + 3*x1 <= 6
    //        x0, x1 >= 0
    // Optimum at intersection x0 = 3, x1 = 1, obj = -14.0
    model::LinearProgram lp;
    lp.name = "multi_2d";
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

    SolverConfig config = SolverConfig::BaselineCPU();
    config.primal_tol = 1e-4;
    config.dual_tol = 1e-4;
    config.max_iterations = 30000;

    CPUPDPOptimizer optimizer(config);
    SolverResult res = optimizer.solve(lp);

    EXPECT_TRUE(res.is_converged());
    EXPECT_NEAR(res.primal_objective, -14.0, 1e-2);
    EXPECT_NEAR(res.x[0], 3.0, 2e-2);
    EXPECT_NEAR(res.x[1], 1.0, 2e-2);

    VerificationResult ver = SolutionVerifier::verify(lp, res.x, res.y, res.primal_objective, 1e-2);
    EXPECT_TRUE(ver.is_valid());
}

TEST_F(PDHGCPUTest, DegenerateZeroObjective) {
    // min 0*x0 + 0*x1
    // s.t. x0 + x1 = 1, x0, x1 >= 0
    // Any point on simplex is optimal, obj = 0.0
    model::LinearProgram lp;
    lp.name = "zero_obj";
    lp.c = {0.0, 0.0};
    lp.col_lower = {0.0, 0.0};
    lp.col_upper = {1.0, 1.0};
    lp.col_names = {"x0", "x1"};
    lp.row_names = {"c0"};
    lp.row_lower = {1.0};
    lp.row_upper = {1.0};

    lp.csr_row_ptr = {0, 2};
    lp.csr_col_ind = {0, 1};
    lp.csr_values = {1.0, 1.0};

    SolverConfig config = SolverConfig::BaselineCPU();
    config.primal_tol = 1e-4;
    config.dual_tol = 1e-4;

    CPUPDPOptimizer optimizer(config);
    SolverResult res = optimizer.solve(lp);

    EXPECT_TRUE(res.is_converged());
    EXPECT_NEAR(res.primal_objective, 0.0, 1e-4);
    EXPECT_NEAR(res.x[0] + res.x[1], 1.0, 1e-3);
}

TEST_F(PDHGCPUTest, InfeasibleLPHandling) {
    // min x0 s.t. x0 <= 1, x0 >= 3
    model::LinearProgram lp;
    lp.name = "infeasible_lp";
    lp.c = {1.0};
    lp.col_lower = {0.0};
    lp.col_upper = {10.0};
    lp.col_names = {"x0"};
    lp.row_names = {"c0", "c1"};
    lp.row_lower = {-1e15, 3.0};
    lp.row_upper = {1.0, 1e15};

    lp.csr_row_ptr = {0, 1, 2};
    lp.csr_col_ind = {0, 0};
    lp.csr_values = {1.0, 1.0};

    SolverConfig config = SolverConfig::BaselineCPU();
    config.max_iterations = 2000;

    CPUPDPOptimizer optimizer(config);
    SolverResult res = optimizer.solve(lp);

    // It should either not converge to optimal or report infeasibility / iteration limit
    EXPECT_FALSE(res.status == TerminationStatus::OPTIMAL);
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

TEST_F(PDHGCPUTest, StepSizeAblationStudy) {
    // Test on 2D LP: CONSTANT vs POCK_CHAMBOLLE vs ADAPTIVE
    model::LinearProgram lp;
    lp.col_names = {"x0", "x1"};
    lp.row_names = {"c0", "c1"};
    lp.c = {-3.0, -5.0};
    lp.col_lower = {0.0, 0.0};
    lp.col_upper = {10.0, 10.0};
    lp.row_lower = {-1e15, -1e15};
    lp.row_upper = {4.0, 6.0};
    lp.csr_row_ptr = {0, 2, 4};
    lp.csr_col_ind = {0, 1, 0, 1};
    lp.csr_values = {1.0, 1.0, 1.0, 3.0};

    // 1. Constant
    SolverConfig cfg_const = SolverConfig::BaselineCPU();
    cfg_const.step_size_strategy = StepSizeStrategy::CONSTANT;
    cfg_const.primal_tol = 1e-4;
    cfg_const.dual_tol = 1e-4;
    cfg_const.max_iterations = 30000;
    SolverResult res_const = CPUPDPOptimizer(cfg_const).solve(lp);

    // 2. Pock-Chambolle diagonal
    SolverConfig cfg_pc = SolverConfig::BaselineCPU();
    cfg_pc.step_size_strategy = StepSizeStrategy::POCK_CHAMBOLLE;
    cfg_pc.primal_tol = 1e-4;
    cfg_pc.dual_tol = 1e-4;
    cfg_pc.max_iterations = 30000;
    SolverResult res_pc = CPUPDPOptimizer(cfg_pc).solve(lp);

    // 3. Adaptive
    SolverConfig cfg_adapt = SolverConfig::AdaptiveCPU();
    cfg_adapt.primal_tol = 1e-4;
    cfg_adapt.dual_tol = 1e-4;
    cfg_adapt.max_iterations = 30000;
    SolverResult res_adapt = CPUPDPOptimizer(cfg_adapt).solve(lp);

    EXPECT_TRUE(res_const.is_converged());
    EXPECT_TRUE(res_pc.is_converged());
    EXPECT_TRUE(res_adapt.is_converged());

    EXPECT_NEAR(res_const.primal_objective, -14.0, 5e-2);
    EXPECT_NEAR(res_pc.primal_objective, -14.0, 5e-2);
    EXPECT_NEAR(res_adapt.primal_objective, -14.0, 5e-2);
}

TEST_F(PDHGCPUTest, RestartAblationStudy) {
    model::LinearProgram lp;
    lp.col_names = {"x0", "x1"};
    lp.row_names = {"c0", "c1"};
    lp.c = {-3.0, -5.0};
    lp.col_lower = {0.0, 0.0};
    lp.col_upper = {10.0, 10.0};
    lp.row_lower = {-1e15, -1e15};
    lp.row_upper = {4.0, 6.0};
    lp.csr_row_ptr = {0, 2, 4};
    lp.csr_col_ind = {0, 1, 0, 1};
    lp.csr_values = {1.0, 1.0, 1.0, 3.0};

    SolverConfig cfg_no_rst = SolverConfig::BaselineCPU();
    cfg_no_rst.restart_strategy = RestartStrategy::NONE;
    cfg_no_rst.primal_tol = 1e-4;
    cfg_no_rst.dual_tol = 1e-4;
    cfg_no_rst.max_iterations = 30000;

    SolverConfig cfg_adapt_rst = SolverConfig::AdaptiveRestartCPU();
    cfg_adapt_rst.primal_tol = 1e-4;
    cfg_adapt_rst.dual_tol = 1e-4;
    cfg_adapt_rst.max_iterations = 30000;

    SolverResult res_no_rst = CPUPDPOptimizer(cfg_no_rst).solve(lp);
    SolverResult res_adapt_rst = CPUPDPOptimizer(cfg_adapt_rst).solve(lp);

    EXPECT_TRUE(res_no_rst.is_converged());
    EXPECT_TRUE(res_adapt_rst.is_converged());
    EXPECT_NEAR(res_no_rst.primal_objective, -14.0, 5e-2);
    EXPECT_NEAR(res_adapt_rst.primal_objective, -14.0, 5e-2);
}

TEST_F(PDHGCPUTest, SolutionVerifierCatchesViolations) {
    model::LinearProgram lp;
    lp.col_names = {"x0", "x1"};
    lp.row_names = {"c0"};
    lp.c = {1.0, 2.0};
    lp.col_lower = {0.0, 0.0};
    lp.col_upper = {2.0, 2.0};
    lp.row_lower = {1.0};
    lp.row_upper = {3.0};
    lp.csr_row_ptr = {0, 2};
    lp.csr_col_ind = {0, 1};
    lp.csr_values = {1.0, 1.0};

    // Valid solution: x = [1, 1], Ax = 2 in [1, 3], bounds satisfied, obj = 3.0
    std::vector<scalar_t> x_good = {1.0, 1.0};
    std::vector<scalar_t> y_good = {1.0};
    VerificationResult ver_good = SolutionVerifier::verify(lp, x_good, y_good, 3.0, 1e-4);
    EXPECT_TRUE(ver_good.is_valid());
    EXPECT_TRUE(ver_good.is_feasible);
    EXPECT_TRUE(ver_good.is_objective_consistent);

    // Corrupted variable bound: x0 = 3.5 > 2.0
    std::vector<scalar_t> x_bad_bound = {3.5, 0.0};
    VerificationResult ver_bad_bound = SolutionVerifier::verify(lp, x_bad_bound, y_good, 3.5, 1e-4);
    EXPECT_FALSE(ver_bad_bound.is_valid());
    EXPECT_FALSE(ver_bad_bound.is_feasible);

    // Corrupted row constraint: x = [0, 0] => Ax = 0 < 1.0
    std::vector<scalar_t> x_bad_row = {0.0, 0.0};
    VerificationResult ver_bad_row = SolutionVerifier::verify(lp, x_bad_row, y_good, 0.0, 1e-4);
    EXPECT_FALSE(ver_bad_row.is_valid());
    EXPECT_FALSE(ver_bad_row.is_feasible);

    // Corrupted objective claim: obj = 99.0 instead of 3.0
    VerificationResult ver_bad_obj = SolutionVerifier::verify(lp, x_good, y_good, 99.0, 1e-4);
    EXPECT_FALSE(ver_bad_obj.is_valid());
    EXPECT_FALSE(ver_bad_obj.is_objective_consistent);
}

TEST_F(PDHGCPUTest, EndToEndSolveNetlibAFIRO) {
    std::string filepath = find_netlib_file("afiro.mps");
    ASSERT_FALSE(filepath.empty());

    model::LinearProgram orig_lp;
    auto parse_status = model::MPSParser::parse_file(filepath, orig_lp);
    ASSERT_TRUE(parse_status.is_ok());

    pipeline::PipelineConfig pipe_cfg = pipeline::PipelineConfig::PresolveAndScaling();
    SolverConfig solver_cfg = SolverConfig::AdaptiveRestartCPU();
    solver_cfg.primal_tol = 1e-5;
    solver_cfg.dual_tol = 1e-5;
    solver_cfg.max_iterations = 40000;

    auto [res, ver] = PDHGSolver::solve_end_to_end(orig_lp, pipe_cfg, solver_cfg, 0.5);

    std::cout << "PDHG AFIRO solve completed in " << res.iterations << " iterations. Status: " << static_cast<int>(res.status) << "\n";
    std::cout << ver.format_report() << "\n";

    EXPECT_TRUE(res.is_converged());
    // AFIRO optimal objective is -464.753142857 (PDHG gets -463.6 to -464.7 depending on tol)
    EXPECT_NEAR(res.primal_objective, -464.753, 2.0);
    EXPECT_TRUE(ver.is_valid());
    EXPECT_TRUE(ver.is_feasible);
    EXPECT_TRUE(ver.is_objective_consistent);
}
