#include <pipepye/model/mps_parser.hpp>
#include <pipepye/simplex/dual_simplex.hpp>
#include <pipepye/utils/timer.hpp>
#include <iostream>
#include <iomanip>
#include <vector>

using namespace pipepye;

int main() {
    std::vector<std::string> lps = {
        "workloads/case_a_crude_blending/Small.mps",
        "workloads/case_a_crude_blending/Medium.mps",
        "workloads/case_a_crude_blending/Large.mps",
        "workloads/case_b_multi_period_planning/T10.mps",
        "workloads/case_b_multi_period_planning/T25.mps",
        "workloads/case_b_multi_period_planning/T50.mps",
        "workloads/case_b_multi_period_planning/T100.mps"
    };

    simplex::DualSimplexSolver solver;
    std::cout << std::left << std::setw(45) << "Model" 
              << std::setw(10) << "Rows" 
              << std::setw(10) << "Cols" 
              << std::setw(10) << "Pivots" 
              << std::setw(15) << "Time (ms)" 
              << "Objective\n";
    std::cout << std::string(95, '-') << "\n";

    for (const auto& path : lps) {
        model::LinearProgram lp;
        auto st = model::MPSParser::parse_file(path, lp);
        if (!st.is_ok()) {
            std::cerr << "Failed to parse: " << path << "\n";
            continue;
        }
        
        utils::CPUTimer t;
        auto res = solver.solve(lp);
        double ms = t.elapsed_milliseconds();

        std::cout << std::left << std::setw(45) << path 
                  << std::setw(10) << lp.num_rows() 
                  << std::setw(10) << lp.num_cols() 
                  << std::setw(10) << res.iterations 
                  << std::fixed << std::setprecision(3) << std::setw(15) << ms 
                  << std::scientific << std::setprecision(6) << res.objective_value << "\n";
    }
    return 0;
}
