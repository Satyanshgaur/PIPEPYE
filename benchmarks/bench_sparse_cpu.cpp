#include <pipepye/sparse/vector.hpp>
#include <pipepye/sparse/coo_matrix.hpp>
#include <pipepye/sparse/csr_matrix.hpp>
#include <pipepye/sparse/csc_matrix.hpp>
#include <pipepye/sparse/cpu_ops.hpp>
#include <pipepye/sparse/matrix_generator.hpp>
#include <pipepye/model/mps_parser.hpp>
#include <pipepye/utils/timer.hpp>

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <cmath>
#include <filesystem>
#include <numeric>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace pipepye;
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

struct BenchmarkRecord {
    std::string category;
    std::string name;
    index_t rows{0};
    index_t cols{0};
    size_t nnz{0};
    int threads{1};
    double time_ms{0.0};
    double gflops{0.0};
    double bandwidth_gbs{0.0};
    double speedup{1.0};
    double efficiency_pct{100.0};
};

void print_separator(char c = '=', int len = 105) {
    std::cout << std::string(len, c) << "\n";
}

} // namespace

int main(int argc, char** argv) {
    std::cout << "\n";
    print_separator('=', 105);
    std::cout << " PipePye Sparse Linear Algebra & Vector Primitives CPU Benchmark Suite\n";
    print_separator('=', 105);

    int max_threads = cpu_ops::get_max_threads();
    std::cout << " [System Topology] Available Execution Threads: " << max_threads << "\n";
#ifdef _OPENMP
    std::cout << " [OpenMP Version] OpenMP " << _OPENMP << " Enabled\n";
#else
    std::cout << " [OpenMP Version] Disabled (Running single-threaded fallback)\n";
#endif
    std::cout << "\n";

    std::vector<int> thread_counts;
    for (int t : {1, 2, 4, 8, max_threads}) {
        if (t <= max_threads && (thread_counts.empty() || t > thread_counts.back())) {
            thread_counts.push_back(t);
        }
    }

    std::vector<BenchmarkRecord> all_records;

    // =========================================================================
    // 1. Dense Vector Primitives Benchmark (AXPY, Dot Product, L2 Norm)
    // =========================================================================
    std::cout << ">>> BENCHMARK SECTION 1: Dense Vector Primitives (BLAS-1)\n";
    print_separator('-', 105);
    std::cout << std::left 
              << std::setw(15) << "Operation"
              << std::setw(15) << "Size (N)"
              << std::setw(10) << "Threads"
              << std::setw(16) << "Time (ms)"
              << std::setw(16) << "Throughput (GFLOPS)"
              << std::setw(18) << "Bandwidth (GB/s)"
              << std::setw(12) << "Speedup"
              << "\n";
    print_separator('-', 105);

    const std::vector<index_t> vector_sizes = {100'000, 1'000'000, 10'000'000};

    // Vector AXPY: y = alpha * x + y
    for (index_t n : vector_sizes) {
        std::vector<scalar_t> x(n, 1.5);
        std::vector<scalar_t> y(n, 0.5);
        ConstVectorView x_view(x.data(), n);
        MutableVectorView y_view(y.data(), n);
        const scalar_t alpha = 2.0;

        double baseline_time_ms = 0.0;

        for (int threads : thread_counts) {
            // Warmup
            for (int w = 0; w < 5; ++w) {
                if (threads == 1) {
                    cpu_ops::axpy(alpha, x_view, y_view);
                } else {
                    cpu_ops::axpy_parallel(alpha, x_view, y_view, threads);
                }
            }

            const int runs = (n >= 10'000'000) ? 20 : 50;
            utils::CPUTimer timer;
            timer.start();
            for (int r = 0; r < runs; ++r) {
                if (threads == 1) {
                    cpu_ops::axpy(alpha, x_view, y_view);
                } else {
                    cpu_ops::axpy_parallel(alpha, x_view, y_view, threads);
                }
            }
            timer.stop();

            double elapsed_ms = timer.elapsed_milliseconds() / runs;
            if (threads == 1) baseline_time_ms = elapsed_ms;

            double elapsed_sec = elapsed_ms / 1000.0;
            double gflops = (2.0 * static_cast<double>(n)) / (elapsed_sec * 1e9);
            // 3 memory operations per element: read x (8B), read y (8B), write y (8B) = 24B
            double bandwidth_gbs = (24.0 * static_cast<double>(n)) / (elapsed_sec * 1e9);
            double speedup = baseline_time_ms / elapsed_ms;
            double efficiency = (speedup / threads) * 100.0;

            BenchmarkRecord rec{
                "Vector", "AXPY", n, 1, static_cast<size_t>(n),
                threads, elapsed_ms, gflops, bandwidth_gbs, speedup, efficiency
            };
            all_records.push_back(rec);

            std::cout << std::left
                      << std::setw(15) << "AXPY"
                      << std::setw(15) << n
                      << std::setw(10) << threads
                      << std::fixed << std::setprecision(4)
                      << std::setw(16) << elapsed_ms
                      << std::setprecision(2)
                      << std::setw(16) << gflops
                      << std::setw(18) << bandwidth_gbs
                      << std::setprecision(2)
                      << std::setw(12) << (std::to_string(speedup).substr(0, 4) + "x")
                      << "\n";
        }
    }

    // Vector Dot Product: sum(x_i * y_i)
    for (index_t n : vector_sizes) {
        std::vector<scalar_t> x(n, 1.0001);
        std::vector<scalar_t> y(n, 0.9999);
        ConstVectorView x_view(x.data(), n);
        ConstVectorView y_view(y.data(), n);

        double baseline_time_ms = 0.0;

        for (int threads : thread_counts) {
            // Warmup
            volatile scalar_t dummy = 0.0;
            for (int w = 0; w < 5; ++w) {
                if (threads == 1) {
                    dummy = cpu_ops::dot(x_view, y_view);
                } else {
                    dummy = cpu_ops::dot_parallel(x_view, y_view, threads);
                }
            }

            const int runs = (n >= 10'000'000) ? 20 : 50;
            utils::CPUTimer timer;
            timer.start();
            for (int r = 0; r < runs; ++r) {
                if (threads == 1) {
                    dummy = cpu_ops::dot(x_view, y_view);
                } else {
                    dummy = cpu_ops::dot_parallel(x_view, y_view, threads);
                }
            }
            timer.stop();
            (void)dummy;

            double elapsed_ms = timer.elapsed_milliseconds() / runs;
            if (threads == 1) baseline_time_ms = elapsed_ms;

            double elapsed_sec = elapsed_ms / 1000.0;
            double gflops = (2.0 * static_cast<double>(n)) / (elapsed_sec * 1e9);
            // 2 memory reads per element: read x (8B), read y (8B) = 16B
            double bandwidth_gbs = (16.0 * static_cast<double>(n)) / (elapsed_sec * 1e9);
            double speedup = baseline_time_ms / elapsed_ms;
            double efficiency = (speedup / threads) * 100.0;

            BenchmarkRecord rec{
                "Vector", "DOT", n, 1, static_cast<size_t>(n),
                threads, elapsed_ms, gflops, bandwidth_gbs, speedup, efficiency
            };
            all_records.push_back(rec);

            std::cout << std::left
                      << std::setw(15) << "DOT"
                      << std::setw(15) << n
                      << std::setw(10) << threads
                      << std::fixed << std::setprecision(4)
                      << std::setw(16) << elapsed_ms
                      << std::setprecision(2)
                      << std::setw(16) << gflops
                      << std::setw(18) << bandwidth_gbs
                      << std::setprecision(2)
                      << std::setw(12) << (std::to_string(speedup).substr(0, 4) + "x")
                      << "\n";
        }
    }

    std::cout << "\n";

    // =========================================================================
    // 2. Controlled Sparse Matrices Benchmark (SpMV & SpMV^T)
    // =========================================================================
    std::cout << ">>> BENCHMARK SECTION 2: Controlled Sparse Matrices (SpMV & SpMV^T)\n";
    print_separator('-', 105);

    struct MatrixBenchmarkCase {
        std::string name;
        std::string description;
        CSRMatrix csr;
        CSCMatrix csc;
    };

    std::vector<MatrixBenchmarkCase> matrix_cases;

    std::cout << "Generating controlled benchmark matrices...\n";

    // Case A: Random Uniform Sparse Matrix
    {
        COOMatrix coo = MatrixGenerator::generate_random(10'000, 10'000, 0.002, -5.0, 5.0, 1001);
        matrix_cases.push_back({"Random (10k x 10k)", "Uniform random, density 0.2%", coo.to_csr(), coo.to_csc()});
    }

    // Case B: Banded Sparse Matrix
    {
        COOMatrix coo = MatrixGenerator::generate_banded(20'000, 20'000, 15, 15, -2.0, 4.0, 1002);
        matrix_cases.push_back({"Banded (20k x 20k)", "Diagonally dominant band (bw=31)", coo.to_csr(), coo.to_csc()});
    }

    // Case C: Block-Diagonal Sparse Matrix
    {
        COOMatrix coo = MatrixGenerator::generate_block_diagonal(50, 200, 200, 0.03, 0.0005, -3.0, 3.0, 1003);
        matrix_cases.push_back({"Block-Diag (10k x 10k)", "50 blocks of 200x200 with weak coupling", coo.to_csr(), coo.to_csc()});
    }

    // Case D: Staircase Sparse Matrix (Multi-stage LP topology)
    {
        COOMatrix coo = MatrixGenerator::generate_staircase(100, 100, 100, 0.03, -2.0, 2.0, 1004);
        matrix_cases.push_back({"Staircase (10k x 10.1k)", "100 stages, inter-temporal constraints", coo.to_csr(), coo.to_csc()});
    }

    // Case E: Irregular Sparse Matrix (Hub-and-spoke power-law)
    {
        COOMatrix coo = MatrixGenerator::generate_irregular(10'000, 10'000, 200'000, 0.05, 0.50, -4.0, 4.0, 1005);
        matrix_cases.push_back({"Irregular (10k x 10k)", "5% hub rows hold 50% nonzeros", coo.to_csr(), coo.to_csc()});
    }

    // Case F: Real Netlib Matrix (beaconfd.mps)
    std::string beaconfd_path = find_netlib_file("beaconfd.mps");
    if (!beaconfd_path.empty()) {
        model::LinearProgram lp;
        if (model::MPSParser::parse_file(beaconfd_path, lp).is_ok()) {
            matrix_cases.push_back({"Netlib BEACONFD", "Real Netlib LP benchmark", lp.to_csr(), lp.to_csc()});
        }
    }

    // Case G: Real Netlib Matrix (bandm.mps)
    std::string bandm_path = find_netlib_file("bandm.mps");
    if (!bandm_path.empty()) {
        model::LinearProgram lp;
        if (model::MPSParser::parse_file(bandm_path, lp).is_ok()) {
            matrix_cases.push_back({"Netlib BANDM", "Real Netlib LP benchmark", lp.to_csr(), lp.to_csc()});
        }
    }

    for (const auto& mc : matrix_cases) {
        index_t m = mc.csr.num_rows();
        index_t n = mc.csr.num_cols();
        size_t nnz = mc.csr.num_nonzeros();
        double density = (static_cast<double>(nnz) / (static_cast<double>(m) * static_cast<double>(n))) * 100.0;

        std::cout << "\n Matrix: " << mc.name << " [" << mc.description << "]\n";
        std::cout << " Dimensions: " << m << " x " << n << " | Nonzeros: " << nnz << " (" << std::fixed << std::setprecision(4) << density << "% density)\n";
        print_separator('-', 105);
        std::cout << std::left
                  << std::setw(12) << "Op"
                  << std::setw(10) << "Threads"
                  << std::setw(16) << "Time (ms)"
                  << std::setw(16) << "Throughput (GFLOPS)"
                  << std::setw(18) << "Bandwidth (GB/s)"
                  << std::setw(14) << "Speedup"
                  << std::setw(14) << "Efficiency"
                  << "\n";
        print_separator('-', 105);

        // Benchmark SpMV: y = A * x (CSR)
        {
            std::vector<scalar_t> x(n, 1.0);
            std::vector<scalar_t> y(m, 0.0);
            ConstVectorView x_view(x.data(), n);
            MutableVectorView y_view(y.data(), m);

            double baseline_time_ms = 0.0;

            for (int threads : thread_counts) {
                // Warmup
                for (int w = 0; w < 10; ++w) {
                    if (threads == 1) {
                        cpu_ops::spmv_csr(1.0, mc.csr, x_view, 0.0, y_view);
                    } else {
                        cpu_ops::spmv_csr_parallel(1.0, mc.csr, x_view, 0.0, y_view, threads);
                    }
                }

                const int runs = (nnz < 10'000) ? 200 : ((nnz < 200'000) ? 50 : 25);
                utils::CPUTimer timer;
                timer.start();
                for (int r = 0; r < runs; ++r) {
                    if (threads == 1) {
                        cpu_ops::spmv_csr(1.0, mc.csr, x_view, 0.0, y_view);
                    } else {
                        cpu_ops::spmv_csr_parallel(1.0, mc.csr, x_view, 0.0, y_view, threads);
                    }
                }
                timer.stop();

                double elapsed_ms = timer.elapsed_milliseconds() / runs;
                if (threads == 1) baseline_time_ms = elapsed_ms;

                double elapsed_sec = elapsed_ms / 1000.0;
                double gflops = (2.0 * static_cast<double>(nnz)) / (elapsed_sec * 1e9);
                // Data traffic: row_ptr (4*(m+1)), col_ind (4*nnz), values (8*nnz), read x (8*n), write y (8*m)
                double bytes = 12.0 * static_cast<double>(nnz) + 4.0 * static_cast<double>(m + 1) + 8.0 * static_cast<double>(m + n);
                double bandwidth_gbs = bytes / (elapsed_sec * 1e9);
                double speedup = baseline_time_ms / elapsed_ms;
                double efficiency = (speedup / threads) * 100.0;

                BenchmarkRecord rec{
                    "SpMV", mc.name, m, n, nnz,
                    threads, elapsed_ms, gflops, bandwidth_gbs, speedup, efficiency
                };
                all_records.push_back(rec);

                std::cout << std::left
                          << std::setw(12) << "SpMV (Ax)"
                          << std::setw(10) << threads
                          << std::fixed << std::setprecision(4)
                          << std::setw(16) << elapsed_ms
                          << std::setprecision(2)
                          << std::setw(16) << gflops
                          << std::setw(18) << bandwidth_gbs
                          << std::setprecision(2)
                          << std::setw(14) << (std::to_string(speedup).substr(0, 4) + "x")
                          << std::setw(14) << (std::to_string(efficiency).substr(0, 5) + "%")
                          << "\n";
            }
        }

        // Benchmark SpMV^T: y = A^T * x (CSC)
        {
            std::vector<scalar_t> x(m, 1.0);
            std::vector<scalar_t> y(n, 0.0);
            ConstVectorView x_view(x.data(), m);
            MutableVectorView y_view(y.data(), n);

            double baseline_time_ms = 0.0;

            for (int threads : thread_counts) {
                // Warmup
                for (int w = 0; w < 10; ++w) {
                    if (threads == 1) {
                        cpu_ops::spmv_transpose_csc(1.0, mc.csc, x_view, 0.0, y_view);
                    } else {
                        cpu_ops::spmv_transpose_csc_parallel(1.0, mc.csc, x_view, 0.0, y_view, threads);
                    }
                }

                const int runs = (nnz < 10'000) ? 200 : ((nnz < 200'000) ? 50 : 25);
                utils::CPUTimer timer;
                timer.start();
                for (int r = 0; r < runs; ++r) {
                    if (threads == 1) {
                        cpu_ops::spmv_transpose_csc(1.0, mc.csc, x_view, 0.0, y_view);
                    } else {
                        cpu_ops::spmv_transpose_csc_parallel(1.0, mc.csc, x_view, 0.0, y_view, threads);
                    }
                }
                timer.stop();

                double elapsed_ms = timer.elapsed_milliseconds() / runs;
                if (threads == 1) baseline_time_ms = elapsed_ms;

                double elapsed_sec = elapsed_ms / 1000.0;
                double gflops = (2.0 * static_cast<double>(nnz)) / (elapsed_sec * 1e9);
                double bytes = 12.0 * static_cast<double>(nnz) + 4.0 * static_cast<double>(n + 1) + 8.0 * static_cast<double>(m + n);
                double bandwidth_gbs = bytes / (elapsed_sec * 1e9);
                double speedup = baseline_time_ms / elapsed_ms;
                double efficiency = (speedup / threads) * 100.0;

                BenchmarkRecord rec{
                    "SpMV^T", mc.name, m, n, nnz,
                    threads, elapsed_ms, gflops, bandwidth_gbs, speedup, efficiency
                };
                all_records.push_back(rec);

                std::cout << std::left
                          << std::setw(12) << "SpMV^T (A^Tx)"
                          << std::setw(10) << threads
                          << std::fixed << std::setprecision(4)
                          << std::setw(16) << elapsed_ms
                          << std::setprecision(2)
                          << std::setw(16) << gflops
                          << std::setw(18) << bandwidth_gbs
                          << std::setprecision(2)
                          << std::setw(14) << (std::to_string(speedup).substr(0, 4) + "x")
                          << std::setw(14) << (std::to_string(efficiency).substr(0, 5) + "%")
                          << "\n";
            }
        }
    }

    std::cout << "\n";
    print_separator('=', 105);
    std::cout << " Benchmark Suite Complete. Recorded " << all_records.size() << " performance data points.\n";
    print_separator('=', 105);
    std::cout << "\n";

    return 0;
}
