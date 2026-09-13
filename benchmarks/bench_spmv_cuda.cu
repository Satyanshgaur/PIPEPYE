#include <pipepye/cuda/device_info.cuh>
#include <pipepye/cuda/cuda_check.cuh>
#include <pipepye/cuda/spmv.cuh>
#include <pipepye/cuda/reductions.cuh>
#include <pipepye/sparse/matrix_generator.hpp>
#include <pipepye/sparse/cpu_ops.hpp>
#include <pipepye/model/mps_parser.hpp>
#include <pipepye/utils/timer.hpp>

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <cmath>
#include <filesystem>

using namespace pipepye;
using namespace pipepye::cuda;
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

void print_sep(char c = '=', int len = 110) {
    std::cout << std::string(len, c) << "\n";
}

} // namespace

int main() {
    int dev_count = get_device_count();
    if (dev_count == 0) {
        std::cerr << "No CUDA device found!\n";
        return 1;
    }

    DeviceProperties dev = query_device(0);
    print_sep('=', 110);
    std::cout << " PipePye CUDA SpMV Variants & Reductions Performance Benchmark\n";
    print_sep('=', 110);
    std::cout << " Device: " << dev.name << " (Compute " << dev.major << "." << dev.minor
              << ", " << dev.multi_processor_count << " SMs, "
              << (dev.total_memory_bytes / (1024 * 1024)) << " MiB VRAM)\n\n";

    // =========================================================================
    // SECTION 1: CUDA Reductions Benchmark
    // =========================================================================
    std::cout << ">>> SECTION 1: CUDA Warp/Block-Level Reductions (Dot, Norm2, NormInf)\n";
    print_sep('-', 110);
    std::cout << std::left
              << std::setw(14) << "Operation"
              << std::setw(14) << "Size (N)"
              << std::setw(16) << "CPU Time (ms)"
              << std::setw(16) << "GPU Time (ms)"
              << std::setw(18) << "GPU GFLOPS"
              << std::setw(18) << "GPU BW (GB/s)"
              << std::setw(14) << "Speedup"
              << "\n";
    print_sep('-', 110);

    for (size_t n : {100'000UL, 1'000'000UL, 10'000'000UL}) {
        std::vector<scalar_t> h_x(n, 1.001);
        std::vector<scalar_t> h_y(n, 0.999);
        ConstVectorView x_view(h_x.data(), n);
        ConstVectorView y_view(h_y.data(), n);

        DeviceVector d_x(x_view);
        DeviceVector d_y(y_view);

        // Benchmark Dot Product
        {
            // CPU single-thread
            utils::CPUTimer cpu_timer;
            cpu_timer.start();
            volatile scalar_t cpu_res = 0.0;
            const int runs = (n >= 10'000'000) ? 10 : 30;
            for (int r = 0; r < runs; ++r) {
                cpu_res = cpu_ops::dot(x_view, y_view);
            }
            cpu_timer.stop();
            double cpu_time_ms = cpu_timer.elapsed_milliseconds() / runs;
            (void)cpu_res;

            // GPU warm-up
            for (int w = 0; w < 5; ++w) {
                (void)cuda_dot(d_x.data(), d_y.data(), n);
            }

            cudaEvent_t start, stop;
            CUDA_CHECK(cudaEventCreate(&start));
            CUDA_CHECK(cudaEventCreate(&stop));

            CUDA_CHECK(cudaEventRecord(start));
            for (int r = 0; r < runs; ++r) {
                (void)cuda_dot(d_x.data(), d_y.data(), n);
            }
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));

            float total_gpu_ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&total_gpu_ms, start, stop));
            double gpu_time_ms = total_gpu_ms / runs;

            double elapsed_sec = gpu_time_ms / 1000.0;
            double gflops = (2.0 * static_cast<double>(n)) / (elapsed_sec * 1e9);
            double bw_gbs = (16.0 * static_cast<double>(n)) / (elapsed_sec * 1e9);
            double speedup = cpu_time_ms / gpu_time_ms;

            std::cout << std::left
                      << std::setw(14) << "Dot Product"
                      << std::setw(14) << n
                      << std::fixed << std::setprecision(4)
                      << std::setw(16) << cpu_time_ms
                      << std::setw(16) << gpu_time_ms
                      << std::setprecision(2)
                      << std::setw(18) << gflops
                      << std::setw(18) << bw_gbs
                      << std::setprecision(2)
                      << std::setw(14) << (std::to_string(speedup).substr(0, 4) + "x")
                      << "\n";

            CUDA_CHECK(cudaEventDestroy(start));
            CUDA_CHECK(cudaEventDestroy(stop));
        }

        // Benchmark Norm 2
        {
            utils::CPUTimer cpu_timer;
            cpu_timer.start();
            volatile scalar_t cpu_res = 0.0;
            const int runs = (n >= 10'000'000) ? 10 : 30;
            for (int r = 0; r < runs; ++r) {
                cpu_res = x_view.norm_2();
            }
            cpu_timer.stop();
            double cpu_time_ms = cpu_timer.elapsed_milliseconds() / runs;
            (void)cpu_res;

            for (int w = 0; w < 5; ++w) {
                (void)cuda_norm_2(d_x.data(), n);
            }

            cudaEvent_t start, stop;
            CUDA_CHECK(cudaEventCreate(&start));
            CUDA_CHECK(cudaEventCreate(&stop));

            CUDA_CHECK(cudaEventRecord(start));
            for (int r = 0; r < runs; ++r) {
                (void)cuda_norm_2(d_x.data(), n);
            }
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));

            float total_gpu_ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&total_gpu_ms, start, stop));
            double gpu_time_ms = total_gpu_ms / runs;

            double elapsed_sec = gpu_time_ms / 1000.0;
            double gflops = (2.0 * static_cast<double>(n)) / (elapsed_sec * 1e9);
            double bw_gbs = (8.0 * static_cast<double>(n)) / (elapsed_sec * 1e9);
            double speedup = cpu_time_ms / gpu_time_ms;

            std::cout << std::left
                      << std::setw(14) << "Norm-2 (L2)"
                      << std::setw(14) << n
                      << std::fixed << std::setprecision(4)
                      << std::setw(16) << cpu_time_ms
                      << std::setw(16) << gpu_time_ms
                      << std::setprecision(2)
                      << std::setw(18) << gflops
                      << std::setw(18) << bw_gbs
                      << std::setprecision(2)
                      << std::setw(14) << (std::to_string(speedup).substr(0, 4) + "x")
                      << "\n";

            CUDA_CHECK(cudaEventDestroy(start));
            CUDA_CHECK(cudaEventDestroy(stop));
        }

        // Benchmark Norm Inf
        {
            utils::CPUTimer cpu_timer;
            cpu_timer.start();
            volatile scalar_t cpu_res = 0.0;
            const int runs = (n >= 10'000'000) ? 10 : 30;
            for (int r = 0; r < runs; ++r) {
                cpu_res = x_view.norm_inf();
            }
            cpu_timer.stop();
            double cpu_time_ms = cpu_timer.elapsed_milliseconds() / runs;
            (void)cpu_res;

            for (int w = 0; w < 5; ++w) {
                (void)cuda_norm_inf(d_x.data(), n);
            }

            cudaEvent_t start, stop;
            CUDA_CHECK(cudaEventCreate(&start));
            CUDA_CHECK(cudaEventCreate(&stop));

            CUDA_CHECK(cudaEventRecord(start));
            for (int r = 0; r < runs; ++r) {
                (void)cuda_norm_inf(d_x.data(), n);
            }
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));

            float total_gpu_ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&total_gpu_ms, start, stop));
            double gpu_time_ms = total_gpu_ms / runs;

            double elapsed_sec = gpu_time_ms / 1000.0;
            double bw_gbs = (8.0 * static_cast<double>(n)) / (elapsed_sec * 1e9);
            double speedup = cpu_time_ms / gpu_time_ms;

            std::cout << std::left
                      << std::setw(14) << "Norm-Inf"
                      << std::setw(14) << n
                      << std::fixed << std::setprecision(4)
                      << std::setw(16) << cpu_time_ms
                      << std::setw(16) << gpu_time_ms
                      << std::setprecision(2)
                      << std::setw(18) << "-"
                      << std::setw(18) << bw_gbs
                      << std::setprecision(2)
                      << std::setw(14) << (std::to_string(speedup).substr(0, 4) + "x")
                      << "\n";

            CUDA_CHECK(cudaEventDestroy(start));
            CUDA_CHECK(cudaEventDestroy(stop));
        }
    }

    std::cout << "\n";

    // =========================================================================
    // SECTION 2: CUDA SpMV Strategy Benchmark across Matrix Topologies
    // =========================================================================
    std::cout << ">>> SECTION 2: CUDA CSR SpMV Variants Across Controlled Matrix Topologies\n";
    print_sep('-', 110);

    struct BenchmarkCase {
        std::string name;
        std::string desc;
        CSRMatrix csr;
    };

    std::vector<BenchmarkCase> cases;
    {
        COOMatrix coo = MatrixGenerator::generate_random(10'000, 10'000, 0.002, -5.0, 5.0, 101);
        cases.push_back({"Random Sparse", "10k x 10k, uniform density 0.2%", coo.to_csr()});
    }
    {
        COOMatrix coo = MatrixGenerator::generate_banded(20'000, 20'000, 15, 15, -3.0, 3.0, 102);
        cases.push_back({"Banded Matrix", "20k x 20k, bw=31, highly localized", coo.to_csr()});
    }
    {
        COOMatrix coo = MatrixGenerator::generate_block_diagonal(50, 200, 200, 0.03, 0.0005, -2.0, 2.0, 103);
        cases.push_back({"Block-Diagonal", "10k x 10k, 50 blocks of 200x200", coo.to_csr()});
    }
    {
        COOMatrix coo = MatrixGenerator::generate_staircase(100, 100, 100, 0.03, -2.0, 2.0, 104);
        cases.push_back({"Staircase Matrix", "10k x 10.1k, 100 time-staged periods", coo.to_csr()});
    }
    {
        COOMatrix coo = MatrixGenerator::generate_irregular(10'000, 10'000, 200'000, 0.05, 0.50, -4.0, 4.0, 105);
        cases.push_back({"Irregular Hub", "10k x 10k, 5% hub rows hold 50% NNZ", coo.to_csr()});
    }

    // Add Netlib real problems
    std::string beaconfd_path = find_netlib_file("beaconfd.mps");
    if (!beaconfd_path.empty()) {
        model::LinearProgram lp;
        if (model::MPSParser::parse_file(beaconfd_path, lp).is_ok()) {
            cases.push_back({"Netlib BEACONFD", "Real Netlib LP (173 x 262)", lp.to_csr()});
        }
    }

    for (const auto& bc : cases) {
        index_t m = bc.csr.num_rows();
        index_t n = bc.csr.num_cols();
        size_t nnz = bc.csr.num_nonzeros();

        std::cout << "\n Matrix: " << bc.name << " (" << m << " x " << n << ", NNZ=" << nnz << ")\n";
        print_sep('-', 110);
        std::cout << std::left
                  << std::setw(28) << "Kernel Strategy"
                  << std::setw(16) << "Time (ms)"
                  << std::setw(18) << "Throughput (GFLOPS)"
                  << std::setw(18) << "Bandwidth (GB/s)"
                  << std::setw(16) << "Speedup (vs CPU 1T)"
                  << std::setw(14) << "vs GPU Scalar"
                  << "\n";
        print_sep('-', 110);

        // Measure CPU single-thread baseline
        std::vector<scalar_t> h_x(n, 1.0);
        std::vector<scalar_t> h_y(m, 0.0);
        ConstVectorView x_view(h_x.data(), n);
        MutableVectorView y_view(h_y.data(), m);

        utils::CPUTimer cpu_timer;
        cpu_timer.start();
        const int cpu_runs = (nnz < 10'000) ? 200 : ((nnz < 200'000) ? 40 : 20);
        for (int r = 0; r < cpu_runs; ++r) {
            cpu_ops::spmv_csr(1.0, bc.csr, x_view, 0.0, y_view);
        }
        cpu_timer.stop();
        double cpu_time_ms = cpu_timer.elapsed_milliseconds() / cpu_runs;

        // Device setup
        DeviceCSRMatrix d_mat(bc.csr);
        DeviceVector d_x(x_view);
        DeviceVector d_y(y_view);

        double scalar_gpu_ms = 0.0;

        for (auto variant : {SpMVKernelVariant::Scalar,
                             SpMVKernelVariant::Vector,
                             SpMVKernelVariant::Adaptive,
                             SpMVKernelVariant::Balanced}) {
            // Warmup
            for (int w = 0; w < 10; ++w) {
                d_mat.spmv(1.0, d_x, 0.0, d_y, variant);
            }

            cudaEvent_t start, stop;
            CUDA_CHECK(cudaEventCreate(&start));
            CUDA_CHECK(cudaEventCreate(&stop));

            const int gpu_runs = (nnz < 10'000) ? 300 : ((nnz < 200'000) ? 100 : 50);
            CUDA_CHECK(cudaEventRecord(start));
            for (int r = 0; r < gpu_runs; ++r) {
                d_mat.spmv(1.0, d_x, 0.0, d_y, variant);
            }
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));

            float total_gpu_ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&total_gpu_ms, start, stop));
            double gpu_time_ms = total_gpu_ms / gpu_runs;

            if (variant == SpMVKernelVariant::Scalar) {
                scalar_gpu_ms = gpu_time_ms;
            }

            double elapsed_sec = gpu_time_ms / 1000.0;
            double gflops = (2.0 * static_cast<double>(nnz)) / (elapsed_sec * 1e9);
            double bytes = 12.0 * static_cast<double>(nnz) + 4.0 * static_cast<double>(m + 1) + 8.0 * static_cast<double>(m + n);
            double bw_gbs = bytes / (elapsed_sec * 1e9);
            double speedup_cpu = cpu_time_ms / gpu_time_ms;
            double speedup_scalar = scalar_gpu_ms / gpu_time_ms;

            std::cout << std::left
                      << std::setw(28) << to_string(variant)
                      << std::fixed << std::setprecision(4)
                      << std::setw(16) << gpu_time_ms
                      << std::setprecision(2)
                      << std::setw(18) << gflops
                      << std::setw(18) << bw_gbs
                      << std::setprecision(2)
                      << std::setw(16) << (std::to_string(speedup_cpu).substr(0, 4) + "x")
                      << std::setw(14) << (std::to_string(speedup_scalar).substr(0, 4) + "x")
                      << "\n";

            CUDA_CHECK(cudaEventDestroy(start));
            CUDA_CHECK(cudaEventDestroy(stop));
        }
    }

    std::cout << "\n";
    print_sep('=', 110);
    std::cout << " Benchmark Complete.\n";
    print_sep('=', 110);
    return 0;
}
