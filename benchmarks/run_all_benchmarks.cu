#include <pipepye/cuda/device_info.cuh>
#include <pipepye/cuda/cuda_check.cuh>
#include <pipepye/cuda/spmv.cuh>
#include <pipepye/cuda/reductions.cuh>
#include <pipepye/sparse/matrix_generator.hpp>
#include <pipepye/sparse/cpu_ops.hpp>
#include <pipepye/model/mps_parser.hpp>
#include <pipepye/utils/timer.hpp>

#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <string>
#include <cmath>
#include <filesystem>
#include <sstream>

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

struct RowStats {
    index_t min_nnz{0};
    index_t max_nnz{0};
    double avg_nnz{0.0};
    double stddev_nnz{0.0};
};

RowStats compute_row_stats(const CSRMatrix& csr) {
    if (csr.num_rows() == 0) return {0, 0, 0.0, 0.0};
    index_t min_nnz = csr.row_nnz(0);
    index_t max_nnz = csr.row_nnz(0);
    double sum = 0.0;

    for (index_t i = 0; i < csr.num_rows(); ++i) {
        index_t rnnz = csr.row_nnz(i);
        if (rnnz < min_nnz) min_nnz = rnnz;
        if (rnnz > max_nnz) max_nnz = rnnz;
        sum += rnnz;
    }

    double avg = sum / csr.num_rows();
    double var_sum = 0.0;
    for (index_t i = 0; i < csr.num_rows(); ++i) {
        double diff = csr.row_nnz(i) - avg;
        var_sum += diff * diff;
    }
    double stddev = std::sqrt(var_sum / csr.num_rows());
    return {min_nnz, max_nnz, avg, stddev};
}

struct BenchmarkRecord {
    std::string experiment_type;
    std::string matrix_name;
    std::string topology;
    index_t rows{0};
    index_t cols{0};
    size_t nnz{0};
    double density_pct{0.0};
    index_t min_row_nnz{0};
    index_t max_row_nnz{0};
    double avg_row_nnz{0.0};
    double stddev_row_nnz{0.0};
    std::string variant;
    double runtime_ms{0.0};
    double gflops{0.0};
    double bandwidth_gbs{0.0};
    double speedup_vs_cpu_1t{1.0};
    double speedup_vs_gpu_scalar{1.0};
    std::string verification_status{"PASSED"};
    double max_abs_error{0.0};
};

struct ReductionRecord {
    std::string operation;
    size_t size{0};
    double data_mb{0.0};
    std::string variant;
    double runtime_ms{0.0};
    double gflops{0.0};
    double bandwidth_gbs{0.0};
    double speedup_vs_cpu_1t{1.0};
    std::string verification_status{"PASSED"};
};

} // namespace

int main(int argc, char** argv) {
    std::string out_dir = "benchmarks/results";
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--out-dir" && i + 1 < argc) {
            out_dir = argv[++i];
        }
    }
    std::filesystem::create_directories(out_dir);

    int dev_count = get_device_count();
    if (dev_count == 0) {
        std::cerr << "Error: No CUDA GPU available for automated benchmark runner.\n";
        return 1;
    }

    DeviceProperties dev = query_device(0);
    int cpu_max_threads = cpu_ops::get_max_threads();

    std::cout << "========================================================================================\n";
    std::cout << " PipePye Automated Benchmark Runner: CPU vs CUDA High-Resolution Profiling Engine\n";
    std::cout << " Host CPU: Intel Core i5-13420H (" << cpu_max_threads << " threads)\n";
    std::cout << " Target GPU: " << dev.name << " (sm_" << dev.major << dev.minor << ", "
              << dev.multi_processor_count << " SMs, " << (dev.total_memory_bytes / (1024 * 1024)) << " MiB VRAM)\n";
    std::cout << " Output Directory: " << out_dir << "\n";
    std::cout << "========================================================================================\n\n";

    std::vector<BenchmarkRecord> matrix_records;
    std::vector<ReductionRecord> reduction_records;

    // =========================================================================
    // 1. Vector Reductions Experiments (Dot, Norm2, NormInf)
    // =========================================================================
    std::cout << "[1/4] Running Vector Reductions Sweep (Dot, Norm2, NormInf)...\n";
    const std::vector<size_t> reduction_sizes = {
        10'000UL, 50'000UL, 100'000UL, 500'000UL, 1'000'000UL, 5'000'000UL, 10'000'000UL
    };

    for (size_t n : reduction_sizes) {
        std::vector<scalar_t> h_x(n);
        std::vector<scalar_t> h_y(n);
        for (size_t i = 0; i < n; ++i) {
            h_x[i] = std::sin(static_cast<double>(i) * 0.05);
            h_y[i] = std::cos(static_cast<double>(i) * 0.05);
        }
        ConstVectorView x_view(h_x.data(), n);
        ConstVectorView y_view(h_y.data(), n);

        DeviceVector d_x(x_view);
        DeviceVector d_y(y_view);

        const int red_runs = (n >= 5'000'000) ? 10 : 30;

        // A. Dot Product
        {
            // CPU 1T
            utils::CPUTimer t_cpu1;
            t_cpu1.start();
            volatile scalar_t cpu_res = 0.0;
            for (int r = 0; r < red_runs; ++r) {
                cpu_res = cpu_ops::dot(x_view, y_view);
            }
            t_cpu1.stop();
            double cpu1_ms = t_cpu1.elapsed_milliseconds() / red_runs;

            // CPU 12T
            utils::CPUTimer t_cpu12;
            t_cpu12.start();
            volatile scalar_t cpu12_res = 0.0;
            for (int r = 0; r < red_runs; ++r) {
                cpu12_res = cpu_ops::dot_parallel(x_view, y_view, cpu_max_threads);
            }
            t_cpu12.stop();
            double cpu12_ms = t_cpu12.elapsed_milliseconds() / red_runs;
            (void)cpu12_res;

            // GPU
            for (int w = 0; w < 5; ++w) { (void)cuda_dot(d_x.data(), d_y.data(), n); }
            cudaEvent_t ev_start, ev_stop;
            CUDA_CHECK(cudaEventCreate(&ev_start));
            CUDA_CHECK(cudaEventCreate(&ev_stop));

            CUDA_CHECK(cudaEventRecord(ev_start));
            for (int r = 0; r < red_runs; ++r) { (void)cuda_dot(d_x.data(), d_y.data(), n); }
            CUDA_CHECK(cudaEventRecord(ev_stop));
            CUDA_CHECK(cudaEventSynchronize(ev_stop));

            float gpu_tot_ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&gpu_tot_ms, ev_start, ev_stop));
            double gpu_ms = gpu_tot_ms / red_runs;
            CUDA_CHECK(cudaEventDestroy(ev_start));
            CUDA_CHECK(cudaEventDestroy(ev_stop));

            // Verify
            scalar_t gpu_val = cuda_dot(d_x.data(), d_y.data(), n);
            double rel_diff = std::abs(gpu_val - cpu_res) / (std::abs(cpu_res) + 1e-15);
            std::string status = (rel_diff < 1e-11) ? "PASSED" : "FAILED";

            double mb = (2.0 * n * sizeof(scalar_t)) / (1024.0 * 1024.0);
            double gflops = (2.0 * n) / ((gpu_ms / 1000.0) * 1e9);
            double bw = (16.0 * n) / ((gpu_ms / 1000.0) * 1e9);

            reduction_records.push_back({"dot", n, mb, "CPU_1T", cpu1_ms, (2.0 * n) / ((cpu1_ms / 1000.0) * 1e9),
                                         (16.0 * n) / ((cpu1_ms / 1000.0) * 1e9), 1.0, "PASSED"});
            reduction_records.push_back({"dot", n, mb, "CPU_12T", cpu12_ms, (2.0 * n) / ((cpu12_ms / 1000.0) * 1e9),
                                         (16.0 * n) / ((cpu12_ms / 1000.0) * 1e9), cpu1_ms / cpu12_ms, "PASSED"});
            reduction_records.push_back({"dot", n, mb, "CUDA", gpu_ms, gflops, bw, cpu1_ms / gpu_ms, status});
        }

        // B. Norm-2
        {
            utils::CPUTimer t_cpu1;
            t_cpu1.start();
            volatile scalar_t cpu_res = 0.0;
            for (int r = 0; r < red_runs; ++r) { cpu_res = x_view.norm_2(); }
            t_cpu1.stop();
            double cpu1_ms = t_cpu1.elapsed_milliseconds() / red_runs;

            for (int w = 0; w < 5; ++w) { (void)cuda_norm_2(d_x.data(), n); }
            cudaEvent_t ev_start, ev_stop;
            CUDA_CHECK(cudaEventCreate(&ev_start));
            CUDA_CHECK(cudaEventCreate(&ev_stop));

            CUDA_CHECK(cudaEventRecord(ev_start));
            for (int r = 0; r < red_runs; ++r) { (void)cuda_norm_2(d_x.data(), n); }
            CUDA_CHECK(cudaEventRecord(ev_stop));
            CUDA_CHECK(cudaEventSynchronize(ev_stop));

            float gpu_tot_ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&gpu_tot_ms, ev_start, ev_stop));
            double gpu_ms = gpu_tot_ms / red_runs;
            CUDA_CHECK(cudaEventDestroy(ev_start));
            CUDA_CHECK(cudaEventDestroy(ev_stop));

            scalar_t gpu_val = cuda_norm_2(d_x.data(), n);
            double rel_diff = std::abs(gpu_val - cpu_res) / (std::abs(cpu_res) + 1e-15);
            std::string status = (rel_diff < 1e-11) ? "PASSED" : "FAILED";

            double mb = (n * sizeof(scalar_t)) / (1024.0 * 1024.0);
            double gflops = (2.0 * n) / ((gpu_ms / 1000.0) * 1e9);
            double bw = (8.0 * n) / ((gpu_ms / 1000.0) * 1e9);

            reduction_records.push_back({"norm_2", n, mb, "CPU_1T", cpu1_ms, (2.0 * n) / ((cpu1_ms / 1000.0) * 1e9),
                                         (8.0 * n) / ((cpu1_ms / 1000.0) * 1e9), 1.0, "PASSED"});
            reduction_records.push_back({"norm_2", n, mb, "CUDA", gpu_ms, gflops, bw, cpu1_ms / gpu_ms, status});
        }

        // C. Norm-Inf
        {
            utils::CPUTimer t_cpu1;
            t_cpu1.start();
            volatile scalar_t cpu_res = 0.0;
            for (int r = 0; r < red_runs; ++r) { cpu_res = x_view.norm_inf(); }
            t_cpu1.stop();
            double cpu1_ms = t_cpu1.elapsed_milliseconds() / red_runs;

            for (int w = 0; w < 5; ++w) { (void)cuda_norm_inf(d_x.data(), n); }
            cudaEvent_t ev_start, ev_stop;
            CUDA_CHECK(cudaEventCreate(&ev_start));
            CUDA_CHECK(cudaEventCreate(&ev_stop));

            CUDA_CHECK(cudaEventRecord(ev_start));
            for (int r = 0; r < red_runs; ++r) { (void)cuda_norm_inf(d_x.data(), n); }
            CUDA_CHECK(cudaEventRecord(ev_stop));
            CUDA_CHECK(cudaEventSynchronize(ev_stop));

            float gpu_tot_ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&gpu_tot_ms, ev_start, ev_stop));
            double gpu_ms = gpu_tot_ms / red_runs;
            CUDA_CHECK(cudaEventDestroy(ev_start));
            CUDA_CHECK(cudaEventDestroy(ev_stop));

            scalar_t gpu_val = cuda_norm_inf(d_x.data(), n);
            std::string status = (gpu_val == cpu_res) ? "PASSED" : "FAILED";

            double mb = (n * sizeof(scalar_t)) / (1024.0 * 1024.0);
            double bw = (8.0 * n) / ((gpu_ms / 1000.0) * 1e9);

            reduction_records.push_back({"norm_inf", n, mb, "CPU_1T", cpu1_ms, 0.0,
                                         (8.0 * n) / ((cpu1_ms / 1000.0) * 1e9), 1.0, "PASSED"});
            reduction_records.push_back({"norm_inf", n, mb, "CUDA", gpu_ms, 0.0, bw, cpu1_ms / gpu_ms, status});
        }
    }

    // Helper lambda to run all 6 variants (CPU 1T, CPU 12T, CUDA Scalar, Vector, Adaptive, Balanced) on a matrix
    auto evaluate_matrix = [&](const std::string& exp_type, const std::string& name, const std::string& topology, const CSRMatrix& csr) {
        index_t m = csr.num_rows();
        index_t n = csr.num_cols();
        size_t nnz = csr.num_nonzeros();
        double density = (static_cast<double>(nnz) / (static_cast<double>(m) * static_cast<double>(n))) * 100.0;
        RowStats rstats = compute_row_stats(csr);

        std::vector<scalar_t> h_x(n, 1.0);
        std::vector<scalar_t> h_y_ref(m, 0.0);
        std::vector<scalar_t> h_y_check(m, 0.0);

        ConstVectorView x_view(h_x.data(), n);
        MutableVectorView y_ref_view(h_y_ref.data(), m);

        const int cpu_runs = (nnz < 10'000) ? 200 : ((nnz < 200'000) ? 40 : 20);
        const int gpu_runs = (nnz < 10'000) ? 300 : ((nnz < 200'000) ? 100 : 50);

        // 1. CPU Single-Thread
        utils::CPUTimer t_cpu1;
        t_cpu1.start();
        for (int r = 0; r < cpu_runs; ++r) {
            cpu_ops::spmv_csr(1.0, csr, x_view, 0.0, y_ref_view);
        }
        t_cpu1.stop();
        double cpu1_ms = t_cpu1.elapsed_milliseconds() / cpu_runs;

        double bytes = 12.0 * nnz + 4.0 * (m + 1) + 8.0 * (m + n);
        double cpu1_gflops = (2.0 * nnz) / ((cpu1_ms / 1000.0) * 1e9);
        double cpu1_bw = bytes / ((cpu1_ms / 1000.0) * 1e9);

        matrix_records.push_back({
            exp_type, name, topology, m, n, nnz, density,
            rstats.min_nnz, rstats.max_nnz, rstats.avg_nnz, rstats.stddev_nnz,
            "CPU_1T", cpu1_ms, cpu1_gflops, cpu1_bw, 1.0, 1.0, "PASSED", 0.0
        });

        // 2. CPU Multi-Thread (OpenMP 12T)
        utils::CPUTimer t_cpu12;
        t_cpu12.start();
        for (int r = 0; r < cpu_runs; ++r) {
            cpu_ops::spmv_csr_parallel(1.0, csr, x_view, 0.0, y_ref_view, cpu_max_threads);
        }
        t_cpu12.stop();
        double cpu12_ms = t_cpu12.elapsed_milliseconds() / cpu_runs;
        double cpu12_gflops = (2.0 * nnz) / ((cpu12_ms / 1000.0) * 1e9);
        double cpu12_bw = bytes / ((cpu12_ms / 1000.0) * 1e9);

        matrix_records.push_back({
            exp_type, name, topology, m, n, nnz, density,
            rstats.min_nnz, rstats.max_nnz, rstats.avg_nnz, rstats.stddev_nnz,
            "CPU_12T", cpu12_ms, cpu12_gflops, cpu12_bw, cpu1_ms / cpu12_ms, 1.0, "PASSED", 0.0
        });

        // GPU Device Allocations
        DeviceCSRMatrix d_mat(csr);
        DeviceVector d_x(x_view);
        DeviceVector d_y(m);

        double gpu_scalar_ms = 0.0;

        // 3. Four CUDA Variants
        std::vector<std::pair<SpMVKernelVariant, std::string>> variants = {
            {SpMVKernelVariant::Scalar, "CUDA_Scalar"},
            {SpMVKernelVariant::Vector, "CUDA_Vector"},
            {SpMVKernelVariant::Adaptive, "CUDA_Adaptive"},
            {SpMVKernelVariant::Balanced, "CUDA_Balanced"}
        };

        for (const auto& [var_enum, var_name] : variants) {
            // Warmup
            for (int w = 0; w < 10; ++w) {
                d_mat.spmv(1.0, d_x, 0.0, d_y, var_enum);
            }

            cudaEvent_t start, stop;
            CUDA_CHECK(cudaEventCreate(&start));
            CUDA_CHECK(cudaEventCreate(&stop));

            CUDA_CHECK(cudaEventRecord(start));
            for (int r = 0; r < gpu_runs; ++r) {
                d_mat.spmv(1.0, d_x, 0.0, d_y, var_enum);
            }
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));

            float total_ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
            double gpu_ms = total_ms / gpu_runs;

            if (var_enum == SpMVKernelVariant::Scalar) {
                gpu_scalar_ms = gpu_ms;
            }

            // Verify numerical correctness against CPU
            d_y.copy_to_host(MutableVectorView(h_y_check.data(), m));
            double max_err = 0.0;
            for (index_t i = 0; i < m; ++i) {
                double diff = std::abs(h_y_check[i] - h_y_ref[i]);
                if (diff > max_err) max_err = diff;
            }
            std::string status = (max_err < 1e-11) ? "PASSED" : "FAILED";

            double gflops = (2.0 * nnz) / ((gpu_ms / 1000.0) * 1e9);
            double bw = bytes / ((gpu_ms / 1000.0) * 1e9);

            matrix_records.push_back({
                exp_type, name, topology, m, n, nnz, density,
                rstats.min_nnz, rstats.max_nnz, rstats.avg_nnz, rstats.stddev_nnz,
                var_name, gpu_ms, gflops, bw, cpu1_ms / gpu_ms, gpu_scalar_ms / gpu_ms, status, max_err
            });

            CUDA_CHECK(cudaEventDestroy(start));
            CUDA_CHECK(cudaEventDestroy(stop));
        }
    };

    // =========================================================================
    // 2. NNZ Scaling & Crossover Sweep
    // =========================================================================
    std::cout << "[2/4] Running NNZ Scaling & Crossover Sweep...\n";
    std::vector<index_t> nnz_dims = {500, 1000, 2000, 5000, 10000, 15000, 20000, 30000};
    for (index_t dim : nnz_dims) {
        double d = (dim <= 2000) ? 0.01 : 0.001; // Controlled density
        COOMatrix coo = MatrixGenerator::generate_random(dim, dim, d, -5.0, 5.0, 1000 + dim);
        CSRMatrix csr = coo.to_csr();
        std::string name = "Random_" + std::to_string(dim) + "x" + std::to_string(dim);
        evaluate_matrix("nnz_scaling", name, "random", csr);
    }

    // =========================================================================
    // 3. Matrix Sparsity / Density Sweep
    // =========================================================================
    std::cout << "[3/4] Running Sparsity & Density Sweep (Fixed 10k x 10k)...\n";
    std::vector<double> densities = {0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02};
    for (double d : densities) {
        COOMatrix coo = MatrixGenerator::generate_random(10'000, 10'000, d, -4.0, 4.0, 2000 + static_cast<int>(d * 100000));
        CSRMatrix csr = coo.to_csr();
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(4) << (d * 100.0);
        std::string name = "Density_" + ss.str() + "pct";
        evaluate_matrix("density_scaling", name, "random", csr);
    }

    // =========================================================================
    // 4. Matrix Structural Topologies & Real Netlib Sweep
    // =========================================================================
    std::cout << "[4/4] Running Matrix Structural Topologies & Netlib Sweep...\n";
    {
        COOMatrix coo = MatrixGenerator::generate_banded(20'000, 20'000, 15, 15, -3.0, 3.0, 3001);
        evaluate_matrix("structure_comparison", "Banded_20k", "banded", coo.to_csr());
    }
    {
        COOMatrix coo = MatrixGenerator::generate_block_diagonal(50, 200, 200, 0.03, 0.0005, -2.0, 2.0, 3002);
        evaluate_matrix("structure_comparison", "BlockDiag_10k", "block_diagonal", coo.to_csr());
    }
    {
        COOMatrix coo = MatrixGenerator::generate_staircase(100, 100, 100, 0.03, -2.0, 2.0, 3003);
        evaluate_matrix("structure_comparison", "Staircase_10k", "staircase", coo.to_csr());
    }
    {
        COOMatrix coo = MatrixGenerator::generate_irregular(10'000, 10'000, 200'000, 0.05, 0.50, -4.0, 4.0, 3004);
        evaluate_matrix("structure_comparison", "IrregularHub_10k", "irregular", coo.to_csr());
    }

    // Netlib models
    std::vector<std::string> netlib_instances = {"beaconfd.mps", "bandm.mps", "afiro.mps"};
    for (const auto& f : netlib_instances) {
        std::string p = find_netlib_file(f);
        if (!p.empty()) {
            model::LinearProgram lp;
            if (model::MPSParser::parse_file(p, lp).is_ok()) {
                evaluate_matrix("netlib", "Netlib_" + f.substr(0, f.find('.')), "real_netlib", lp.to_csr());
            }
        }
    }

    // =========================================================================
    // Export Results to JSON & CSV
    // =========================================================================
    std::string json_path = out_dir + "/benchmark_results.json";
    std::string csv_path = out_dir + "/benchmark_results.csv";
    std::string red_json_path = out_dir + "/reductions_results.json";
    std::string red_csv_path = out_dir + "/reductions_results.csv";

    // 1. Matrix Results JSON
    {
        std::ofstream jf(json_path);
        jf << "{\n";
        jf << "  \"hardware\": {\n";
        jf << "    \"cpu\": \"Intel Core i5-13420H\",\n";
        jf << "    \"cpu_threads\": " << cpu_max_threads << ",\n";
        jf << "    \"gpu_name\": \"" << dev.name << "\",\n";
        jf << "    \"gpu_compute_capability\": \"" << dev.major << "." << dev.minor << "\",\n";
        jf << "    \"gpu_sm_count\": " << dev.multi_processor_count << ",\n";
        jf << "    \"gpu_vram_mb\": " << (dev.total_memory_bytes / (1024 * 1024)) << "\n";
        jf << "  },\n";
        jf << "  \"experiments\": [\n";
        for (size_t i = 0; i < matrix_records.size(); ++i) {
            const auto& r = matrix_records[i];
            jf << "    {\n";
            jf << "      \"experiment_type\": \"" << r.experiment_type << "\",\n";
            jf << "      \"matrix_name\": \"" << r.matrix_name << "\",\n";
            jf << "      \"topology\": \"" << r.topology << "\",\n";
            jf << "      \"rows\": " << r.rows << ",\n";
            jf << "      \"cols\": " << r.cols << ",\n";
            jf << "      \"nnz\": " << r.nnz << ",\n";
            jf << "      \"density_pct\": " << r.density_pct << ",\n";
            jf << "      \"min_row_nnz\": " << r.min_row_nnz << ",\n";
            jf << "      \"max_row_nnz\": " << r.max_row_nnz << ",\n";
            jf << "      \"avg_row_nnz\": " << r.avg_row_nnz << ",\n";
            jf << "      \"stddev_row_nnz\": " << r.stddev_row_nnz << ",\n";
            jf << "      \"variant\": \"" << r.variant << "\",\n";
            jf << "      \"runtime_ms\": " << r.runtime_ms << ",\n";
            jf << "      \"gflops\": " << r.gflops << ",\n";
            jf << "      \"bandwidth_gbs\": " << r.bandwidth_gbs << ",\n";
            jf << "      \"speedup_vs_cpu_1t\": " << r.speedup_vs_cpu_1t << ",\n";
            jf << "      \"speedup_vs_gpu_scalar\": " << r.speedup_vs_gpu_scalar << ",\n";
            jf << "      \"verification_status\": \"" << r.verification_status << "\",\n";
            jf << "      \"max_abs_error\": " << r.max_abs_error << "\n";
            jf << "    }" << (i + 1 < matrix_records.size() ? "," : "") << "\n";
        }
        jf << "  ]\n";
        jf << "}\n";
    }

    // 2. Matrix Results CSV
    {
        std::ofstream cf(csv_path);
        cf << "experiment_type,matrix_name,topology,rows,cols,nnz,density_pct,"
           << "min_row_nnz,max_row_nnz,avg_row_nnz,stddev_row_nnz,variant,"
           << "runtime_ms,gflops,bandwidth_gbs,speedup_vs_cpu_1t,speedup_vs_gpu_scalar,verification_status,max_abs_error\n";
        for (const auto& r : matrix_records) {
            cf << r.experiment_type << ","
               << r.matrix_name << ","
               << r.topology << ","
               << r.rows << ","
               << r.cols << ","
               << r.nnz << ","
               << r.density_pct << ","
               << r.min_row_nnz << ","
               << r.max_row_nnz << ","
               << r.avg_row_nnz << ","
               << r.stddev_row_nnz << ","
               << r.variant << ","
               << r.runtime_ms << ","
               << r.gflops << ","
               << r.bandwidth_gbs << ","
               << r.speedup_vs_cpu_1t << ","
               << r.speedup_vs_gpu_scalar << ","
               << r.verification_status << ","
               << r.max_abs_error << "\n";
        }
    }

    // 3. Reductions Results JSON
    {
        std::ofstream rf(red_json_path);
        rf << "[\n";
        for (size_t i = 0; i < reduction_records.size(); ++i) {
            const auto& r = reduction_records[i];
            rf << "  {\n";
            rf << "    \"operation\": \"" << r.operation << "\",\n";
            rf << "    \"size\": " << r.size << ",\n";
            rf << "    \"data_mb\": " << r.data_mb << ",\n";
            rf << "    \"variant\": \"" << r.variant << "\",\n";
            rf << "    \"runtime_ms\": " << r.runtime_ms << ",\n";
            rf << "    \"gflops\": " << r.gflops << ",\n";
            rf << "    \"bandwidth_gbs\": " << r.bandwidth_gbs << ",\n";
            rf << "    \"speedup_vs_cpu_1t\": " << r.speedup_vs_cpu_1t << ",\n";
            rf << "    \"verification_status\": \"" << r.verification_status << "\"\n";
            rf << "  }" << (i + 1 < reduction_records.size() ? "," : "") << "\n";
        }
        rf << "]\n";
    }

    // 4. Reductions Results CSV
    {
        std::ofstream rcf(red_csv_path);
        rcf << "operation,size,data_mb,variant,runtime_ms,gflops,bandwidth_gbs,speedup_vs_cpu_1t,verification_status\n";
        for (const auto& r : reduction_records) {
            rcf << r.operation << ","
                << r.size << ","
                << r.data_mb << ","
                << r.variant << ","
                << r.runtime_ms << ","
                << r.gflops << ","
                << r.bandwidth_gbs << ","
                << r.speedup_vs_cpu_1t << ","
                << r.verification_status << "\n";
        }
    }

    std::cout << "\n========================================================================================\n";
    std::cout << " Benchmark Execution Complete!\n";
    std::cout << "   Matrix data points collected: " << matrix_records.size() << "\n";
    std::cout << "   Reduction data points collected: " << reduction_records.size() << "\n";
    std::cout << "   Exported JSON: " << json_path << "\n";
    std::cout << "   Exported CSV:  " << csv_path << "\n";
    std::cout << "   Exported Reductions JSON: " << red_json_path << "\n";
    std::cout << "   Exported Reductions CSV:  " << red_csv_path << "\n";
    std::cout << "========================================================================================\n";

    return 0;
}
