#include <pipepye/cuda/device_info.cuh>
#include <pipepye/cuda/cuda_check.cuh>
#include <pipepye/utils/timer.hpp>
#include <pipepye/utils/nvtx_markers.hpp>
#include "../cuda/kernels/axpy.cuh"

#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>

void run_cpu_daxpy(const double* x, double* y, double alpha, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        y[i] = alpha * x[i] + y[i];
    }
}

struct BenchResult {
    size_t n;
    double cpu_time_ms;
    double gpu_kernel_ms;
    double h2d_ms;
    double d2h_ms;
    double total_gpu_e2e_ms;
    double achieved_bandwidth_gbs;
    double speedup_kernel;
    double speedup_e2e;
};

BenchResult benchmark_size(size_t n, int warmup_runs = 3, int bench_runs = 10) {
    const double alpha = 3.14159265;
    const size_t bytes = n * sizeof(double);

    std::vector<double> h_x(n, 1.0);
    std::vector<double> h_y(n, 2.0);
    std::vector<double> h_y_cpu(n, 2.0);

    // 1. Measure CPU performance
    pipepye::utils::CPUTimer cpu_timer;
    cpu_timer.start();
    for (int r = 0; r < bench_runs; ++r) {
        run_cpu_daxpy(h_x.data(), h_y_cpu.data(), alpha, n);
    }
    cpu_timer.stop();
    double cpu_time_ms = cpu_timer.elapsed_milliseconds() / bench_runs;

    // 2. Allocate GPU memory
    double *d_x = nullptr, *d_y = nullptr;
    CUDA_CHECK(cudaMalloc(&d_x, bytes));
    CUDA_CHECK(cudaMalloc(&d_y, bytes));

    // Warmup GPU
    for (int w = 0; w < warmup_runs; ++w) {
        CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_y, h_y.data(), bytes, cudaMemcpyHostToDevice));
        pipepye::cuda::kernels::launch_daxpy(d_x, d_y, alpha, n);
        CUDA_SYNC_AND_CHECK();
        CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, bytes, cudaMemcpyDeviceToHost));
    }

    // Benchmark H2D
    pipepye::utils::CPUTimer h2d_timer;
    h2d_timer.start();
    for (int r = 0; r < bench_runs; ++r) {
        CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_y, h_y.data(), bytes, cudaMemcpyHostToDevice));
    }
    h2d_timer.stop();
    double h2d_ms = h2d_timer.elapsed_milliseconds() / bench_runs;

    // Benchmark Kernel
    pipepye::utils::CPUTimer kernel_timer;
    kernel_timer.start();
    for (int r = 0; r < bench_runs; ++r) {
        pipepye::cuda::kernels::launch_daxpy(d_x, d_y, alpha, n);
    }
    CUDA_SYNC_AND_CHECK();
    kernel_timer.stop();
    double gpu_kernel_ms = kernel_timer.elapsed_milliseconds() / bench_runs;

    // Benchmark D2H
    pipepye::utils::CPUTimer d2h_timer;
    d2h_timer.start();
    for (int r = 0; r < bench_runs; ++r) {
        CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, bytes, cudaMemcpyDeviceToHost));
    }
    d2h_timer.stop();
    double d2h_ms = d2h_timer.elapsed_milliseconds() / bench_runs;

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));

    double total_gpu_e2e_ms = h2d_ms + gpu_kernel_ms + d2h_ms;
    double total_bytes = 3.0 * bytes; // 2 reads (x, y) + 1 write (y)
    double achieved_bandwidth_gbs = (total_bytes / 1e9) / (gpu_kernel_ms / 1000.0);

    BenchResult res;
    res.n = n;
    res.cpu_time_ms = cpu_time_ms;
    res.gpu_kernel_ms = gpu_kernel_ms;
    res.h2d_ms = h2d_ms;
    res.d2h_ms = d2h_ms;
    res.total_gpu_e2e_ms = total_gpu_e2e_ms;
    res.achieved_bandwidth_gbs = achieved_bandwidth_gbs;
    res.speedup_kernel = cpu_time_ms / gpu_kernel_ms;
    res.speedup_e2e = cpu_time_ms / total_gpu_e2e_ms;

    return res;
}

int main() {
    std::cout << "========================================================================================\n";
    std::cout << " PipePye Micro-Benchmark: CPU vs CUDA DAXPY (y = alpha * x + y)\n";
    std::cout << "========================================================================================\n";

    int dev_count = pipepye::cuda::get_device_count();
    if (dev_count == 0) {
        std::cerr << "No CUDA device available for micro-benchmark.\n";
        return 1;
    }
    pipepye::cuda::DeviceProperties dev = pipepye::cuda::query_device(0);
    std::cout << "Device: " << dev.name << " (Compute " << dev.major << "." << dev.minor << ", "
              << std::fixed << std::setprecision(2) << dev.total_memory_gb() << " GiB VRAM)\n\n";

    std::vector<size_t> sizes = {
        10'000,          // 10K
        100'000,         // 100K
        1'000'000,       // 1M
        5'000'000,       // 5M
        10'000'000       // 10M
    };

    std::cout << std::setw(10) << "Vector N"
              << std::setw(12) << "Size (MB)"
              << std::setw(14) << "CPU (ms)"
              << std::setw(14) << "GPU Kern(ms)"
              << std::setw(14) << "GPU E2E(ms)"
              << std::setw(14) << "BW (GB/s)"
              << std::setw(16) << "Speedup(Kern)"
              << std::setw(14) << "Speedup(E2E)" << "\n";
    std::cout << std::string(108, '-') << "\n";

    for (size_t n : sizes) {
        auto res = benchmark_size(n);
        double size_mb = (3.0 * n * sizeof(double)) / (1024.0 * 1024.0);
        std::cout << std::setw(10) << n
                  << std::setw(12) << std::fixed << std::setprecision(2) << size_mb
                  << std::setw(14) << std::fixed << std::setprecision(4) << res.cpu_time_ms
                  << std::setw(14) << std::fixed << std::setprecision(4) << res.gpu_kernel_ms
                  << std::setw(14) << std::fixed << std::setprecision(4) << res.total_gpu_e2e_ms
                  << std::setw(14) << std::fixed << std::setprecision(2) << res.achieved_bandwidth_gbs
                  << std::setw(15) << std::fixed << std::setprecision(1) << res.speedup_kernel << "x"
                  << std::setw(13) << std::fixed << std::setprecision(2) << res.speedup_e2e << "x\n";
    }

    std::cout << std::string(108, '-') << "\n";
    std::cout << "Observation: Highlights CPU vs GPU crossover point, validating Phase 0 & 1 hardware mapping.\n";
    std::cout << "========================================================================================\n";
    return 0;
}
