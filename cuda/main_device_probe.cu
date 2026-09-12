#include <pipepye/cuda/device_info.cuh>
#include <pipepye/cuda/cuda_check.cuh>
#include <pipepye/utils/nvtx_markers.hpp>
#include <pipepye/utils/timer.hpp>
#include "kernels/axpy.cuh"

#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <cstdlib>

int main() {
    std::cout << "=================================================================\n";
    std::cout << " PipePye Solver: Phase 0 CUDA Toolchain & Hardware Probe\n";
    std::cout << "=================================================================\n";

    // 1. Device Detection & Diagnostics
    int dev_count = pipepye::cuda::get_device_count();
    if (dev_count == 0) {
        std::cerr << "[FATAL] No CUDA devices detected on this system.\n";
        return EXIT_FAILURE;
    }

    pipepye::cuda::DeviceProperties dev_info;
    try {
        dev_info = pipepye::cuda::query_device(0);
        pipepye::cuda::print_device_summary(dev_info);
    } catch (const pipepye::cuda::CudaException& e) {
        std::cerr << "[CUDA Exception during query]: " << e.what() << "\n";
        return EXIT_FAILURE;
    }

    if (dev_info.is_rtx_3050) {
        std::cout << "[SUCCESS] Confirmed target GPU: NVIDIA GeForce RTX 3050 Laptop GPU.\n";
    } else {
        std::cout << "[INFO] Running on compatible CUDA GPU: " << dev_info.name << "\n";
    }

    // 2. Problem Setup (Double precision vector AXPY: y = alpha * x + y)
    // Double precision is essential for simplex and linear programming KKT conditions.
    const size_t n = 4 * 1024 * 1024; // 4 Million elements (~32 MB per vector, 96 MB working set)
    const double alpha = 3.141592653589793;
    const size_t bytes = n * sizeof(double);

    std::cout << "\n[Benchmark Task] DAXPY (y = alpha * x + y)\n";
    std::cout << "  Vector dimension : " << n << " elements (" << bytes / (1024 * 1024) << " MiB each)\n";
    std::cout << "  Total Memory Set : " << (bytes * 3) / (1024 * 1024) << " MiB (2 reads + 1 write)\n";
    std::cout << "  Precision        : FP64 (IEEE 754 double precision)\n";

    pipepye::utils::CPUTimer total_timer;
    total_timer.start();

    // 3. Host Memory Allocation & Initialization
    std::vector<double> h_x(n);
    std::vector<double> h_y(n);
    std::vector<double> h_y_ref(n);

    {
        pipepye::utils::ScopedRange r("CPU_Initialization");
        for (size_t i = 0; i < n; ++i) {
            h_x[i] = static_cast<double>(i % 1000) * 0.001;
            h_y[i] = static_cast<double>((i + 500) % 1000) * 0.002;
            h_y_ref[i] = alpha * h_x[i] + h_y[i]; // CPU Ground Truth
        }
    }

    // 4. Device Memory Allocation with Error Checking
    double* d_x = nullptr;
    double* d_y = nullptr;
    try {
        CUDA_CHECK(cudaMalloc(&d_x, bytes));
        CUDA_CHECK(cudaMalloc(&d_y, bytes));
    } catch (const pipepye::cuda::CudaException& e) {
        std::cerr << "[CUDA Exception during cudaMalloc]: " << e.what() << "\n";
        return EXIT_FAILURE;
    }

    // 5. Host-to-Device Memory Transfer (annotated for Nsight Systems timeline)
    pipepye::utils::CPUTimer h2d_timer;
    h2d_timer.start();
    {
        pipepye::utils::ScopedRange r("Host_To_Device_Transfer");
        CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_y, h_y.data(), bytes, cudaMemcpyHostToDevice));
    }
    h2d_timer.stop();

    // 6. Kernel Launch & GPU Execution
    pipepye::utils::CPUTimer kernel_timer;
    kernel_timer.start();
    {
        pipepye::utils::ScopedRange r("Kernel_Execution");
        pipepye::cuda::kernels::launch_daxpy(d_x, d_y, alpha, n);
        CUDA_SYNC_AND_CHECK();
    }
    kernel_timer.stop();

    // 7. Device-to-Host Memory Transfer
    pipepye::utils::CPUTimer d2h_timer;
    d2h_timer.start();
    {
        pipepye::utils::ScopedRange r("Device_To_Host_Transfer");
        CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, bytes, cudaMemcpyDeviceToHost));
    }
    d2h_timer.stop();

    // 8. Device Memory Cleanup
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));

    // 9. CPU Verification
    bool passed = true;
    double max_abs_diff = 0.0;
    size_t error_count = 0;

    {
        pipepye::utils::ScopedRange r("CPU_Verification");
        for (size_t i = 0; i < n; ++i) {
            double diff = std::abs(h_y[i] - h_y_ref[i]);
            if (diff > max_abs_diff) {
                max_abs_diff = diff;
            }
            if (diff > 1e-12) {
                passed = false;
                if (error_count < 5) {
                    std::cerr << "Mismatch at [" << i << "]: GPU=" << h_y[i]
                              << ", CPU=" << h_y_ref[i] << ", diff=" << diff << "\n";
                }
                error_count++;
            }
        }
    }

    total_timer.stop();

    // 10. Performance & Throughput Reporting
    double kernel_time_ms = kernel_timer.elapsed_milliseconds();
    double h2d_time_ms = h2d_timer.elapsed_milliseconds();
    double d2h_time_ms = d2h_timer.elapsed_milliseconds();
    double total_time_ms = total_timer.elapsed_milliseconds();

    // Bandwidth: 2 reads (x, y) + 1 write (y) = 3 * bytes
    double total_data_gb = (3.0 * bytes) / 1e9;
    double achieved_bandwidth_gbs = total_data_gb / (kernel_time_ms / 1000.0);
    // FLOPs: 1 mult + 1 add per element = 2 * n operations
    double gflops = (2.0 * n) / (kernel_time_ms / 1000.0) / 1e9;

    std::cout << "\n-----------------------------------------------------------------\n";
    std::cout << " Execution Breakdown & Metrics:\n";
    std::cout << "-----------------------------------------------------------------\n";
    std::cout << "  Host -> Device Transfer    : " << std::fixed << std::setprecision(3) << h2d_time_ms << " ms\n";
    std::cout << "  GPU Kernel Execution       : " << kernel_time_ms << " ms\n";
    std::cout << "  Device -> Host Transfer    : " << d2h_time_ms << " ms\n";
    std::cout << "  End-to-End Total Time      : " << total_time_ms << " ms\n";
    std::cout << "  Achieved Kernel Bandwidth  : " << std::setprecision(2) << achieved_bandwidth_gbs << " GB/s\n";
    std::cout << "  Compute Throughput         : " << std::setprecision(2) << gflops << " GFLOPS\n";
    std::cout << "  Maximum Absolute Diff      : " << std::scientific << std::setprecision(3) << max_abs_diff << "\n";
    std::cout << "-----------------------------------------------------------------\n";

    if (passed) {
        std::cout << " [VERIFICATION PASSED] 100% of " << n
                  << " elements numerically identical to CPU ground truth (tol <= 1e-12).\n";
        std::cout << "=================================================================\n";
        return EXIT_SUCCESS;
    } else {
        std::cerr << " [VERIFICATION FAILED] " << error_count << " discrepancies detected!\n";
        std::cerr << "=================================================================\n";
        return EXIT_FAILURE;
    }
}
