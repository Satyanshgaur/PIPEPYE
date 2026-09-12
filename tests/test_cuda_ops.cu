#include <gtest/gtest.h>
#include <pipepye/cuda/cuda_check.cuh>
#include <pipepye/cuda/device_info.cuh>
#include "../cuda/kernels/axpy.cuh"

#include <vector>
#include <cmath>

TEST(CudaErrorHandlingTest, CatchesInvalidCallAndThrowsCudaException) {
    // Deliberately trigger an error with an out-of-range device ID
    EXPECT_THROW({
        CUDA_CHECK(cudaSetDevice(9999));
    }, pipepye::cuda::CudaException);

    try {
        CUDA_CHECK(cudaSetDevice(9999));
    } catch (const pipepye::cuda::CudaException& e) {
        EXPECT_EQ(e.error_code(), cudaErrorInvalidDevice);
        std::string msg = e.what();
        EXPECT_NE(msg.find("cudaErrorInvalidDevice"), std::string::npos);
        EXPECT_NE(msg.find("cudaSetDevice"), std::string::npos);
    }
    // Clear CUDA error state
    cudaGetLastError();
}

TEST(CudaDeviceTest, QueryDeviceCapabilities) {
    int dev_count = pipepye::cuda::get_device_count();
    if (dev_count == 0) {
        GTEST_SKIP() << "No physical CUDA GPU available in this test environment.";
    }

    pipepye::cuda::DeviceProperties dev = pipepye::cuda::query_device(0);
    EXPECT_EQ(dev.device_id, 0);
    EXPECT_FALSE(dev.name.empty());
    EXPECT_GE(dev.major, 5); // At least Maxwell or newer
    EXPECT_EQ(dev.warp_size, 32);
    EXPECT_GT(dev.total_memory_bytes, 0);
    EXPECT_GT(dev.multi_processor_count, 0);
    EXPECT_GE(dev.max_threads_per_block, 512);

    // If on target hardware, verify Ampere / RTX 3050 capability
    if (dev.is_rtx_3050) {
        EXPECT_TRUE(pipepye::cuda::is_rtx_3050_compatible(dev));
        EXPECT_EQ(dev.major, 8);
        EXPECT_EQ(dev.minor, 6);
    }
}

TEST(CudaKernelTest, DoublePrecisionAxpyNumericalVerification) {
    int dev_count = pipepye::cuda::get_device_count();
    if (dev_count == 0) {
        GTEST_SKIP() << "No physical CUDA GPU available in this test environment.";
    }

    const size_t n = 65536;
    const double alpha = 2.718281828459;
    const size_t bytes = n * sizeof(double);

    std::vector<double> h_x(n);
    std::vector<double> h_y(n);
    std::vector<double> h_ref(n);

    for (size_t i = 0; i < n; ++i) {
        h_x[i] = static_cast<double>(i) * 0.01;
        h_y[i] = static_cast<double>(n - i) * 0.02;
        h_ref[i] = alpha * h_x[i] + h_y[i];
    }

    double *d_x = nullptr, *d_y = nullptr;
    CUDA_CHECK(cudaMalloc(&d_x, bytes));
    CUDA_CHECK(cudaMalloc(&d_y, bytes));

    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_y.data(), bytes, cudaMemcpyHostToDevice));

    pipepye::cuda::kernels::launch_daxpy(d_x, d_y, alpha, n);
    CUDA_SYNC_AND_CHECK();

    CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));

    for (size_t i = 0; i < n; ++i) {
        double diff = std::abs(h_y[i] - h_ref[i]);
        ASSERT_LE(diff, 1e-12) << "Mismatch at index " << i;
    }
}

TEST(CudaKernelTest, SinglePrecisionAxpyNumericalVerification) {
    int dev_count = pipepye::cuda::get_device_count();
    if (dev_count == 0) {
        GTEST_SKIP() << "No physical CUDA GPU available in this test environment.";
    }

    const size_t n = 65536;
    const float alpha = 1.41421356f;
    const size_t bytes = n * sizeof(float);

    std::vector<float> h_x(n);
    std::vector<float> h_y(n);
    std::vector<float> h_ref(n);

    for (size_t i = 0; i < n; ++i) {
        h_x[i] = static_cast<float>(i) * 0.1f;
        h_y[i] = static_cast<float>(n - i) * 0.2f;
        h_ref[i] = alpha * h_x[i] + h_y[i];
    }

    float *d_x = nullptr, *d_y = nullptr;
    CUDA_CHECK(cudaMalloc(&d_x, bytes));
    CUDA_CHECK(cudaMalloc(&d_y, bytes));

    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_y.data(), bytes, cudaMemcpyHostToDevice));

    pipepye::cuda::kernels::launch_saxpy(d_x, d_y, alpha, n);
    CUDA_SYNC_AND_CHECK();

    CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));

    for (size_t i = 0; i < n; ++i) {
        float diff = std::abs(h_y[i] - h_ref[i]);
        float rel_err = diff / (std::abs(h_ref[i]) + 1.0f);
        ASSERT_LE(rel_err, 1e-5f) << "Single precision mismatch at index " << i
                                  << " (diff=" << diff << ", rel=" << rel_err << ")";
    }
}
