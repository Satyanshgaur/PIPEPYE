#include <gtest/gtest.h>
#include <pipepye/cuda/cuda_check.cuh>
#include <pipepye/cuda/device_info.cuh>
#include <pipepye/cuda/spmv.cuh>
#include <pipepye/cuda/reductions.cuh>
#include <pipepye/sparse/matrix_generator.hpp>
#include <pipepye/sparse/cpu_ops.hpp>
#include <pipepye/sparse/dense_matrix.hpp>
#include <pipepye/model/mps_parser.hpp>

#include <vector>
#include <cmath>
#include <random>
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

// Numerical verification helper comparing GPU SpMV output against CPU reference
void verify_spmv_variant_against_cpu(const CSRMatrix& host_csr, SpMVKernelVariant variant,
                                     scalar_t alpha = 1.0, scalar_t beta = 0.0,
                                     double tol = 1e-12) {
    index_t m = host_csr.num_rows();
    index_t n = host_csr.num_cols();

    std::vector<scalar_t> h_x(n);
    std::vector<scalar_t> h_y_cpu(m);
    std::vector<scalar_t> h_y_gpu(m);

    // Populate deterministic test vectors
    for (index_t j = 0; j < n; ++j) {
        h_x[j] = std::sin(static_cast<double>(j + 1) * 0.1);
    }
    for (index_t i = 0; i < m; ++i) {
        h_y_cpu[i] = std::cos(static_cast<double>(i + 1) * 0.1);
        h_y_gpu[i] = h_y_cpu[i];
    }

    // 1. Compute CPU reference result
    cpu_ops::spmv_csr(alpha, host_csr,
                      ConstVectorView(h_x.data(), n),
                      beta,
                      MutableVectorView(h_y_cpu.data(), m));

    // 2. Upload to GPU and execute kernel variant
    DeviceCSRMatrix d_mat(host_csr);
    DeviceVector d_x(ConstVectorView(h_x.data(), n));
    DeviceVector d_y(ConstVectorView(h_y_gpu.data(), m));

    d_mat.spmv(alpha, d_x, beta, d_y, variant);
    CUDA_CHECK(cudaDeviceSynchronize());

    d_y.copy_to_host(MutableVectorView(h_y_gpu.data(), m));

    // 3. Compare elementwise max absolute difference
    double max_err = 0.0;
    for (index_t i = 0; i < m; ++i) {
        double err = std::abs(h_y_gpu[i] - h_y_cpu[i]);
        if (err > max_err) {
            max_err = err;
        }
    }

    EXPECT_LE(max_err, tol) << "Variant " << to_string(variant)
                            << " failed parity check against CPU! Max err: " << max_err;
}

} // namespace

// =============================================================================
// CUDA Reductions Tests
// =============================================================================

TEST(CudaReductionsTest, DotProductParityAgainstCPU) {
    int dev_count = get_device_count();
    if (dev_count == 0) GTEST_SKIP() << "No GPU available";

    for (size_t n : {100UL, 1000UL, 65536UL, 200000UL}) {
        std::vector<scalar_t> h_x(n);
        std::vector<scalar_t> h_y(n);
        for (size_t i = 0; i < n; ++i) {
            h_x[i] = std::sin(static_cast<double>(i));
            h_y[i] = std::cos(static_cast<double>(i) * 0.5);
        }

        scalar_t ref_dot = cpu_ops::dot(ConstVectorView(h_x.data(), n), ConstVectorView(h_y.data(), n));

        DeviceVector d_x(ConstVectorView(h_x.data(), n));
        DeviceVector d_y(ConstVectorView(h_y.data(), n));

        scalar_t gpu_dot = cuda_dot(d_x.data(), d_y.data(), n);

        double rel_err = std::abs(gpu_dot - ref_dot) / (std::abs(ref_dot) + 1e-15);
        EXPECT_LT(rel_err, 1e-12) << "Dot product mismatch for N=" << n;
    }
}

TEST(CudaReductionsTest, NormsAndSumParityAgainstCPU) {
    int dev_count = get_device_count();
    if (dev_count == 0) GTEST_SKIP() << "No GPU available";

    const size_t n = 131072;
    std::vector<scalar_t> h_x(n);
    for (size_t i = 0; i < n; ++i) {
        h_x[i] = (i % 2 == 0) ? -static_cast<double>(i) * 0.001 : static_cast<double>(i) * 0.0015;
    }
    h_x[n / 2] = -999999.5; // Distinct maximum absolute element

    ConstVectorView view(h_x.data(), n);
    scalar_t ref_norm2 = view.norm_2();
    scalar_t ref_norm1 = view.norm_1();
    scalar_t ref_norm_inf = view.norm_inf();
    scalar_t ref_sum = 0.0;
    for (scalar_t v : h_x) ref_sum += v;

    DeviceVector d_x(view);

    scalar_t gpu_norm2 = cuda_norm_2(d_x.data(), n);
    scalar_t gpu_norm1 = cuda_norm_1(d_x.data(), n);
    scalar_t gpu_norm_inf = cuda_norm_inf(d_x.data(), n);
    scalar_t gpu_sum = cuda_sum(d_x.data(), n);

    EXPECT_LT(std::abs(gpu_norm2 - ref_norm2) / ref_norm2, 1e-12);
    EXPECT_LT(std::abs(gpu_norm1 - ref_norm1) / ref_norm1, 1e-12);
    EXPECT_DOUBLE_EQ(gpu_norm_inf, ref_norm_inf);
    EXPECT_LT(std::abs(gpu_sum - ref_sum) / (std::abs(ref_sum) + 1.0), 1e-12);
}

TEST(CudaReductionsTest, BoundaryDimensions) {
    int dev_count = get_device_count();
    if (dev_count == 0) GTEST_SKIP() << "No GPU available";

    // Test small and boundary vector sizes
    for (size_t n : {1UL, 31UL, 32UL, 33UL, 255UL, 256UL, 257UL}) {
        std::vector<scalar_t> h_x(n, 2.5);
        ConstVectorView view(h_x.data(), n);
        DeviceVector d_x(view);

        EXPECT_NEAR(cuda_norm_inf(d_x.data(), n), 2.5, 1e-12);
        EXPECT_NEAR(cuda_sum(d_x.data(), n), 2.5 * static_cast<double>(n), 1e-10);
    }
}

// =============================================================================
// CUDA SpMV Variants & CPU Parity Verification Tests
// =============================================================================

TEST(CudaSpMVVerificationTest, RandomSparseMatrixParityAllVariants) {
    int dev_count = get_device_count();
    if (dev_count == 0) GTEST_SKIP() << "No GPU available";

    COOMatrix coo = MatrixGenerator::generate_random(500, 400, 0.02, -5.0, 5.0, 42);
    CSRMatrix csr = coo.to_csr();

    for (auto variant : {SpMVKernelVariant::Scalar,
                         SpMVKernelVariant::Vector,
                         SpMVKernelVariant::Adaptive,
                         SpMVKernelVariant::Balanced}) {
        verify_spmv_variant_against_cpu(csr, variant, 1.0, 0.0);
        // Also test alpha and beta scaling
        verify_spmv_variant_against_cpu(csr, variant, 2.5, -1.5);
    }
}

TEST(CudaSpMVVerificationTest, BandedMatrixParityAllVariants) {
    int dev_count = get_device_count();
    if (dev_count == 0) GTEST_SKIP() << "No GPU available";

    COOMatrix coo = MatrixGenerator::generate_banded(1000, 1000, 10, 10, -2.0, 3.0, 43);
    CSRMatrix csr = coo.to_csr();

    for (auto variant : {SpMVKernelVariant::Scalar,
                         SpMVKernelVariant::Vector,
                         SpMVKernelVariant::Adaptive,
                         SpMVKernelVariant::Balanced}) {
        verify_spmv_variant_against_cpu(csr, variant, 1.0, 0.0);
    }
}

TEST(CudaSpMVVerificationTest, BlockDiagonalMatrixParityAllVariants) {
    int dev_count = get_device_count();
    if (dev_count == 0) GTEST_SKIP() << "No GPU available";

    COOMatrix coo = MatrixGenerator::generate_block_diagonal(20, 50, 50, 0.1, 0.005, -3.0, 3.0, 44);
    CSRMatrix csr = coo.to_csr();

    for (auto variant : {SpMVKernelVariant::Scalar,
                         SpMVKernelVariant::Vector,
                         SpMVKernelVariant::Adaptive,
                         SpMVKernelVariant::Balanced}) {
        verify_spmv_variant_against_cpu(csr, variant, 1.0, 0.0);
    }
}

TEST(CudaSpMVVerificationTest, StaircaseMatrixParityAllVariants) {
    int dev_count = get_device_count();
    if (dev_count == 0) GTEST_SKIP() << "No GPU available";

    COOMatrix coo = MatrixGenerator::generate_staircase(25, 40, 40, 0.05, -4.0, 4.0, 45);
    CSRMatrix csr = coo.to_csr();

    for (auto variant : {SpMVKernelVariant::Scalar,
                         SpMVKernelVariant::Vector,
                         SpMVKernelVariant::Adaptive,
                         SpMVKernelVariant::Balanced}) {
        verify_spmv_variant_against_cpu(csr, variant, 1.0, 0.0);
    }
}

TEST(CudaSpMVVerificationTest, IrregularHubMatrixParityAllVariants) {
    int dev_count = get_device_count();
    if (dev_count == 0) GTEST_SKIP() << "No GPU available";

    // 5% hub rows holding 50% nonzeros
    COOMatrix coo = MatrixGenerator::generate_irregular(1000, 1000, 20000, 0.05, 0.50, -5.0, 5.0, 46);
    CSRMatrix csr = coo.to_csr();

    for (auto variant : {SpMVKernelVariant::Scalar,
                         SpMVKernelVariant::Vector,
                         SpMVKernelVariant::Adaptive,
                         SpMVKernelVariant::Balanced}) {
        verify_spmv_variant_against_cpu(csr, variant, 1.0, 0.0);
        verify_spmv_variant_against_cpu(csr, variant, 1.5, 0.8);
    }
}

TEST(CudaSpMVVerificationTest, NetlibLPModelsParityAllVariants) {
    int dev_count = get_device_count();
    if (dev_count == 0) GTEST_SKIP() << "No GPU available";

    std::vector<std::string> netlib_files = {"beaconfd.mps", "bandm.mps", "afiro.mps"};

    for (const auto& file : netlib_files) {
        std::string path = find_netlib_file(file);
        if (path.empty()) continue;

        model::LinearProgram lp;
        ASSERT_TRUE(model::MPSParser::parse_file(path, lp).is_ok());
        CSRMatrix csr = lp.to_csr();

        for (auto variant : {SpMVKernelVariant::Scalar,
                             SpMVKernelVariant::Vector,
                             SpMVKernelVariant::Adaptive,
                             SpMVKernelVariant::Balanced}) {
            verify_spmv_variant_against_cpu(csr, variant, 1.0, 0.0);
        }
    }
}

TEST(CudaSpMVVerificationTest, EmptyMatrixAndEmptyRowsEdgeCases) {
    int dev_count = get_device_count();
    if (dev_count == 0) GTEST_SKIP() << "No GPU available";

    // 1. Matrix with alternating completely empty rows
    COOMatrix coo(6, 6);
    coo.add_entry(0, 1, 3.0);
    coo.add_entry(2, 4, -2.0);
    coo.add_entry(4, 0, 1.5);
    // Rows 1, 3, 5 are completely empty
    CSRMatrix csr = coo.to_csr();

    for (auto variant : {SpMVKernelVariant::Scalar,
                         SpMVKernelVariant::Vector,
                         SpMVKernelVariant::Adaptive,
                         SpMVKernelVariant::Balanced}) {
        verify_spmv_variant_against_cpu(csr, variant, 1.0, 0.0);
    }
}
