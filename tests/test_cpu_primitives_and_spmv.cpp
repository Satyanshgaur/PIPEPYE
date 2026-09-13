#include <gtest/gtest.h>
#include <pipepye/sparse/vector.hpp>
#include <pipepye/sparse/cpu_ops.hpp>
#include <pipepye/sparse/dense_matrix.hpp>
#include <pipepye/sparse/csr_matrix.hpp>
#include <pipepye/sparse/csc_matrix.hpp>
#include <pipepye/sparse/coo_matrix.hpp>
#include <cmath>

using namespace pipepye::sparse;
using pipepye::index_t;
using pipepye::scalar_t;

// =============================================================================
// CPU Vector Primitives
// =============================================================================

TEST(CpuVectorPrimitivesTest, AxpyAndAxpby) {
    Vector x = {1.0, -2.0, 3.0, -4.0};
    Vector y = {10.0, 20.0, 30.0, 40.0};

    // y = 2.0 * x + y -> [12.0, 16.0, 36.0, 32.0]
    cpu_ops::axpy(2.0, x.view(), y.view());
    EXPECT_DOUBLE_EQ(y[0], 12.0);
    EXPECT_DOUBLE_EQ(y[1], 16.0);
    EXPECT_DOUBLE_EQ(y[2], 36.0);
    EXPECT_DOUBLE_EQ(y[3], 32.0);

    // axpby: y = -1.0 * x + 0.5 * y
    // y[0] = -1*(1) + 0.5*(12) = -1 + 6 = 5.0
    // y[1] = -1*(-2) + 0.5*(16) = 2 + 8 = 10.0
    // y[2] = -1*(3) + 0.5*(36) = -3 + 18 = 15.0
    // y[3] = -1*(-4) + 0.5*(32) = 4 + 16 = 20.0
    cpu_ops::axpby(-1.0, x.view(), 0.5, y.view());
    EXPECT_DOUBLE_EQ(y[0], 5.0);
    EXPECT_DOUBLE_EQ(y[1], 10.0);
    EXPECT_DOUBLE_EQ(y[2], 15.0);
    EXPECT_DOUBLE_EQ(y[3], 20.0);

    // Dimension mismatch checks
    Vector short_vec = {1.0, 2.0};
    EXPECT_THROW(cpu_ops::axpy(1.0, short_vec.view(), y.view()), std::invalid_argument);
    EXPECT_THROW(cpu_ops::axpby(1.0, short_vec.view(), 1.0, y.view()), std::invalid_argument);
}

TEST(CpuVectorPrimitivesTest, DotProductAndNorms) {
    Vector x = {3.0, -4.0, 0.0, 12.0};
    Vector y = {-2.0, 5.0, 10.0, 1.0};

    // Dot product: 3*(-2) + (-4)*5 + 0*10 + 12*1 = -6 - 20 + 0 + 12 = -14
    EXPECT_DOUBLE_EQ(cpu_ops::dot(x.view(), y.view()), -14.0);

    // L1 norm: |3| + |-4| + |0| + |12| = 19
    EXPECT_DOUBLE_EQ(cpu_ops::norm_1(x.view()), 19.0);

    // L2 squared norm: 9 + 16 + 0 + 144 = 169
    EXPECT_DOUBLE_EQ(cpu_ops::norm_2_sq(x.view()), 169.0);

    // L2 norm: sqrt(169) = 13.0
    EXPECT_DOUBLE_EQ(cpu_ops::norm_2(x.view()), 13.0);

    // L_infinity norm: max(|3|, |-4|, |0|, |12|) = 12.0
    EXPECT_DOUBLE_EQ(cpu_ops::norm_inf(x.view()), 12.0);
}

TEST(CpuVectorPrimitivesTest, Reductions) {
    Vector v = {5.0, -3.0, 8.0, 2.0, -7.0, 4.0};

    // Sum: 5 - 3 + 8 + 2 - 7 + 4 = 9
    EXPECT_DOUBLE_EQ(cpu_ops::sum(v.view()), 9.0);

    // Mean: 9 / 6 = 1.5
    EXPECT_DOUBLE_EQ(cpu_ops::mean(v.view()), 1.5);

    // Min & Argmin: -7.0 at index 4
    EXPECT_DOUBLE_EQ(cpu_ops::min(v.view()), -7.0);
    EXPECT_EQ(cpu_ops::argmin(v.view()), 4);

    // Max & Argmax: 8.0 at index 2
    EXPECT_DOUBLE_EQ(cpu_ops::max(v.view()), 8.0);
    EXPECT_EQ(cpu_ops::argmax(v.view()), 2);

    // Empty vector exceptions
    Vector empty_vec;
    EXPECT_THROW((void)cpu_ops::min(empty_vec.view()), std::runtime_error);
    EXPECT_THROW((void)cpu_ops::max(empty_vec.view()), std::runtime_error);
    EXPECT_THROW((void)cpu_ops::argmin(empty_vec.view()), std::runtime_error);
    EXPECT_THROW((void)cpu_ops::argmax(empty_vec.view()), std::runtime_error);
    EXPECT_THROW((void)cpu_ops::mean(empty_vec.view()), std::runtime_error);
}

TEST(CpuVectorPrimitivesTest, CopyFillScaleAndSetZero) {
    Vector src = {1.5, -2.5, 3.5};
    Vector dst(3, 0.0);

    cpu_ops::copy(src.view(), dst.view());
    EXPECT_DOUBLE_EQ(dst[0], 1.5);
    EXPECT_DOUBLE_EQ(dst[1], -2.5);
    EXPECT_DOUBLE_EQ(dst[2], 3.5);

    cpu_ops::scale(2.0, dst.view());
    EXPECT_DOUBLE_EQ(dst[0], 3.0);
    EXPECT_DOUBLE_EQ(dst[1], -5.0);
    EXPECT_DOUBLE_EQ(dst[2], 7.0);

    cpu_ops::fill(dst.view(), 42.0);
    EXPECT_DOUBLE_EQ(dst[0], 42.0);
    EXPECT_DOUBLE_EQ(dst[1], 42.0);
    EXPECT_DOUBLE_EQ(dst[2], 42.0);

    cpu_ops::set_zero(dst.view());
    EXPECT_DOUBLE_EQ(dst[0], 0.0);
    EXPECT_DOUBLE_EQ(dst[1], 0.0);
    EXPECT_DOUBLE_EQ(dst[2], 0.0);
}

TEST(CpuVectorPrimitivesTest, BoxProjectionAndHadamard) {
    Vector x = {-5.0, 2.0, 10.0, 7.0};
    Vector lower = {0.0, 0.0, 0.0, 0.0};
    Vector upper = {4.0, 4.0, 4.0, 4.0};

    cpu_ops::project_box(x.view(), lower.view(), upper.view());
    EXPECT_DOUBLE_EQ(x[0], 0.0); // clamped from -5 to 0
    EXPECT_DOUBLE_EQ(x[1], 2.0); // unchanged
    EXPECT_DOUBLE_EQ(x[2], 4.0); // clamped from 10 to 4
    EXPECT_DOUBLE_EQ(x[3], 4.0); // clamped from 7 to 4

    Vector a = {2.0, -3.0, 4.0};
    Vector b = {5.0, 2.0, -1.5};
    Vector out(3, 0.0);

    cpu_ops::hadamard(a.view(), b.view(), out.view());
    EXPECT_DOUBLE_EQ(out[0], 10.0);
    EXPECT_DOUBLE_EQ(out[1], -6.0);
    EXPECT_DOUBLE_EQ(out[2], -6.0);

    // Distance metrics
    EXPECT_DOUBLE_EQ(cpu_ops::abs_diff_inf(a.view(), b.view()), 5.5); // max(|2-5|, |-3-2|, |4 - (-1.5)|) = 5.5
}

// =============================================================================
// Dense Matrix Reference Oracle Tests
// =============================================================================

TEST(DenseMatrixOracleTest, MatrixMultiplicationAndGEMV) {
    // A: 2 x 3
    // [ 1, 2, 3 ]
    // [ 4, 5, 6 ]
    DenseMatrix A(2, 3, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});

    // B: 3 x 2
    // [ 7,  8 ]
    // [ 9,  1 ]
    // [ 2,  3 ]
    DenseMatrix B(3, 2, {7.0, 8.0, 9.0, 1.0, 2.0, 3.0});

    // C = A * B = 2 x 2
    // C[0,0] = 1*7 + 2*9 + 3*2 = 7 + 18 + 6 = 31
    // C[0,1] = 1*8 + 2*1 + 3*3 = 8 + 2 + 9 = 19
    // C[1,0] = 4*7 + 5*9 + 6*2 = 28 + 45 + 12 = 85
    // C[1,1] = 4*8 + 5*1 + 6*3 = 32 + 5 + 18 = 55
    DenseMatrix C = A.matmul(B);
    ASSERT_EQ(C.num_rows(), 2);
    ASSERT_EQ(C.num_cols(), 2);
    EXPECT_DOUBLE_EQ(C(0, 0), 31.0);
    EXPECT_DOUBLE_EQ(C(0, 1), 19.0);
    EXPECT_DOUBLE_EQ(C(1, 0), 85.0);
    EXPECT_DOUBLE_EQ(C(1, 1), 55.0);

    // GEMV: y = 2.0 * A * x + 1.0 * y
    Vector x = {1.0, -1.0, 2.0};
    // A * x = [ 1*(1) + 2*(-1) + 3*(2), 4*(1) + 5*(-1) + 6*(2) ] = [ 5.0, 11.0 ]
    Vector y = {10.0, 20.0};
    // y = 2.0 * [5, 11] + 1.0 * [10, 20] = [20.0, 42.0]
    A.gemv(2.0, x.view(), 1.0, y.view());
    EXPECT_DOUBLE_EQ(y[0], 20.0);
    EXPECT_DOUBLE_EQ(y[1], 42.0);

    // GEMV Transpose: y = A^T * x
    Vector x_t = {2.0, 3.0};
    Vector y_t(3, 0.0);
    // A^T * x_t = [ 1*2 + 4*3, 2*2 + 5*3, 3*2 + 6*3 ] = [ 14.0, 19.0, 24.0 ]
    A.gemv_transpose(1.0, x_t.view(), 0.0, y_t.view());
    EXPECT_DOUBLE_EQ(y_t[0], 14.0);
    EXPECT_DOUBLE_EQ(y_t[1], 19.0);
    EXPECT_DOUBLE_EQ(y_t[2], 24.0);
}

// =============================================================================
// SpMV Correctness Against Dense Reference Oracle
// =============================================================================

TEST(SpMVVerificationTest, CompareCSRSpMVAgainstDenseOracle) {
    // Generate a diverse random test matrix (12 x 8)
    DenseMatrix dense_A = DenseMatrix::random(12, 8, -5.0, 5.0, 12345);

    // Zero out ~70% of entries to create a realistic sparse structure
    for (index_t i = 0; i < 12; ++i) {
        for (index_t j = 0; j < 8; ++j) {
            if ((i * 3 + j * 7) % 3 != 0) {
                dense_A(i, j) = 0.0;
            }
        }
    }

    CSRMatrix csr_A = dense_A.to_csr();
    CSCMatrix csc_A = dense_A.to_csc();

    // Verify NNZ count matches
    EXPECT_EQ(csr_A.num_nonzeros(), dense_A.num_nonzeros());
    EXPECT_EQ(csc_A.num_nonzeros(), dense_A.num_nonzeros());

    // Generate random input vector x
    Vector x = {1.2, -3.4, 0.5, -2.1, 4.3, -1.8, 0.0, 2.7};
    Vector y_dense(12, 0.0);
    Vector y_csr(12, 0.0);
    Vector y_csc(12, 0.0);

    // 1. Basic y = A * x (alpha=1.0, beta=0.0)
    dense_A.gemv(1.0, x.view(), 0.0, y_dense.view());
    cpu_ops::spmv_csr(1.0, csr_A, x.view(), 0.0, y_csr.view());
    cpu_ops::spmv_csc(1.0, csc_A, x.view(), 0.0, y_csc.view());

    for (index_t i = 0; i < 12; ++i) {
        EXPECT_NEAR(y_csr[i], y_dense[i], 1e-13);
        EXPECT_NEAR(y_csc[i], y_dense[i], 1e-13);
    }

    // 2. Scaled accumulation: y = 2.5 * A * x - 1.5 * y
    dense_A.gemv(2.5, x.view(), -1.5, y_dense.view());
    cpu_ops::spmv_csr(2.5, csr_A, x.view(), -1.5, y_csr.view());
    cpu_ops::spmv_csc(2.5, csc_A, x.view(), -1.5, y_csc.view());

    for (index_t i = 0; i < 12; ++i) {
        EXPECT_NEAR(y_csr[i], y_dense[i], 1e-13);
        EXPECT_NEAR(y_csc[i], y_dense[i], 1e-13);
    }
}

// =============================================================================
// SpMVᵀ Correctness Against Dense Reference Oracle
// =============================================================================

TEST(SpMVTransposeVerificationTest, CompareSpMVTransposeAgainstDenseOracle) {
    // Generate a non-square matrix (7 x 11)
    DenseMatrix dense_A = DenseMatrix::random(7, 11, -10.0, 10.0, 54321);

    // Sparsify
    for (index_t i = 0; i < 7; ++i) {
        for (index_t j = 0; j < 11; ++j) {
            if ((i + j) % 2 != 0) {
                dense_A(i, j) = 0.0;
            }
        }
    }

    CSRMatrix csr_A = dense_A.to_csr();
    CSCMatrix csc_A = dense_A.to_csc();

    Vector x = {-2.1, 3.5, 0.0, 1.8, -4.2, 5.1, -1.0}; // Size 7 (num_rows)
    Vector y_dense(11, 0.0);
    Vector y_csr(11, 0.0);
    Vector y_csc(11, 0.0);

    // 1. Basic y = A^T * x (alpha=1.0, beta=0.0)
    dense_A.gemv_transpose(1.0, x.view(), 0.0, y_dense.view());
    cpu_ops::spmv_transpose_csr(1.0, csr_A, x.view(), 0.0, y_csr.view());
    cpu_ops::spmv_transpose_csc(1.0, csc_A, x.view(), 0.0, y_csc.view());

    for (index_t j = 0; j < 11; ++j) {
        EXPECT_NEAR(y_csr[j], y_dense[j], 1e-13);
        EXPECT_NEAR(y_csc[j], y_dense[j], 1e-13);
        EXPECT_NEAR(y_csr[j], y_csc[j], 1e-14);
    }

    // 2. Scaled accumulation: y = -3.2 * A^T * x + 2.0 * y
    dense_A.gemv_transpose(-3.2, x.view(), 2.0, y_dense.view());
    cpu_ops::spmv_transpose_csr(-3.2, csr_A, x.view(), 2.0, y_csr.view());
    cpu_ops::spmv_transpose_csc(-3.2, csc_A, x.view(), 2.0, y_csc.view());

    for (index_t j = 0; j < 11; ++j) {
        EXPECT_NEAR(y_csr[j], y_dense[j], 1e-13);
        EXPECT_NEAR(y_csc[j], y_dense[j], 1e-13);
    }
}

// =============================================================================
// Simulation of First-Order Primal-Dual (PDHG) Iterations
// =============================================================================

TEST(PDHGSimulationTest, DenseVsSparseParityAcrossMultipleIterations) {
    // Problem setup:
    // Constraint matrix A: 5 x 4
    // Verify that running 15 steps of primal-dual matrix operations
    // produces identical state trajectories across Dense, CSR, and CSC.
    DenseMatrix dense_A = DenseMatrix::random(5, 4, -2.0, 2.0, 9999);
    dense_A(0, 1) = 0.0;
    dense_A(2, 3) = 0.0;
    dense_A(4, 0) = 0.0;

    CSRMatrix csr_A = dense_A.to_csr();
    CSCMatrix csc_A = dense_A.to_csc();

    scalar_t tau = 0.1;
    scalar_t sigma = 0.1;

    Vector lower_x = {-2.0, -2.0, -2.0, -2.0};
    Vector upper_x = {2.0, 2.0, 2.0, 2.0};
    Vector lower_y = {-5.0, -5.0, -5.0, -5.0, -5.0};
    Vector upper_y = {5.0, 5.0, 5.0, 5.0, 5.0};

    // State 1: Dense Reference Oracle
    Vector x_dense = {0.5, -0.5, 1.0, 0.0};
    Vector y_dense = {0.0, 0.0, 0.0, 0.0, 0.0};

    // State 2: CSR
    Vector x_csr = x_dense;
    Vector y_csr = y_dense;

    // State 3: CSC
    Vector x_csc = x_dense;
    Vector y_csc = y_dense;

    Vector grad_x_dense(4, 0.0);
    Vector grad_x_csr(4, 0.0);
    Vector grad_x_csc(4, 0.0);

    Vector Ax_dense(5, 0.0);
    Vector Ax_csr(5, 0.0);
    Vector Ax_csc(5, 0.0);

    for (int iter = 0; iter < 15; ++iter) {
        // --- Primal Step: x = project(x - tau * A^T * y) ---
        // 1. Dense
        dense_A.gemv_transpose(1.0, y_dense.view(), 0.0, grad_x_dense.view());
        cpu_ops::axpy(-tau, grad_x_dense.view(), x_dense.view());
        cpu_ops::project_box(x_dense.view(), lower_x.view(), upper_x.view());

        // 2. CSR
        cpu_ops::spmv_transpose_csr(1.0, csr_A, y_csr.view(), 0.0, grad_x_csr.view());
        cpu_ops::axpy(-tau, grad_x_csr.view(), x_csr.view());
        cpu_ops::project_box(x_csr.view(), lower_x.view(), upper_x.view());

        // 3. CSC
        cpu_ops::spmv_transpose_csc(1.0, csc_A, y_csc.view(), 0.0, grad_x_csc.view());
        cpu_ops::axpy(-tau, grad_x_csc.view(), x_csc.view());
        cpu_ops::project_box(x_csc.view(), lower_x.view(), upper_x.view());

        // Check primal parity at iteration
        for (index_t j = 0; j < 4; ++j) {
            EXPECT_NEAR(x_csr[j], x_dense[j], 1e-13);
            EXPECT_NEAR(x_csc[j], x_dense[j], 1e-13);
        }

        // --- Dual Step: y = project(y + sigma * A * x) ---
        // 1. Dense
        dense_A.gemv(1.0, x_dense.view(), 0.0, Ax_dense.view());
        cpu_ops::axpy(sigma, Ax_dense.view(), y_dense.view());
        cpu_ops::project_box(y_dense.view(), lower_y.view(), upper_y.view());

        // 2. CSR
        cpu_ops::spmv_csr(1.0, csr_A, x_csr.view(), 0.0, Ax_csr.view());
        cpu_ops::axpy(sigma, Ax_csr.view(), y_csr.view());
        cpu_ops::project_box(y_csr.view(), lower_y.view(), upper_y.view());

        // 3. CSC
        cpu_ops::spmv_csc(1.0, csc_A, x_csc.view(), 0.0, Ax_csc.view());
        cpu_ops::axpy(sigma, Ax_csc.view(), y_csc.view());
        cpu_ops::project_box(y_csc.view(), lower_y.view(), upper_y.view());

        // Check dual parity at iteration
        for (index_t i = 0; i < 5; ++i) {
            EXPECT_NEAR(y_csr[i], y_dense[i], 1e-13);
            EXPECT_NEAR(y_csc[i], y_dense[i], 1e-13);
        }
    }
}
