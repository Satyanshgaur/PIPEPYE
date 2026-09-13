#include <gtest/gtest.h>
#include <pipepye/sparse/vector.hpp>
#include <sstream>
#include <cmath>

using namespace pipepye::sparse;
using pipepye::index_t;
using pipepye::scalar_t;

TEST(SparseVectorTest, DefaultConstructionAndSizing) {
    DenseVector<scalar_t> v;
    EXPECT_EQ(v.size(), 0);
    EXPECT_TRUE(v.empty());

    DenseVector<scalar_t> v10(10, 3.14);
    EXPECT_EQ(v10.size(), 10);
    EXPECT_FALSE(v10.empty());
    for (index_t i = 0; i < 10; ++i) {
        EXPECT_DOUBLE_EQ(v10[i], 3.14);
    }
}

TEST(SparseVectorTest, InitializerListAndVectorConversion) {
    Vector v = {1.0, 2.0, 3.0, 4.0, 5.0};
    EXPECT_EQ(v.size(), 5);
    EXPECT_DOUBLE_EQ(v[0], 1.0);
    EXPECT_DOUBLE_EQ(v[4], 5.0);

    std::vector<scalar_t> raw = {10.0, 20.0};
    Vector from_raw(raw);
    EXPECT_EQ(from_raw.size(), 2);
    EXPECT_DOUBLE_EQ(from_raw[1], 20.0);
}

TEST(SparseVectorTest, ElementAccessAndBoundsChecking) {
    Vector v = {10.0, 20.0, 30.0};
    EXPECT_DOUBLE_EQ(v(0), 10.0);
    EXPECT_DOUBLE_EQ(v[1], 20.0);
    EXPECT_DOUBLE_EQ(v.at(2), 30.0);

    EXPECT_THROW((void)v.at(-1), std::out_of_range);
    EXPECT_THROW((void)v.at(3), std::out_of_range);

    ConstVectorView view = v.view();
    EXPECT_THROW((void)view.at(-1), std::out_of_range);
    EXPECT_THROW((void)view.at(3), std::out_of_range);
}

TEST(SparseVectorTest, SubvectorSlicing) {
    Vector v = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0};
    ConstVectorView view = v.view();

    auto slice = view.subvector(2, 3); // elements 2, 3, 4
    EXPECT_EQ(slice.size(), 3);
    EXPECT_DOUBLE_EQ(slice[0], 2.0);
    EXPECT_DOUBLE_EQ(slice[1], 3.0);
    EXPECT_DOUBLE_EQ(slice[2], 4.0);

    EXPECT_THROW((void)view.subvector(-1, 2), std::out_of_range);
    EXPECT_THROW((void)view.subvector(2, 5), std::out_of_range);
}

TEST(SparseVectorTest, MathematicalNorms) {
    Vector v = {-3.0, 4.0, 0.0};

    // 1-norm: |-3| + |4| + |0| = 7
    EXPECT_DOUBLE_EQ(v.norm_1(), 7.0);

    // 2-norm: sqrt(9 + 16 + 0) = 5
    EXPECT_DOUBLE_EQ(v.norm_2(), 5.0);

    // Inf-norm: max(|-3|, |4|, |0|) = 4
    EXPECT_DOUBLE_EQ(v.norm_inf(), 4.0);

    // Sum: -3 + 4 + 0 = 1
    EXPECT_DOUBLE_EQ(v.sum(), 1.0);
}

TEST(SparseVectorTest, DotProduct) {
    Vector a = {1.0, 2.0, 3.0};
    Vector b = {4.0, -5.0, 6.0};

    // 1*4 + 2*(-5) + 3*6 = 4 - 10 + 18 = 12
    EXPECT_DOUBLE_EQ(a.dot(b), 12.0);

    Vector c = {1.0, 2.0};
    EXPECT_THROW((void)a.dot(c), std::invalid_argument);
}

TEST(SparseVectorTest, AxpyOperation) {
    Vector y = {1.0, 2.0, 3.0};
    Vector x = {10.0, 20.0, 30.0};

    // y = 2.5 * x + y
    y.axpy(2.5, x.view());

    EXPECT_DOUBLE_EQ(y[0], 26.0);
    EXPECT_DOUBLE_EQ(y[1], 52.0);
    EXPECT_DOUBLE_EQ(y[2], 78.0);

    Vector mismatch = {1.0, 2.0};
    EXPECT_THROW(y.axpy(1.0, mismatch.view()), std::invalid_argument);
}

TEST(SparseVectorTest, BoundProjection) {
    Vector x = {-2.0, 3.0, 7.0, 5.0};
    Vector lower = {0.0, 0.0, 0.0, 0.0};
    Vector upper = {5.0, 5.0, 5.0, 5.0};

    x.project_bounds(lower.view(), upper.view());

    EXPECT_DOUBLE_EQ(x[0], 0.0); // clamped from -2.0 to 0.0
    EXPECT_DOUBLE_EQ(x[1], 3.0); // untouched
    EXPECT_DOUBLE_EQ(x[2], 5.0); // clamped from 7.0 to 5.0
    EXPECT_DOUBLE_EQ(x[3], 5.0); // untouched
}

TEST(SparseVectorTest, StreamOutputFormatting) {
    Vector v = {1.5, -2.5, 3.0};
    std::ostringstream oss;
    oss << v;
    EXPECT_EQ(oss.str(), "[1.5, -2.5, 3]");
}
