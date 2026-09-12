#include <gtest/gtest.h>
#include <pipepye/core/types.hpp>
#include <pipepye/core/version.hpp>
#include <pipepye/core/status.hpp>

TEST(CoreTypesTest, PrecisionAndSizes) {
    // Crucial for numerical stability in linear programming: double precision (64-bit IEEE 754)
    EXPECT_EQ(sizeof(pipepye::scalar_t), 8);
    EXPECT_EQ(sizeof(pipepye::index_t), 4);
    EXPECT_EQ(sizeof(pipepye::big_index_t), 8);
}

TEST(VersionTest, SemanticVersionFormat) {
    std::string ver = pipepye::Version::to_string();
    EXPECT_FALSE(ver.empty());
    EXPECT_NE(ver.find("0.1.0"), std::string::npos);

    std::string info = pipepye::Version::build_info();
    EXPECT_NE(info.find("PipePye"), std::string::npos);
    EXPECT_NE(info.find("C++20"), std::string::npos);
}

TEST(StatusTest, OKStatus) {
    pipepye::Status status = pipepye::Status::OK();
    EXPECT_TRUE(status.is_ok());
    EXPECT_EQ(status.code(), pipepye::StatusCode::Success);
    EXPECT_EQ(status.message(), "OK");
}

TEST(StatusTest, ErrorStatus) {
    pipepye::Status err = pipepye::Status::InvalidArgument("Matrix dimension mismatch (m=10, n=0)");
    EXPECT_FALSE(err.is_ok());
    EXPECT_EQ(err.code(), pipepye::StatusCode::InvalidArgument);
    EXPECT_NE(err.to_string().find("InvalidArgument"), std::string::npos);
    EXPECT_NE(err.to_string().find("Matrix dimension mismatch"), std::string::npos);
}

TEST(StatusTest, EqualityComparisons) {
    pipepye::Status s1 = pipepye::Status::OK();
    pipepye::Status s2 = pipepye::Status::OK();
    pipepye::Status s3 = pipepye::Status::CudaError("Out of VRAM");

    EXPECT_EQ(s1, s2);
    EXPECT_NE(s1, s3);
}
