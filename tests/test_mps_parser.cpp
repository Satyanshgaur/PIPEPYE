#include <gtest/gtest.h>
#include <pipepye/model/mps_parser.hpp>
#include <cmath>
#include <filesystem>

using pipepye::index_t;

namespace {
std::string get_test_data_path(const std::string& rel_path) {
    // Check multiple potential working directories for test data
    std::vector<std::string> search_roots = {
        ".",
        "..",
        "../..",
        "/home/satyansh/pipepye"
    };
    for (const auto& root : search_roots) {
        std::filesystem::path p = std::filesystem::path(root) / rel_path;
        if (std::filesystem::exists(p)) {
            return p.string();
        }
    }
    return rel_path;
}
} // namespace

// =========================================================================
// Edge Case Tests (Six Hand-Authored Files)
// =========================================================================

TEST(MPSParserEdgeCases, ObjectiveParsingAndMaximization) {
    std::string path = get_test_data_path("tests/data/mps/edge_cases/objective.mps");
    pipepye::model::LinearProgram lp;
    auto status = pipepye::model::MPSParser::parse_file(path, lp);

    ASSERT_TRUE(status.is_ok()) << status.message();
    EXPECT_EQ(lp.name, "OBJTEST");
    EXPECT_TRUE(lp.is_maximization);
    EXPECT_EQ(lp.obj_name, "OBJ_MAIN");

    // 1 constraint (C01), 2 variables (X01, X02)
    EXPECT_EQ(lp.num_rows(), 1);
    EXPECT_EQ(lp.num_cols(), 2);
    EXPECT_EQ(lp.num_nonzeros(), 2);

    // Maximization normalized: min (-c)^T x
    // Original max 2*x01 - 3*x02 -> min -2*x01 + 3*x02
    ASSERT_EQ(lp.c.size(), 2);
    EXPECT_DOUBLE_EQ(lp.c[lp.col_name_to_idx["X01"]], -2.0);
    EXPECT_DOUBLE_EQ(lp.c[lp.col_name_to_idx["X02"]], 3.0);

    // Row C01: x01 + 2*x02 <= 10.0
    index_t r_idx = lp.row_name_to_idx["C01"];
    EXPECT_EQ(lp.row_senses[r_idx], pipepye::model::RowSense::LessEqual);
    EXPECT_DOUBLE_EQ(lp.row_lower[r_idx], -pipepye::model::Infinity);
    EXPECT_DOUBLE_EQ(lp.row_upper[r_idx], 10.0);

    // Bounds: 0 <= x1 <= 5, 0 <= x2 <= 5
    index_t c1 = lp.col_name_to_idx["X01"];
    index_t c2 = lp.col_name_to_idx["X02"];
    EXPECT_DOUBLE_EQ(lp.col_lower[c1], 0.0);
    EXPECT_DOUBLE_EQ(lp.col_upper[c1], 5.0);
    EXPECT_DOUBLE_EQ(lp.col_lower[c2], 0.0);
    EXPECT_DOUBLE_EQ(lp.col_upper[c2], 5.0);
}

TEST(MPSParserEdgeCases, EqualityConstraints) {
    std::string path = get_test_data_path("tests/data/mps/edge_cases/equality.mps");
    pipepye::model::LinearProgram lp;
    auto status = pipepye::model::MPSParser::parse_file(path, lp);

    ASSERT_TRUE(status.is_ok()) << status.message();
    EXPECT_EQ(lp.num_rows(), 2);
    EXPECT_EQ(lp.num_cols(), 2);
    EXPECT_EQ(lp.num_nonzeros(), 4);
    EXPECT_FALSE(lp.is_maximization);

    index_t eq1 = lp.row_name_to_idx["EQ1"];
    index_t eq2 = lp.row_name_to_idx["EQ2"];
    EXPECT_EQ(lp.row_senses[eq1], pipepye::model::RowSense::Equality);
    EXPECT_EQ(lp.row_senses[eq2], pipepye::model::RowSense::Equality);
    EXPECT_DOUBLE_EQ(lp.row_lower[eq1], 10.0);
    EXPECT_DOUBLE_EQ(lp.row_upper[eq1], 10.0);
    EXPECT_DOUBLE_EQ(lp.row_lower[eq2], 5.0);
    EXPECT_DOUBLE_EQ(lp.row_upper[eq2], 5.0);

    // Free variable bounds: FR -> [-inf, +inf]
    for (index_t j = 0; j < lp.num_cols(); ++j) {
        EXPECT_DOUBLE_EQ(lp.col_lower[j], -pipepye::model::Infinity);
        EXPECT_DOUBLE_EQ(lp.col_upper[j], pipepye::model::Infinity);
    }
}

TEST(MPSParserEdgeCases, InequalityLessConstraints) {
    std::string path = get_test_data_path("tests/data/mps/edge_cases/inequality_less.mps");
    pipepye::model::LinearProgram lp;
    auto status = pipepye::model::MPSParser::parse_file(path, lp);

    ASSERT_TRUE(status.is_ok()) << status.message();
    EXPECT_EQ(lp.num_rows(), 2);
    EXPECT_EQ(lp.num_cols(), 2);
    EXPECT_EQ(lp.num_nonzeros(), 4);

    index_t le1 = lp.row_name_to_idx["LE1"];
    index_t le2 = lp.row_name_to_idx["LE2"];
    EXPECT_EQ(lp.row_senses[le1], pipepye::model::RowSense::LessEqual);
    EXPECT_EQ(lp.row_senses[le2], pipepye::model::RowSense::LessEqual);
    EXPECT_DOUBLE_EQ(lp.row_lower[le1], -pipepye::model::Infinity);
    EXPECT_DOUBLE_EQ(lp.row_upper[le1], 8.0);
    EXPECT_DOUBLE_EQ(lp.row_lower[le2], -pipepye::model::Infinity);
    EXPECT_DOUBLE_EQ(lp.row_upper[le2], 12.0);
}

TEST(MPSParserEdgeCases, InequalityGreaterConstraints) {
    std::string path = get_test_data_path("tests/data/mps/edge_cases/inequality_greater.mps");
    pipepye::model::LinearProgram lp;
    auto status = pipepye::model::MPSParser::parse_file(path, lp);

    ASSERT_TRUE(status.is_ok()) << status.message();
    EXPECT_EQ(lp.num_rows(), 2);
    EXPECT_EQ(lp.num_cols(), 2);
    EXPECT_EQ(lp.num_nonzeros(), 4);

    index_t ge1 = lp.row_name_to_idx["GE1"];
    index_t ge2 = lp.row_name_to_idx["GE2"];
    EXPECT_EQ(lp.row_senses[ge1], pipepye::model::RowSense::GreaterEqual);
    EXPECT_EQ(lp.row_senses[ge2], pipepye::model::RowSense::GreaterEqual);
    EXPECT_DOUBLE_EQ(lp.row_lower[ge1], 3.0);
    EXPECT_DOUBLE_EQ(lp.row_upper[ge1], pipepye::model::Infinity);
    EXPECT_DOUBLE_EQ(lp.row_lower[ge2], 9.0);
    EXPECT_DOUBLE_EQ(lp.row_upper[ge2], pipepye::model::Infinity);
}

TEST(MPSParserEdgeCases, DiverseBoundTypes) {
    std::string path = get_test_data_path("tests/data/mps/edge_cases/bounds.mps");
    pipepye::model::LinearProgram lp;
    auto status = pipepye::model::MPSParser::parse_file(path, lp);

    ASSERT_TRUE(status.is_ok()) << status.message();
    EXPECT_EQ(lp.num_rows(), 1);
    EXPECT_EQ(lp.num_cols(), 7);
    EXPECT_EQ(lp.num_nonzeros(), 7);

    // Default variable: [0, +inf)
    index_t def_idx = lp.col_name_to_idx["X_DEF"];
    EXPECT_DOUBLE_EQ(lp.col_lower[def_idx], 0.0);
    EXPECT_DOUBLE_EQ(lp.col_upper[def_idx], pipepye::model::Infinity);

    // LO: [2.5, +inf)
    index_t lo_idx = lp.col_name_to_idx["X_LO"];
    EXPECT_DOUBLE_EQ(lp.col_lower[lo_idx], 2.5);
    EXPECT_DOUBLE_EQ(lp.col_upper[lo_idx], pipepye::model::Infinity);

    // UP: [0, 5.0]
    index_t up_idx = lp.col_name_to_idx["X_UP"];
    EXPECT_DOUBLE_EQ(lp.col_lower[up_idx], 0.0);
    EXPECT_DOUBLE_EQ(lp.col_upper[up_idx], 5.0);

    // FX: [7.0, 7.0]
    index_t fx_idx = lp.col_name_to_idx["X_FX"];
    EXPECT_DOUBLE_EQ(lp.col_lower[fx_idx], 7.0);
    EXPECT_DOUBLE_EQ(lp.col_upper[fx_idx], 7.0);

    // FR: [-inf, +inf]
    index_t fr_idx = lp.col_name_to_idx["X_FR"];
    EXPECT_DOUBLE_EQ(lp.col_lower[fr_idx], -pipepye::model::Infinity);
    EXPECT_DOUBLE_EQ(lp.col_upper[fr_idx], pipepye::model::Infinity);

    // MI + UP: [-inf, 3.0]
    index_t mi_idx = lp.col_name_to_idx["X_MI"];
    EXPECT_DOUBLE_EQ(lp.col_lower[mi_idx], -pipepye::model::Infinity);
    EXPECT_DOUBLE_EQ(lp.col_upper[mi_idx], 3.0);

    // PL: [0, +inf]
    index_t pl_idx = lp.col_name_to_idx["X_PL"];
    EXPECT_DOUBLE_EQ(lp.col_lower[pl_idx], 0.0);
    EXPECT_DOUBLE_EQ(lp.col_upper[pl_idx], pipepye::model::Infinity);
}

TEST(MPSParserEdgeCases, RangesAndIntegerMarkers) {
    std::string path = get_test_data_path("tests/data/mps/edge_cases/ranges_and_integers.mps");
    pipepye::model::LinearProgram lp;
    auto status = pipepye::model::MPSParser::parse_file(path, lp);

    ASSERT_TRUE(status.is_ok()) << status.message();
    EXPECT_EQ(lp.num_rows(), 3);
    EXPECT_EQ(lp.num_cols(), 3);
    EXPECT_EQ(lp.num_nonzeros(), 6);

    // RANGES:
    // R_L: sense L, RHS 10.0, range 3.0 -> [7.0, 10.0]
    index_t rl = lp.row_name_to_idx["R_L"];
    EXPECT_EQ(lp.row_senses[rl], pipepye::model::RowSense::Ranged);
    EXPECT_DOUBLE_EQ(lp.row_lower[rl], 7.0);
    EXPECT_DOUBLE_EQ(lp.row_upper[rl], 10.0);

    // R_G: sense G, RHS 5.0, range 4.0 -> [5.0, 9.0]
    index_t rg = lp.row_name_to_idx["R_G"];
    EXPECT_EQ(lp.row_senses[rg], pipepye::model::RowSense::Ranged);
    EXPECT_DOUBLE_EQ(lp.row_lower[rg], 5.0);
    EXPECT_DOUBLE_EQ(lp.row_upper[rg], 9.0);

    // R_E: sense E, RHS 8.0, range 2.0 -> [8.0, 10.0]
    index_t re = lp.row_name_to_idx["R_E"];
    EXPECT_EQ(lp.row_senses[re], pipepye::model::RowSense::Ranged);
    EXPECT_DOUBLE_EQ(lp.row_lower[re], 8.0);
    EXPECT_DOUBLE_EQ(lp.row_upper[re], 10.0);

    // Integer marker cards and bounds
    index_t c_cont = lp.col_name_to_idx["X_CONT"];
    index_t c_int = lp.col_name_to_idx["X_INT"];
    index_t c_bin = lp.col_name_to_idx["X_BIN"];

    EXPECT_EQ(lp.var_types[c_cont], pipepye::model::VariableType::Continuous);
    EXPECT_EQ(lp.var_types[c_int], pipepye::model::VariableType::Integer);
    EXPECT_EQ(lp.var_types[c_bin], pipepye::model::VariableType::Binary);

    EXPECT_DOUBLE_EQ(lp.col_lower[c_int], 0.0);
    EXPECT_DOUBLE_EQ(lp.col_upper[c_int], 10.0); // UI 10.0
    EXPECT_DOUBLE_EQ(lp.col_lower[c_bin], 0.0);
    EXPECT_DOUBLE_EQ(lp.col_upper[c_bin], 1.0);  // BV
}

// =========================================================================
// Canonical Netlib Benchmark Instance Verification
// =========================================================================

TEST(MPSParserNetlib, VerifyAFIRO) {
    std::string path = get_test_data_path("tests/data/mps/netlib/afiro.mps");
    pipepye::model::LinearProgram lp;
    auto status = pipepye::model::MPSParser::parse_file(path, lp);

    ASSERT_TRUE(status.is_ok()) << status.message();
    EXPECT_EQ(lp.name, "AFIRO");
    EXPECT_EQ(lp.obj_name, "COST");
    EXPECT_FALSE(lp.is_maximization);

    // Canonical Netlib dimensions: 27 constraints, 32 variables, 83 constraint NNZ
    EXPECT_EQ(lp.num_rows(), 27);
    EXPECT_EQ(lp.num_cols(), 32);
    EXPECT_EQ(lp.num_nonzeros(), 83);

    // Objective has 5 non-zero entries (making 88 total nonzeros in the MPS file)
    size_t obj_nonzeros = 0;
    for (auto coeff : lp.c) {
        if (std::abs(coeff) > 1e-15) obj_nonzeros++;
    }
    EXPECT_EQ(obj_nonzeros, 5);

    // Matrix representation validation (CSC vs CSR consistency)
    EXPECT_EQ(lp.csc_values.size(), 83);
    EXPECT_EQ(lp.csr_values.size(), 83);
    EXPECT_EQ(lp.csc_col_ptr.size(), 33);
    EXPECT_EQ(lp.csr_row_ptr.size(), 28);
    EXPECT_EQ(lp.csc_col_ptr[32], 83);
    EXPECT_EQ(lp.csr_row_ptr[27], 83);
}

TEST(MPSParserNetlib, VerifyBLEND) {
    std::string path = get_test_data_path("tests/data/mps/netlib/blend.mps");
    pipepye::model::LinearProgram lp;
    auto status = pipepye::model::MPSParser::parse_file(path, lp);

    ASSERT_TRUE(status.is_ok()) << status.message();
    EXPECT_EQ(lp.name, "BLEND");
    EXPECT_EQ(lp.obj_name, "C");
    EXPECT_FALSE(lp.is_maximization);

    // Canonical Netlib dimensions: 74 constraints, 83 variables, 491 constraint NNZ
    EXPECT_EQ(lp.num_rows(), 74);
    EXPECT_EQ(lp.num_cols(), 83);
    EXPECT_EQ(lp.num_nonzeros(), 491);

    size_t obj_nonzeros = 0;
    for (auto coeff : lp.c) {
        if (std::abs(coeff) > 1e-15) obj_nonzeros++;
    }
    EXPECT_EQ(obj_nonzeros, 30); // 491 + 30 = 521 total MPS non-zeros

    EXPECT_EQ(lp.csc_col_ptr[83], 491);
    EXPECT_EQ(lp.csr_row_ptr[74], 491);
}

TEST(MPSParserNetlib, VerifyADLITTLE) {
    std::string path = get_test_data_path("tests/data/mps/netlib/adlittle.mps");
    pipepye::model::LinearProgram lp;
    auto status = pipepye::model::MPSParser::parse_file(path, lp);

    ASSERT_TRUE(status.is_ok()) << status.message();
    EXPECT_EQ(lp.name, "ADLITTLE");
    EXPECT_EQ(lp.obj_name, ".Z....");
    EXPECT_FALSE(lp.is_maximization);

    // Canonical Netlib dimensions: 56 constraints, 97 variables, 383 constraint NNZ
    EXPECT_EQ(lp.num_rows(), 56);
    EXPECT_EQ(lp.num_cols(), 97);
    EXPECT_EQ(lp.num_nonzeros(), 383);

    size_t obj_nonzeros = 0;
    for (auto coeff : lp.c) {
        if (std::abs(coeff) > 1e-15) obj_nonzeros++;
    }
    EXPECT_EQ(obj_nonzeros, 82); // 383 + 82 = 465 total MPS non-zeros

    EXPECT_EQ(lp.csc_col_ptr[97], 383);
    EXPECT_EQ(lp.csr_row_ptr[56], 383);
}

TEST(MPSParserNetlib, VerifyBANDM) {
    std::string path = get_test_data_path("tests/data/mps/netlib/bandm.mps");
    pipepye::model::LinearProgram lp;
    auto status = pipepye::model::MPSParser::parse_file(path, lp);

    ASSERT_TRUE(status.is_ok()) << status.message();
    EXPECT_EQ(lp.name, "BANDM");
    EXPECT_EQ(lp.obj_name, "....1");
    EXPECT_FALSE(lp.is_maximization);

    // Canonical Netlib dimensions: 305 constraints, 472 variables, 2494 constraint NNZ
    EXPECT_EQ(lp.num_rows(), 305);
    EXPECT_EQ(lp.num_cols(), 472);
    EXPECT_EQ(lp.num_nonzeros(), 2494);

    size_t obj_nonzeros = 0;
    for (auto coeff : lp.c) {
        if (std::abs(coeff) > 1e-15) obj_nonzeros++;
    }
    EXPECT_EQ(obj_nonzeros, 165); // 2494 + 165 = 2659 total MPS non-zeros

    EXPECT_EQ(lp.csc_col_ptr[472], 2494);
    EXPECT_EQ(lp.csr_row_ptr[305], 2494);
}

TEST(MPSParserNetlib, VerifyBEACONFD) {
    std::string path = get_test_data_path("tests/data/mps/netlib/beaconfd.mps");
    pipepye::model::LinearProgram lp;
    auto status = pipepye::model::MPSParser::parse_file(path, lp);

    ASSERT_TRUE(status.is_ok()) << status.message();
    EXPECT_EQ(lp.name, "BEACONFD");
    EXPECT_EQ(lp.obj_name, "11CSTR");
    EXPECT_FALSE(lp.is_maximization);

    // Canonical Netlib dimensions: 173 constraints, 262 variables, 3375 constraint NNZ
    EXPECT_EQ(lp.num_rows(), 173);
    EXPECT_EQ(lp.num_cols(), 262);
    EXPECT_EQ(lp.num_nonzeros(), 3375);

    size_t obj_nonzeros = 0;
    for (auto coeff : lp.c) {
        if (std::abs(coeff) > 1e-15) obj_nonzeros++;
    }
    EXPECT_EQ(obj_nonzeros, 101); // 3375 + 101 = 3476 total MPS non-zeros

    EXPECT_EQ(lp.csc_col_ptr[262], 3375);
    EXPECT_EQ(lp.csr_row_ptr[173], 3375);
}
