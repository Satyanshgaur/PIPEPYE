#include <gtest/gtest.h>
#include <pipepye/model/mps_parser.hpp>
#include <sstream>

using namespace pipepye;
using namespace pipepye::model;

TEST(MPSWriterTest, RoundtripContinuousLP) {
    LinearProgram orig;
    orig.name = "ROUNDTRIP_LP";
    orig.is_maximization = false;
    orig.obj_name = "OBJ";

    orig.col_names = {"X1", "X2"};
    orig.col_name_to_idx["X1"] = 0;
    orig.col_name_to_idx["X2"] = 1;
    orig.c = {2.0, 3.0};
    orig.col_lower = {0.0, 1.0};
    orig.col_upper = {10.0, 5.0};
    orig.var_types = {VariableType::Continuous, VariableType::Continuous};

    orig.row_names = {"C1", "C2"};
    orig.row_name_to_idx["C1"] = 0;
    orig.row_name_to_idx["C2"] = 1;
    orig.row_lower = {0.0, -Infinity};
    orig.row_upper = {12.0, 8.0};
    orig.row_senses = {RowSense::LessEqual, RowSense::LessEqual};

    // A = [1 1; 2 1]
    orig.A_coo = sparse::COOMatrix(2, 2);
    orig.A_coo.add_entry(0, 0, 1.0);
    orig.A_coo.add_entry(0, 1, 1.0);
    orig.A_coo.add_entry(1, 0, 2.0);
    orig.A_coo.add_entry(1, 1, 1.0);

    auto csc = orig.A_coo.to_csc();
    orig.csc_col_ptr = csc.col_ptr_vector();
    orig.csc_row_ind = csc.row_ind_vector();
    orig.csc_values = csc.values_vector();

    // Write to stream
    std::ostringstream out;
    Status s_write = MPSParser::write_stream(out, orig);
    ASSERT_TRUE(s_write.is_ok());

    std::string mps_str = out.str();
    EXPECT_NE(mps_str.find("NAME          ROUNDTRIP_LP"), std::string::npos);
    EXPECT_NE(mps_str.find("ROWS"), std::string::npos);
    EXPECT_NE(mps_str.find("COLUMNS"), std::string::npos);
    EXPECT_NE(mps_str.find("RHS"), std::string::npos);
    EXPECT_NE(mps_str.find("BOUNDS"), std::string::npos);
    EXPECT_NE(mps_str.find("ENDATA"), std::string::npos);

    // Parse back
    std::istringstream in(mps_str);
    LinearProgram parsed;
    Status s_parse = MPSParser::parse_stream(in, parsed);
    ASSERT_TRUE(s_parse.is_ok());

    EXPECT_EQ(parsed.num_rows(), orig.num_rows());
    EXPECT_EQ(parsed.num_cols(), orig.num_cols());
    EXPECT_EQ(parsed.num_nonzeros(), orig.num_nonzeros());

    EXPECT_NEAR(parsed.c[0], orig.c[0], 1e-9);
    EXPECT_NEAR(parsed.c[1], orig.c[1], 1e-9);
    EXPECT_NEAR(parsed.col_lower[1], orig.col_lower[1], 1e-9);
    EXPECT_NEAR(parsed.col_upper[0], orig.col_upper[0], 1e-9);
}

TEST(MPSWriterTest, RoundtripBinaryAndIntegerVariables) {
    LinearProgram orig;
    orig.name = "ROUNDTRIP_MILP";
    orig.is_maximization = false;
    orig.obj_name = "COST";

    orig.col_names = {"B1", "I1", "C1"};
    orig.col_name_to_idx["B1"] = 0;
    orig.col_name_to_idx["I1"] = 1;
    orig.col_name_to_idx["C1"] = 2;
    orig.c = {5.0, 8.0, 2.5};
    orig.col_lower = {0.0, 0.0, 0.0};
    orig.col_upper = {1.0, 10.0, Infinity};
    orig.var_types = {VariableType::Binary, VariableType::Integer, VariableType::Continuous};

    orig.row_names = {"CAP"};
    orig.row_name_to_idx["CAP"] = 0;
    orig.row_lower = {-Infinity};
    orig.row_upper = {25.0};
    orig.row_senses = {RowSense::LessEqual};

    orig.A_coo = sparse::COOMatrix(1, 3);
    orig.A_coo.add_entry(0, 0, 3.0);
    orig.A_coo.add_entry(0, 1, 4.0);
    orig.A_coo.add_entry(0, 2, 1.5);

    auto csc = orig.A_coo.to_csc();
    orig.csc_col_ptr = csc.col_ptr_vector();
    orig.csc_row_ind = csc.row_ind_vector();
    orig.csc_values = csc.values_vector();

    std::ostringstream out;
    Status s_write = MPSParser::write_stream(out, orig);
    ASSERT_TRUE(s_write.is_ok());

    std::string mps_str = out.str();
    EXPECT_NE(mps_str.find("'INTORG'"), std::string::npos);
    EXPECT_NE(mps_str.find("'INTEND'"), std::string::npos);
    EXPECT_NE(mps_str.find("BV BND1      B1"), std::string::npos);
    EXPECT_NE(mps_str.find("UI BND1      I1"), std::string::npos);

    std::istringstream in(mps_str);
    LinearProgram parsed;
    Status s_parse = MPSParser::parse_stream(in, parsed);
    ASSERT_TRUE(s_parse.is_ok());

    EXPECT_EQ(parsed.num_cols(), 3);
    EXPECT_EQ(parsed.num_rows(), 1);
    EXPECT_EQ(parsed.var_types[0], VariableType::Binary);
    EXPECT_EQ(parsed.var_types[1], VariableType::Integer);
    EXPECT_EQ(parsed.var_types[2], VariableType::Continuous);
}
