#include <pipepye/model/mps_parser.hpp>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cctype>
#include <cmath>

namespace pipepye::model {

namespace {

inline std::string trim(const std::string& s) {
    auto start = s.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) return "";
    auto end = s.find_last_not_of(" \t\r\n");
    return s.substr(start, end - start + 1);
}

inline std::string to_upper(std::string s) {
    for (char& c : s) {
        c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    }
    return s;
}

std::vector<std::string> tokenize_line(const std::string& line) {
    std::vector<std::string> tokens;
    size_t i = 0;
    while (i < line.size()) {
        while (i < line.size() && (line[i] == ' ' || line[i] == '\t' || line[i] == '\r')) {
            ++i;
        }
        if (i >= line.size()) break;

        if (line[i] == '"' || line[i] == '\'') {
            char quote = line[i++];
            size_t start = i;
            while (i < line.size() && line[i] != quote) {
                ++i;
            }
            tokens.push_back(line.substr(start, i - start));
            if (i < line.size()) ++i;
        } else {
            size_t start = i;
            while (i < line.size() && line[i] != ' ' && line[i] != '\t' && line[i] != '\r') {
                ++i;
            }
            tokens.push_back(line.substr(start, i - start));
        }
    }
    return tokens;
}

enum class Section {
    None = 0,
    Name,
    ObjSense,
    ObjName,
    Rows,
    Columns,
    Rhs,
    Ranges,
    Bounds,
    Endata
};

} // namespace

Status MPSParser::parse_file(const std::string& filepath, LinearProgram& out_model) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        return Status::InvalidArgument("Failed to open MPS file: " + filepath);
    }
    return parse_stream(file, out_model);
}

Status MPSParser::parse_stream(std::istream& in, LinearProgram& out_model) {
    out_model = LinearProgram{};

    Section current_section = Section::None;
    std::string line;
    std::vector<pipepye::sparse::TripletF64> triplets;
    bool in_integer_marker = false;
    std::string first_rhs_name;
    std::string first_range_name;
    std::string first_bounds_name;

    while (std::getline(in, line)) {
        if (line.empty()) continue;
        if (line[0] == '*') continue; // Comment

        // Check for major section headers (always start at column 0 without leading whitespace)
        if (!std::isspace(static_cast<unsigned char>(line[0]))) {
            auto header_tokens = tokenize_line(line);
            if (header_tokens.empty()) continue;

            std::string header = to_upper(header_tokens[0]);
            if (header == "NAME") {
                current_section = Section::Name;
                if (header_tokens.size() > 1) {
                    out_model.name = header_tokens[1];
                }
                continue;
            } else if (header == "OBJSENSE") {
                current_section = Section::ObjSense;
                if (header_tokens.size() > 1) {
                    std::string sense = to_upper(header_tokens[1]);
                    if (sense == "MAX" || sense == "MAXIMIZE") {
                        out_model.is_maximization = true;
                    } else if (sense == "MIN" || sense == "MINIMIZE") {
                        out_model.is_maximization = false;
                    }
                }
                continue;
            } else if (header == "OBJNAME") {
                current_section = Section::ObjName;
                if (header_tokens.size() > 1) {
                    out_model.obj_name = header_tokens[1];
                }
                continue;
            } else if (header == "ROWS") {
                current_section = Section::Rows;
                continue;
            } else if (header == "COLUMNS") {
                current_section = Section::Columns;
                continue;
            } else if (header == "RHS") {
                current_section = Section::Rhs;
                continue;
            } else if (header == "RANGES") {
                current_section = Section::Ranges;
                continue;
            } else if (header == "BOUNDS") {
                current_section = Section::Bounds;
                continue;
            } else if (header == "ENDATA") {
                current_section = Section::Endata;
                break;
            }
        }

        auto tokens = tokenize_line(line);
        if (tokens.empty()) continue;

        switch (current_section) {
            case Section::ObjSense: {
                std::string sense = to_upper(tokens[0]);
                if (sense == "MAX" || sense == "MAXIMIZE") {
                    out_model.is_maximization = true;
                } else if (sense == "MIN" || sense == "MINIMIZE") {
                    out_model.is_maximization = false;
                }
                break;
            }

            case Section::ObjName: {
                out_model.obj_name = tokens[0];
                break;
            }

            case Section::Rows: {
                if (tokens.size() < 2) continue;
                std::string row_type = to_upper(tokens[0]);
                std::string row_name = tokens[1];

                if (row_type == "N") {
                    if (out_model.obj_name.empty()) {
                        out_model.obj_name = row_name;
                    }
                    // Additional N rows are free rows and not active constraints in A
                } else {
                    index_t row_idx = static_cast<index_t>(out_model.row_names.size());
                    out_model.row_names.push_back(row_name);
                    out_model.row_name_to_idx[row_name] = row_idx;

                    if (row_type == "E") {
                        out_model.row_senses.push_back(RowSense::Equality);
                        out_model.row_lower.push_back(0.0);
                        out_model.row_upper.push_back(0.0);
                    } else if (row_type == "L") {
                        out_model.row_senses.push_back(RowSense::LessEqual);
                        out_model.row_lower.push_back(-Infinity);
                        out_model.row_upper.push_back(0.0);
                    } else if (row_type == "G") {
                        out_model.row_senses.push_back(RowSense::GreaterEqual);
                        out_model.row_lower.push_back(0.0);
                        out_model.row_upper.push_back(Infinity);
                    } else {
                        return Status::InvalidArgument("Unknown row type '" + row_type + "' for row: " + row_name);
                    }
                }
                break;
            }

            case Section::Columns: {
                if (tokens.size() < 3) continue;

                // Check for integer marker cards
                if (tokens.size() >= 3 && (tokens[1] == "MARKER" || tokens[1] == "'MARKER'")) {
                    std::string marker_type = to_upper(tokens[2]);
                    if (marker_type == "INTORG" || marker_type == "'INTORG'") {
                        in_integer_marker = true;
                    } else if (marker_type == "INTEND" || marker_type == "'INTEND'") {
                        in_integer_marker = false;
                    }
                    continue;
                }

                std::string col_name = tokens[0];
                index_t col_idx = 0;
                auto it = out_model.col_name_to_idx.find(col_name);
                if (it == out_model.col_name_to_idx.end()) {
                    col_idx = static_cast<index_t>(out_model.col_names.size());
                    out_model.col_names.push_back(col_name);
                    out_model.col_name_to_idx[col_name] = col_idx;
                    out_model.c.push_back(0.0);
                    out_model.col_lower.push_back(0.0); // Standard MPS default
                    out_model.col_upper.push_back(Infinity);
                    out_model.var_types.push_back(in_integer_marker ? VariableType::Integer : VariableType::Continuous);
                } else {
                    col_idx = it->second;
                }

                // Process (row, val) pairs (up to two pairs per line)
                for (size_t p = 1; p + 1 < tokens.size(); p += 2) {
                    std::string row_name = tokens[p];
                    scalar_t val = std::stod(tokens[p + 1]);

                    if (row_name == out_model.obj_name) {
                        out_model.c[col_idx] += val;
                    } else {
                        auto r_it = out_model.row_name_to_idx.find(row_name);
                        if (r_it != out_model.row_name_to_idx.end()) {
                            triplets.push_back({r_it->second, col_idx, val});
                        }
                    }
                }
                break;
            }

            case Section::Rhs: {
                if (tokens.size() < 3) continue;
                std::string rhs_name = tokens[0];
                if (first_rhs_name.empty()) {
                    first_rhs_name = rhs_name;
                } else if (rhs_name != first_rhs_name) {
                    // Ignore secondary RHS vectors per MPS convention
                    continue;
                }

                for (size_t p = 1; p + 1 < tokens.size(); p += 2) {
                    std::string row_name = tokens[p];
                    scalar_t val = std::stod(tokens[p + 1]);

                    if (row_name == out_model.obj_name) {
                        out_model.obj_offset -= val;
                    } else {
                        auto r_it = out_model.row_name_to_idx.find(row_name);
                        if (r_it != out_model.row_name_to_idx.end()) {
                            index_t r_idx = r_it->second;
                            RowSense sense = out_model.row_senses[r_idx];
                            if (sense == RowSense::Equality) {
                                out_model.row_lower[r_idx] = val;
                                out_model.row_upper[r_idx] = val;
                            } else if (sense == RowSense::LessEqual) {
                                out_model.row_upper[r_idx] = val;
                            } else if (sense == RowSense::GreaterEqual) {
                                out_model.row_lower[r_idx] = val;
                            }
                        }
                    }
                }
                break;
            }

            case Section::Ranges: {
                if (tokens.size() < 3) continue;
                std::string range_name = tokens[0];
                if (first_range_name.empty()) {
                    first_range_name = range_name;
                } else if (range_name != first_range_name) {
                    continue;
                }

                for (size_t p = 1; p + 1 < tokens.size(); p += 2) {
                    std::string row_name = tokens[p];
                    scalar_t r_val = std::stod(tokens[p + 1]);

                    auto r_it = out_model.row_name_to_idx.find(row_name);
                    if (r_it != out_model.row_name_to_idx.end()) {
                        index_t r_idx = r_it->second;
                        RowSense sense = out_model.row_senses[r_idx];
                        out_model.row_senses[r_idx] = RowSense::Ranged;

                        if (sense == RowSense::LessEqual) {
                            scalar_t b = out_model.row_upper[r_idx];
                            out_model.row_lower[r_idx] = b - std::abs(r_val);
                        } else if (sense == RowSense::GreaterEqual) {
                            scalar_t b = out_model.row_lower[r_idx];
                            out_model.row_upper[r_idx] = b + std::abs(r_val);
                        } else if (sense == RowSense::Equality) {
                            scalar_t b = out_model.row_lower[r_idx];
                            if (r_val > 0.0) {
                                out_model.row_upper[r_idx] = b + r_val;
                            } else {
                                out_model.row_lower[r_idx] = b + r_val;
                            }
                        }
                    }
                }
                break;
            }

            case Section::Bounds: {
                if (tokens.size() < 3) continue;
                std::string bnd_type = to_upper(tokens[0]);
                std::string bnd_name = tokens[1];
                std::string col_name = tokens[2];

                if (first_bounds_name.empty()) {
                    first_bounds_name = bnd_name;
                } else if (bnd_name != first_bounds_name) {
                    continue;
                }

                auto c_it = out_model.col_name_to_idx.find(col_name);
                if (c_it == out_model.col_name_to_idx.end()) continue;
                index_t col_idx = c_it->second;

                scalar_t val = 0.0;
                if (tokens.size() >= 4 && bnd_type != "FR" && bnd_type != "BV" && bnd_type != "MI" && bnd_type != "PL") {
                    val = std::stod(tokens[3]);
                }

                if (bnd_type == "LO") {
                    out_model.col_lower[col_idx] = val;
                } else if (bnd_type == "UP") {
                    out_model.col_upper[col_idx] = val;
                } else if (bnd_type == "FX") {
                    out_model.col_lower[col_idx] = val;
                    out_model.col_upper[col_idx] = val;
                } else if (bnd_type == "FR") {
                    out_model.col_lower[col_idx] = -Infinity;
                    out_model.col_upper[col_idx] = Infinity;
                } else if (bnd_type == "MI") {
                    out_model.col_lower[col_idx] = -Infinity;
                    if (out_model.col_upper[col_idx] == Infinity) {
                        out_model.col_upper[col_idx] = 0.0;
                    }
                } else if (bnd_type == "PL") {
                    out_model.col_upper[col_idx] = Infinity;
                } else if (bnd_type == "BV") {
                    out_model.col_lower[col_idx] = 0.0;
                    out_model.col_upper[col_idx] = 1.0;
                    out_model.var_types[col_idx] = VariableType::Binary;
                } else if (bnd_type == "UI") {
                    out_model.col_upper[col_idx] = val;
                    out_model.var_types[col_idx] = VariableType::Integer;
                } else if (bnd_type == "LI") {
                    out_model.col_lower[col_idx] = val;
                    out_model.var_types[col_idx] = VariableType::Integer;
                }
                break;
            }

            default:
                break;
        }
    }

    // Normalization for Maximization: min (-c)^T x
    if (out_model.is_maximization) {
        for (auto& coeff : out_model.c) {
            coeff = -coeff;
        }
        out_model.obj_offset = -out_model.obj_offset;
    }

    // Matrix Construction: Convert triplets to COO, CSC & CSR
    index_t num_rows = out_model.num_rows();
    index_t num_cols = out_model.num_cols();

    out_model.A_coo = pipepye::sparse::COOMatrix(num_rows, num_cols, std::move(triplets));
    out_model.A_coo.sort(pipepye::sparse::StorageOrder::ColMajor);
    out_model.A_coo.sum_duplicates();

    auto csc = out_model.A_coo.to_csc();
    auto csr = out_model.A_coo.to_csr();

    out_model.csc_col_ptr = csc.col_ptr_vector();
    out_model.csc_row_ind = csc.row_ind_vector();
    out_model.csc_values = csc.values_vector();

    out_model.csr_row_ptr = csr.row_ptr_vector();
    out_model.csr_col_ind = csr.col_ind_vector();
    out_model.csr_values = csr.values_vector();

    return Status::OK();
}

} // namespace pipepye::model
