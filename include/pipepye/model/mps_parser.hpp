#pragma once

#include <pipepye/model/lp_model.hpp>
#include <pipepye/core/status.hpp>
#include <string>
#include <istream>

namespace pipepye::model {

class MPSParser {
public:
    /// @brief Parses an MPS file from disk into a LinearProgram model.
    /// Supports both Fixed and Free format with auto-detection.
    static Status parse_file(const std::string& filepath, LinearProgram& out_model);

    /// @brief Parses an MPS model from an input stream into a LinearProgram model.
    static Status parse_stream(std::istream& in, LinearProgram& out_model);

    /// @brief Writes a LinearProgram model to disk in standard MPS format.
    static Status write_file(const std::string& filepath, const LinearProgram& model);

    /// @brief Writes a LinearProgram model to an output stream in standard MPS format.
    static Status write_stream(std::ostream& out, const LinearProgram& model);
};

} // namespace pipepye::model
