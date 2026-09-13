#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <limits>
#include <cstdint>
#include <pipepye/core/types.hpp>

namespace pipepye::model {

inline constexpr scalar_t Infinity = std::numeric_limits<scalar_t>::infinity();

enum class VariableType : uint8_t {
    Continuous = 0,
    Binary,
    Integer
};

enum class RowSense : uint8_t {
    Equality = 0,    // a^T x = b       (l_r = u_r = b)
    LessEqual,       // a^T x <= b      (l_r = -inf, u_r = b)
    GreaterEqual,    // a^T x >= b      (l_r = b, u_r = +inf)
    Ranged,          // l <= a^T x <= u (both bounds finite)
    Free             // Free row        (l_r = -inf, u_r = +inf)
};

/// @brief Decoupled internal representation of a Linear Program in General Bounded Form:
///        min c^T x + obj_offset
///        s.t. l_r <= A x <= u_r
///             l_c <= x   <= u_c
struct LinearProgram {
    std::string name;
    bool is_maximization{false};
    scalar_t obj_offset{0.0};
    std::string obj_name;

    // Symbol Tables (bi-directional mapping: index <-> name)
    std::vector<std::string> col_names;
    std::vector<std::string> row_names;
    std::unordered_map<std::string, index_t> col_name_to_idx;
    std::unordered_map<std::string, index_t> row_name_to_idx;

    // Row senses
    std::vector<RowSense> row_senses;

    // Linear Objective: c^T x (always stored in minimization form)
    std::vector<scalar_t> c;

    // Column / Variable Bounds: col_lower <= x <= col_upper
    std::vector<scalar_t> col_lower;
    std::vector<scalar_t> col_upper;
    std::vector<VariableType> var_types;

    // Constraint Bounds: row_lower <= A x <= row_upper
    std::vector<scalar_t> row_lower;
    std::vector<scalar_t> row_upper;

    // Dual Sparse Representations of Constraint Matrix A
    // 1. Column-Compressed Sparse (CSC) for Simplex, Basis Operations & Column Pricing
    std::vector<index_t> csc_col_ptr; // size: num_cols + 1
    std::vector<index_t> csc_row_ind; // size: nnz
    std::vector<scalar_t> csc_values; // size: nnz

    // 2. Row-Compressed Sparse (CSR) for GPU SpMV, Presolve & KKT Checks
    std::vector<index_t> csr_row_ptr; // size: num_rows + 1
    std::vector<index_t> csr_col_ind; // size: nnz
    std::vector<scalar_t> csr_values; // size: nnz

    [[nodiscard]] index_t num_cols() const noexcept {
        return static_cast<index_t>(col_names.size());
    }

    [[nodiscard]] index_t num_rows() const noexcept {
        return static_cast<index_t>(row_names.size());
    }

    [[nodiscard]] size_t num_nonzeros() const noexcept {
        return csc_values.size();
    }
};

} // namespace pipepye::model
