#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <limits>
#include <cstdint>
#include <pipepye/core/types.hpp>
#include <pipepye/sparse/coo_matrix.hpp>

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

    // Coordinate Sparse representation (natural assembly buffer)
    pipepye::sparse::COOMatrix A_coo;

    // Dual Sparse Representations of Constraint Matrix A
    // 1. Column-Compressed Sparse (CSC) for Simplex, Basis Operations & Column Pricing
    std::vector<index_t> csc_col_ptr; // size: num_cols + 1
    std::vector<index_t> csc_row_ind; // size: nnz
    std::vector<scalar_t> csc_values; // size: nnz

    // 2. Row-Compressed Sparse (CSR) for GPU SpMV, Presolve & KKT Checks
    std::vector<index_t> csr_row_ptr; // size: num_rows + 1
    std::vector<index_t> csr_col_ind; // size: nnz
    std::vector<scalar_t> csr_values; // size: nnz

    [[nodiscard]] pipepye::sparse::CSRMatrix to_csr() const {
        return pipepye::sparse::CSRMatrix(num_rows(), num_cols(), csr_row_ptr, csr_col_ind, csr_values);
    }

    [[nodiscard]] pipepye::sparse::CSCMatrix to_csc() const {
        return pipepye::sparse::CSCMatrix(num_rows(), num_cols(), csc_col_ptr, csc_row_ind, csc_values);
    }

    [[nodiscard]] index_t num_cols() const noexcept {
        if (!col_names.empty()) return static_cast<index_t>(col_names.size());
        if (!c.empty()) return static_cast<index_t>(c.size());
        return static_cast<index_t>(col_lower.size());
    }

    [[nodiscard]] index_t num_rows() const noexcept {
        if (!row_names.empty()) return static_cast<index_t>(row_names.size());
        if (!row_lower.empty()) return static_cast<index_t>(row_lower.size());
        if (csr_row_ptr.size() > 1) return static_cast<index_t>(csr_row_ptr.size() - 1);
        return 0;
    }

    [[nodiscard]] size_t num_nonzeros() const noexcept {
        if (!csc_values.empty()) return csc_values.size();
        return csr_values.size();
    }
};

} // namespace pipepye::model
