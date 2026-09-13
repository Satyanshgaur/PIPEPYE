#pragma once

#include <pipepye/core/types.hpp>
#include <compare>
#include <ostream>

namespace pipepye::sparse {

/// @brief Matrix storage format identifier
enum class StorageFormat : uint8_t {
    COO = 0,  // Coordinate format: (row, col, value) triplets
    CSR,      // Compressed Sparse Row
    CSC,      // Compressed Sparse Column
    Dense     // Contiguous 2D dense buffer
};

/// @brief Major storage ordering (for traversal, iteration, and sorting)
enum class StorageOrder : uint8_t {
    RowMajor = 0, // Row-first, then column (natural for CSR)
    ColMajor      // Column-first, then row (natural for CSC)
};

/// @brief Individual non-zero element in coordinate format.
template <typename T = scalar_t, typename I = index_t>
struct Triplet {
    I row{0};
    I col{0};
    T val{static_cast<T>(0)};

    constexpr Triplet() noexcept = default;
    constexpr Triplet(I r, I c, T v) noexcept : row(r), col(c), val(v) {}

    auto operator<=>(const Triplet&) const = default;

    /// @brief Functor to order triplets in Row-Major order (row, then col)
    struct RowMajorLess {
        constexpr bool operator()(const Triplet& a, const Triplet& b) const noexcept {
            if (a.row != b.row) return a.row < b.row;
            return a.col < b.col;
        }
    };

    /// @brief Functor to order triplets in Column-Major order (col, then row)
    struct ColMajorLess {
        constexpr bool operator()(const Triplet& a, const Triplet& b) const noexcept {
            if (a.col != b.col) return a.col < b.col;
            return a.row < b.row;
        }
    };
};

using TripletF64 = Triplet<scalar_t, index_t>;

template <typename T, typename I>
inline std::ostream& operator<<(std::ostream& os, const Triplet<T, I>& t) {
    return os << "(" << t.row << ", " << t.col << "): " << t.val;
}

} // namespace pipepye::sparse
