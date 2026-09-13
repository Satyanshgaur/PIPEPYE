#pragma once

#include <pipepye/sparse/coo_matrix.hpp>
#include <cstdint>
#include <string>

namespace pipepye::sparse {

/// @brief Reproducible synthetic sparse matrix generator.
/// Generates controlled matrices with various dimensions, densities, and structural patterns
/// (random, banded, block-sparse, staircase, and irregular power-law).
class MatrixGenerator {
public:
    /// @brief Generates a uniformly distributed random sparse matrix.
    /// @param num_rows Number of rows
    /// @param num_cols Number of columns
    /// @param density Approximate density in range (0.0, 1.0]
    /// @param min_val Minimum nonzero value
    /// @param max_val Maximum nonzero value
    /// @param seed Deterministic PRNG seed
    static COOMatrix generate_random(index_t num_rows, index_t num_cols,
                                     double density,
                                     scalar_t min_val = -10.0, scalar_t max_val = 10.0,
                                     uint32_t seed = 42);

    /// @brief Generates a banded sparse matrix with lower and upper bandwidths.
    /// Nonzeros occur only where -lower_bw <= col - row <= upper_bw.
    static COOMatrix generate_banded(index_t num_rows, index_t num_cols,
                                     index_t lower_bandwidth, index_t upper_bandwidth,
                                     scalar_t min_val = -10.0, scalar_t max_val = 10.0,
                                     uint32_t seed = 42);

    /// @brief Generates a block-diagonal / block-angular sparse matrix.
    /// @param num_blocks Number of diagonal blocks
    /// @param block_rows Rows per block
    /// @param block_cols Columns per block
    /// @param block_density Density within each diagonal block
    /// @param coupling_density Density of off-block coupling elements
    static COOMatrix generate_block_diagonal(index_t num_blocks,
                                             index_t block_rows, index_t block_cols,
                                             double block_density = 0.2,
                                             double coupling_density = 0.001,
                                             scalar_t min_val = -10.0, scalar_t max_val = 10.0,
                                             uint32_t seed = 42);

    /// @brief Generates a multi-period / staircase sparse matrix.
    /// Common in dynamic linear programming where time stage t couples with t-1 and t.
    static COOMatrix generate_staircase(index_t num_stages,
                                        index_t stage_rows, index_t stage_cols,
                                        double density = 0.15,
                                        scalar_t min_val = -10.0, scalar_t max_val = 10.0,
                                        uint32_t seed = 42);

    /// @brief Generates an irregular / scale-free matrix with heavy-tailed row degrees.
    /// A small percentage of "hub" rows contain a large fraction of total nonzeros.
    /// Crucial for testing thread load imbalance and memory stall characteristics.
    static COOMatrix generate_irregular(index_t num_rows, index_t num_cols,
                                        size_t target_nnz,
                                        double hub_row_fraction = 0.05,
                                        double hub_nnz_fraction = 0.50,
                                        scalar_t min_val = -10.0, scalar_t max_val = 10.0,
                                        uint32_t seed = 42);
};

} // namespace pipepye::sparse
