#include <pipepye/sparse/matrix_generator.hpp>
#include <random>
#include <stdexcept>
#include <algorithm>
#include <unordered_set>

namespace pipepye::sparse {

namespace {

scalar_t generate_nonzero_val(std::mt19937& gen,
                              std::uniform_real_distribution<scalar_t>& dist,
                              scalar_t fallback = 1.0) {
    scalar_t val = dist(gen);
    if (val == 0.0) val = fallback;
    return val;
}

} // namespace

COOMatrix MatrixGenerator::generate_random(index_t num_rows, index_t num_cols,
                                           double density,
                                           scalar_t min_val, scalar_t max_val,
                                           uint32_t seed) {
    if (num_rows <= 0 || num_cols <= 0) {
        throw std::invalid_argument("MatrixGenerator::generate_random dimensions must be positive");
    }
    if (density <= 0.0 || density > 1.0) {
        throw std::invalid_argument("MatrixGenerator::generate_random density must be in (0.0, 1.0]");
    }

    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> prob_dist(0.0, 1.0);
    std::uniform_real_distribution<scalar_t> val_dist(min_val, max_val);

    size_t expected_nnz = static_cast<size_t>(static_cast<double>(num_rows) * static_cast<double>(num_cols) * density);
    COOMatrix coo(num_rows, num_cols, expected_nnz);

    for (index_t i = 0; i < num_rows; ++i) {
        for (index_t j = 0; j < num_cols; ++j) {
            if (prob_dist(gen) <= density) {
                coo.add_entry(i, j, generate_nonzero_val(gen, val_dist));
            }
        }
    }

    coo.sort(StorageOrder::RowMajor);
    return coo;
}

COOMatrix MatrixGenerator::generate_banded(index_t num_rows, index_t num_cols,
                                           index_t lower_bandwidth, index_t upper_bandwidth,
                                           scalar_t min_val, scalar_t max_val,
                                           uint32_t seed) {
    if (num_rows <= 0 || num_cols <= 0) {
        throw std::invalid_argument("MatrixGenerator::generate_banded dimensions must be positive");
    }
    if (lower_bandwidth < 0 || upper_bandwidth < 0) {
        throw std::invalid_argument("MatrixGenerator::generate_banded bandwidths must be non-negative");
    }

    std::mt19937 gen(seed);
    std::uniform_real_distribution<scalar_t> val_dist(min_val, max_val);

    size_t bandwidth = static_cast<size_t>(lower_bandwidth + upper_bandwidth + 1);
    COOMatrix coo(num_rows, num_cols, static_cast<size_t>(num_rows) * bandwidth);

    for (index_t i = 0; i < num_rows; ++i) {
        index_t j_start = std::max(static_cast<index_t>(0), i - lower_bandwidth);
        index_t j_end = std::min(num_cols - 1, i + upper_bandwidth);
        for (index_t j = j_start; j <= j_end; ++j) {
            coo.add_entry(i, j, generate_nonzero_val(gen, val_dist));
        }
    }

    coo.sort(StorageOrder::RowMajor);
    return coo;
}

COOMatrix MatrixGenerator::generate_block_diagonal(index_t num_blocks,
                                                   index_t block_rows, index_t block_cols,
                                                   double block_density,
                                                   double coupling_density,
                                                   scalar_t min_val, scalar_t max_val,
                                                   uint32_t seed) {
    if (num_blocks <= 0 || block_rows <= 0 || block_cols <= 0) {
        throw std::invalid_argument("MatrixGenerator::generate_block_diagonal block counts must be positive");
    }

    index_t total_rows = num_blocks * block_rows;
    index_t total_cols = num_blocks * block_cols;

    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> prob_dist(0.0, 1.0);
    std::uniform_real_distribution<scalar_t> val_dist(min_val, max_val);

    COOMatrix coo(total_rows, total_cols);

    for (index_t b = 0; b < num_blocks; ++b) {
        index_t r_offset = b * block_rows;
        index_t c_offset = b * block_cols;

        // Diagonal block nonzeros
        for (index_t r = 0; r < block_rows; ++r) {
            for (index_t c = 0; c < block_cols; ++c) {
                if (prob_dist(gen) <= block_density) {
                    coo.add_entry(r_offset + r, c_offset + c, generate_nonzero_val(gen, val_dist));
                }
            }
        }
    }

    // Optional off-diagonal coupling elements
    if (coupling_density > 0.0 && num_blocks > 1) {
        for (index_t i = 0; i < total_rows; ++i) {
            index_t my_block = i / block_rows;
            for (index_t j = 0; j < total_cols; ++j) {
                index_t col_block = j / block_cols;
                if (my_block != col_block && prob_dist(gen) <= coupling_density) {
                    coo.add_entry(i, j, generate_nonzero_val(gen, val_dist));
                }
            }
        }
    }

    coo.sort(StorageOrder::RowMajor);
    return coo;
}

COOMatrix MatrixGenerator::generate_staircase(index_t num_stages,
                                              index_t stage_rows, index_t stage_cols,
                                              double density,
                                              scalar_t min_val, scalar_t max_val,
                                              uint32_t seed) {
    if (num_stages <= 0 || stage_rows <= 0 || stage_cols <= 0) {
        throw std::invalid_argument("MatrixGenerator::generate_staircase stages must be positive");
    }

    index_t total_rows = num_stages * stage_rows;
    index_t total_cols = (num_stages + 1) * stage_cols;

    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> prob_dist(0.0, 1.0);
    std::uniform_real_distribution<scalar_t> val_dist(min_val, max_val);

    COOMatrix coo(total_rows, total_cols);

    for (index_t s = 0; s < num_stages; ++s) {
        index_t r_start = s * stage_rows;
        index_t r_end = (s + 1) * stage_rows;

        // Columns span stage s and stage s+1
        index_t c_start = s * stage_cols;
        index_t c_end = (s + 2) * stage_cols;

        for (index_t r = r_start; r < r_end; ++r) {
            for (index_t c = c_start; c < c_end; ++c) {
                if (prob_dist(gen) <= density) {
                    coo.add_entry(r, c, generate_nonzero_val(gen, val_dist));
                }
            }
        }
    }

    coo.sort(StorageOrder::RowMajor);
    return coo;
}

COOMatrix MatrixGenerator::generate_irregular(index_t num_rows, index_t num_cols,
                                              size_t target_nnz,
                                              double hub_row_fraction,
                                              double hub_nnz_fraction,
                                              scalar_t min_val, scalar_t max_val,
                                              uint32_t seed) {
    if (num_rows <= 0 || num_cols <= 0 || target_nnz == 0) {
        throw std::invalid_argument("MatrixGenerator::generate_irregular arguments must be positive");
    }

    size_t max_possible = static_cast<size_t>(num_rows) * static_cast<size_t>(num_cols);
    if (target_nnz > max_possible) target_nnz = max_possible;

    std::mt19937 gen(seed);
    std::uniform_real_distribution<scalar_t> val_dist(min_val, max_val);
    std::uniform_int_distribution<index_t> col_dist(0, num_cols - 1);

    index_t hub_count = std::max(static_cast<index_t>(1), static_cast<index_t>(num_rows * hub_row_fraction));
    size_t hub_nnz = static_cast<size_t>(target_nnz * hub_nnz_fraction);
    size_t max_hub_capacity = static_cast<size_t>(hub_count) * static_cast<size_t>(num_cols);
    if (hub_nnz > max_hub_capacity) hub_nnz = max_hub_capacity;

    size_t regular_nnz = target_nnz - hub_nnz;
    index_t regular_count = num_rows - hub_count;

    COOMatrix coo(num_rows, num_cols, target_nnz);
    std::unordered_set<uint64_t> visited;
    visited.reserve(target_nnz);

    // 1. Populate hub rows (dense)
    size_t hub_placed = 0;
    std::uniform_int_distribution<index_t> hub_row_dist(0, hub_count - 1);
    while (hub_placed < hub_nnz) {
        index_t r = hub_row_dist(gen);
        index_t c = col_dist(gen);
        uint64_t key = (static_cast<uint64_t>(r) << 32) | static_cast<uint64_t>(c);
        if (visited.insert(key).second) {
            coo.add_entry(r, c, generate_nonzero_val(gen, val_dist));
            ++hub_placed;
        }
    }

    // 2. Populate regular rows (sparse)
    if (regular_count > 0 && regular_nnz > 0) {
        size_t regular_placed = 0;
        std::uniform_int_distribution<index_t> regular_row_dist(hub_count, num_rows - 1);
        while (regular_placed < regular_nnz) {
            index_t r = regular_row_dist(gen);
            index_t c = col_dist(gen);
            uint64_t key = (static_cast<uint64_t>(r) << 32) | static_cast<uint64_t>(c);
            if (visited.insert(key).second) {
                coo.add_entry(r, c, generate_nonzero_val(gen, val_dist));
                ++regular_placed;
            }
        }
    }

    coo.sort(StorageOrder::RowMajor);
    return coo;
}

} // namespace pipepye::sparse
