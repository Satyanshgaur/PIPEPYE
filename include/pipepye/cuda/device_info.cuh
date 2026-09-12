#pragma once

#include <string>
#include <cstddef>
#include <pipepye/core/status.hpp>

namespace pipepye::cuda {

struct DeviceProperties {
    int device_id{0};
    std::string name{};
    int major{0};
    int minor{0};
    size_t total_memory_bytes{0};
    size_t free_memory_bytes{0};
    int multi_processor_count{0};
    int warp_size{32};
    int max_threads_per_block{1024};
    int max_threads_per_multiprocessor{1536};
    int clock_rate_khz{0};
    int memory_bus_width_bits{0};
    int l2_cache_size_bytes{0};
    bool is_rtx_3050{false};

    [[nodiscard]] double total_memory_gb() const noexcept {
        return static_cast<double>(total_memory_bytes) / (1024.0 * 1024.0 * 1024.0);
    }

    [[nodiscard]] double free_memory_gb() const noexcept {
        return static_cast<double>(free_memory_bytes) / (1024.0 * 1024.0 * 1024.0);
    }

    [[nodiscard]] double theoretical_bandwidth_gb_s() const noexcept {
        // Bandwidth = Memory Clock (Hz) * Bus Width (bytes) * 2 (DDR) / 10^9
        return 0.0; // Populated during query
    }
};

/// @brief Returns the number of CUDA-capable compute devices available.
int get_device_count();

/// @brief Queries comprehensive hardware attributes of a given CUDA device.
DeviceProperties query_device(int device_id = 0);

/// @brief Pretty-prints GPU capabilities, focusing on Ampere / RTX 3050 traits.
void print_device_summary(const DeviceProperties& dev);

/// @brief Validates whether the queried GPU matches the development target (Ampere / sm_86 / RTX 3050).
bool is_rtx_3050_compatible(const DeviceProperties& dev);

} // namespace pipepye::cuda
