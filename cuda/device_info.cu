#include <pipepye/cuda/device_info.cuh>
#include <pipepye/cuda/cuda_check.cuh>
#include <iostream>
#include <iomanip>
#include <algorithm>

namespace pipepye::cuda {

int get_device_count() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        // If driver is not loaded or no GPU present, return 0 instead of aborting
        cudaGetLastError(); // Clear error
        return 0;
    }
    return count;
}

DeviceProperties query_device(int device_id) {
    int count = get_device_count();
    if (count == 0) {
        throw CudaException(cudaErrorNoDevice, __FILE__, __LINE__, __func__, "No CUDA-capable device detected");
    }
    if (device_id < 0 || device_id >= count) {
        throw CudaException(cudaErrorInvalidDevice, __FILE__, __LINE__, __func__, "Invalid CUDA device index");
    }

    CUDA_CHECK(cudaSetDevice(device_id));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));

    size_t free_mem = 0, total_mem = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));

    DeviceProperties dev;
    dev.device_id = device_id;
    dev.name = prop.name;
    dev.major = prop.major;
    dev.minor = prop.minor;
    dev.total_memory_bytes = total_mem;
    dev.free_memory_bytes = free_mem;
    dev.multi_processor_count = prop.multiProcessorCount;
    dev.warp_size = prop.warpSize;
    dev.max_threads_per_block = prop.maxThreadsPerBlock;
    dev.max_threads_per_multiprocessor = prop.maxThreadsPerMultiProcessor;
    int clock_rate = 0;
    cudaDeviceGetAttribute(&clock_rate, cudaDevAttrClockRate, device_id);
    dev.clock_rate_khz = clock_rate;

    dev.memory_bus_width_bits = prop.memoryBusWidth;
    dev.l2_cache_size_bytes = prop.l2CacheSize;

    // Check for RTX 3050 identifier in device name
    std::string name_lower = dev.name;
    std::transform(name_lower.begin(), name_lower.end(), name_lower.begin(), ::tolower);
    dev.is_rtx_3050 = (name_lower.find("3050") != std::string::npos);

    return dev;
}

void print_device_summary(const DeviceProperties& dev) {
    std::cout << "=================================================================\n";
    std::cout << " CUDA Hardware Diagnostics: Device [" << dev.device_id << "]\n";
    std::cout << "=================================================================\n";
    std::cout << " Device Model Name          : " << dev.name << "\n";
    std::cout << " Target Architecture Match  : " << (dev.is_rtx_3050 ? "YES (RTX 3050 Detected)" : "Compatible GPU") << "\n";
    std::cout << " Compute Capability         : " << dev.major << "." << dev.minor;
    if (dev.major == 8 && dev.minor == 6) {
        std::cout << " (Ampere sm_86 - native)";
    }
    std::cout << "\n";
    std::cout << " Streaming Multiprocessors  : " << dev.multi_processor_count << " SMs\n";
    std::cout << " CUDA Cores (approx)        : " << dev.multi_processor_count * 128 << " FP32 cores\n";
    std::cout << " Total Global VRAM          : " << std::fixed << std::setprecision(2) << dev.total_memory_gb() << " GiB (" << dev.total_memory_bytes << " bytes)\n";
    std::cout << " Available / Free VRAM      : " << std::fixed << std::setprecision(2) << dev.free_memory_gb() << " GiB\n";
    std::cout << " L2 Cache Size              : " << dev.l2_cache_size_bytes / 1024 << " KiB\n";
    std::cout << " Memory Bus Width           : " << dev.memory_bus_width_bits << "-bit\n";
    std::cout << " Warp Size                  : " << dev.warp_size << " threads\n";
    std::cout << " Max Threads per Block      : " << dev.max_threads_per_block << "\n";
    std::cout << " Max Threads per SM         : " << dev.max_threads_per_multiprocessor << "\n";
    std::cout << " Base/Boost Clock Rate      : " << dev.clock_rate_khz / 1000.0 << " MHz\n";
    std::cout << "=================================================================\n";
}

bool is_rtx_3050_compatible(const DeviceProperties& dev) {
    // Either explicit RTX 3050 or any Ampere architecture (sm_86)
    return dev.is_rtx_3050 || (dev.major == 8 && dev.minor == 6);
}

} // namespace pipepye::cuda
