#include <pipepye/core/version.hpp>
#include <sstream>

namespace pipepye {

std::string Version::to_string() {
    std::ostringstream oss;
    oss << Major << "." << Minor << "." << Patch;
    if (!PreRelease.empty()) {
        oss << "-" << PreRelease;
    }
    return oss.str();
}

std::string Version::build_info() {
    std::ostringstream oss;
    oss << "PipePye Sovereign Optimization Solver v" << to_string() << "\n"
        << "  C++ Standard: C++20\n"
#if defined(__GNUC__)
        << "  Host Compiler: GCC " << __GNUC__ << "." << __GNUC_MINOR__ << "." << __GNUC_PATCHLEVEL__ << "\n"
#elif defined(__clang__)
        << "  Host Compiler: Clang " << __clang_version__ << "\n"
#endif
#if defined(__CUDACC__)
        << "  CUDA Compiler: NVCC " << __CUDACC_VER_MAJOR__ << "." << __CUDACC_VER_MINOR__ << "\n"
#endif
#if defined(PIPEPYE_CUDA_ENABLED)
        << "  CUDA Acceleration: Enabled (sm_86 Ampere Target)\n";
#else
        << "  CUDA Acceleration: Disabled (CPU only)\n";
#endif
    return oss.str();
}

} // namespace pipepye
