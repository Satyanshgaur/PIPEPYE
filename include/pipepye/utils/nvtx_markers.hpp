#pragma once

#include <string_view>
#include <optional>

#if defined(PIPEPYE_HAS_CUDA) || defined(__CUDACC__)
    #if __has_include(<nvtx3/nvtx3.hpp>)
        #include <nvtx3/nvtx3.hpp>
        #define PIPEPYE_HAS_NVTX3 1
    #elif __has_include(<nvToolsExt.h>)
        #include <nvToolsExt.h>
        #define PIPEPYE_HAS_NVTX_LEGACY 1
    #endif
#endif

namespace pipepye::utils {

/// @brief RAII wrapper for NVTX profiling ranges.
/// When compiled with CUDA, maps to nvtx3::scoped_range (header-only, dynamic loading)
/// or legacy nvtxRangePushA / nvtxRangePop.
/// When compiled without CUDA (CPU-only), compiles down to a no-op.
class ScopedRange {
public:
    explicit ScopedRange([[maybe_unused]] const char* name) {
#if defined(PIPEPYE_HAS_NVTX3)
        range_.emplace(name);
#elif defined(PIPEPYE_HAS_NVTX_LEGACY)
        nvtxRangePushA(name);
        active_ = true;
#endif
    }

    ~ScopedRange() {
#if defined(PIPEPYE_HAS_NVTX_LEGACY)
        if (active_) {
            nvtxRangePop();
        }
#endif
    }

    ScopedRange(const ScopedRange&) = delete;
    ScopedRange& operator=(const ScopedRange&) = delete;
    ScopedRange(ScopedRange&&) = default;
    ScopedRange& operator=(ScopedRange&&) = default;

private:
#if defined(PIPEPYE_HAS_NVTX3)
    std::optional<nvtx3::scoped_range> range_;
#elif defined(PIPEPYE_HAS_NVTX_LEGACY)
    bool active_{false};
#endif
};

} // namespace pipepye::utils
