#pragma once

#include <string_view>

#if defined(PIPEPYE_HAS_CUDA) || defined(__CUDACC__)
    #if __has_include(<nvtx3/nvtx3.hpp>)
        #include <nvtx3/nvtx3.hpp>
        #define PIPEPYE_HAS_NVTX 1
    #elif __has_include(<nvToolsExt.h>)
        #include <nvToolsExt.h>
        #define PIPEPYE_HAS_NVTX_LEGACY 1
    #endif
#endif

namespace pipepye::utils {

class ScopedRange {
public:
    explicit ScopedRange([[maybe_unused]] const char* name) {
#if defined(PIPEPYE_HAS_NVTX)
        nvtxRangePushA(name);
        active_ = true;
#elif defined(PIPEPYE_HAS_NVTX_LEGACY)
        nvtxRangePushA(name);
        active_ = true;
#endif
    }

    ~ScopedRange() {
#if defined(PIPEPYE_HAS_NVTX) || defined(PIPEPYE_HAS_NVTX_LEGACY)
        if (active_) {
            nvtxRangePop();
        }
#endif
    }

    ScopedRange(const ScopedRange&) = delete;
    ScopedRange& operator=(const ScopedRange&) = delete;

private:
#if defined(PIPEPYE_HAS_NVTX) || defined(PIPEPYE_HAS_NVTX_LEGACY)
    bool active_{false};
#endif
};

} // namespace pipepye::utils
