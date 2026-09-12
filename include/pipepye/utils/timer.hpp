#pragma once

#include <chrono>

namespace pipepye::utils {

class CPUTimer {
public:
    using Clock = std::chrono::high_resolution_clock;
    using TimePoint = std::chrono::time_point<Clock>;

    CPUTimer();

    void start() noexcept;
    void stop() noexcept;
    void reset() noexcept;

    [[nodiscard]] double elapsed_seconds() const noexcept;
    [[nodiscard]] double elapsed_milliseconds() const noexcept;
    [[nodiscard]] double elapsed_microseconds() const noexcept;

private:
    TimePoint start_time_{};
    TimePoint end_time_{};
    bool running_{false};
};

} // namespace pipepye::utils
