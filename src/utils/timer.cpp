#include <pipepye/utils/timer.hpp>

namespace pipepye::utils {

CPUTimer::CPUTimer() {
    reset();
}

void CPUTimer::start() noexcept {
    start_time_ = Clock::now();
    running_ = true;
}

void CPUTimer::stop() noexcept {
    if (running_) {
        end_time_ = Clock::now();
        running_ = false;
    }
}

void CPUTimer::reset() noexcept {
    start_time_ = Clock::now();
    end_time_ = start_time_;
    running_ = false;
}

double CPUTimer::elapsed_seconds() const noexcept {
    auto current = running_ ? Clock::now() : end_time_;
    std::chrono::duration<double> diff = current - start_time_;
    return diff.count();
}

double CPUTimer::elapsed_milliseconds() const noexcept {
    return elapsed_seconds() * 1000.0;
}

double CPUTimer::elapsed_microseconds() const noexcept {
    return elapsed_seconds() * 1000000.0;
}

} // namespace pipepye::utils
