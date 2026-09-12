#include <pipepye/utils/logger.hpp>
#include <iostream>
#include <chrono>
#include <iomanip>

namespace pipepye::utils {

void Logger::set_level(LogLevel level) noexcept {
    current_level_ = level;
}

LogLevel Logger::get_level() noexcept {
    return current_level_;
}

void Logger::log(LogLevel level, std::string_view tag, std::string_view message) {
    if (level < current_level_) {
        return;
    }

    const char* level_str = "[INFO]";
    std::ostream* out = &std::cout;

    switch (level) {
        case LogLevel::Debug:
            level_str = "[DEBUG]";
            break;
        case LogLevel::Info:
            level_str = "[INFO ]";
            break;
        case LogLevel::Warning:
            level_str = "[WARN ]";
            break;
        case LogLevel::Error:
            level_str = "[ERROR]";
            out = &std::cerr;
            break;
    }

    *out << level_str << " [" << tag << "] " << message << "\n";
}

} // namespace pipepye::utils
