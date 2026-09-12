#pragma once

#include <string_view>
#include <string>
#include <iostream>
#include <sstream>

namespace pipepye::utils {

enum class LogLevel {
    Debug = 0,
    Info,
    Warning,
    Error
};

class Logger {
public:
    static void set_level(LogLevel level) noexcept;
    static LogLevel get_level() noexcept;

    static void log(LogLevel level, std::string_view tag, std::string_view message);

    static void debug(std::string_view tag, std::string_view message) {
        log(LogLevel::Debug, tag, message);
    }

    static void info(std::string_view tag, std::string_view message) {
        log(LogLevel::Info, tag, message);
    }

    static void warn(std::string_view tag, std::string_view message) {
        log(LogLevel::Warning, tag, message);
    }

    static void error(std::string_view tag, std::string_view message) {
        log(LogLevel::Error, tag, message);
    }

private:
    static inline LogLevel current_level_{LogLevel::Info};
};

} // namespace pipepye::utils
