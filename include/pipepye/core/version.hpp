#pragma once

#include <string>
#include <string_view>

namespace pipepye {

struct Version {
    static constexpr int Major = 0;
    static constexpr int Minor = 1;
    static constexpr int Patch = 0;
    static constexpr std::string_view PreRelease = "alpha";

    /// @brief Returns the semver string format "MAJOR.MINOR.PATCH[-PRERELEASE]".
    static std::string to_string();

    /// @brief Returns a verbose banner describing project, compiler and CUDA configuration.
    static std::string build_info();
};

} // namespace pipepye
