#pragma once

#include <pipepye/core/types.hpp>
#include <string>
#include <string_view>
#include <ostream>

namespace pipepye {

class Status {
public:
    Status() : code_(StatusCode::Success), message_("") {}
    Status(StatusCode code, std::string message)
        : code_(code), message_(std::move(message)) {}

    [[nodiscard]] bool is_ok() const noexcept {
        return code_ == StatusCode::Success;
    }

    [[nodiscard]] StatusCode code() const noexcept {
        return code_;
    }

    [[nodiscard]] const std::string& message() const noexcept {
        return message_;
    }

    [[nodiscard]] std::string to_string() const;

    static Status OK() {
        return Status(StatusCode::Success, "OK");
    }

    static Status InvalidArgument(std::string msg) {
        return Status(StatusCode::InvalidArgument, std::move(msg));
    }

    static Status OutOfMemory(std::string msg) {
        return Status(StatusCode::OutOfMemory, std::move(msg));
    }

    static Status CudaError(std::string msg) {
        return Status(StatusCode::CudaError, std::move(msg));
    }

    static Status NumericalFailure(std::string msg) {
        return Status(StatusCode::NumericalFailure, std::move(msg));
    }

    static Status NotImplemented(std::string msg) {
        return Status(StatusCode::NotImplemented, std::move(msg));
    }

    bool operator==(const Status& other) const noexcept {
        return code_ == other.code_;
    }

    bool operator!=(const Status& other) const noexcept {
        return code_ != other.code_;
    }

private:
    StatusCode code_;
    std::string message_;
};

std::string_view status_code_to_string(StatusCode code);
std::ostream& operator<<(std::ostream& os, const Status& status);

} // namespace pipepye
