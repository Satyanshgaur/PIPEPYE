#pragma once

#include <pipepye/core/types.hpp>
#include <string>
#include <string_view>
#include <ostream>
#include <variant>
#include <stdexcept>

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

/// @brief Monadic return type representing either a success value T or an error Status.
template <typename T>
class StatusOr {
public:
    StatusOr(const Status& status) : data_(status) {}
    StatusOr(Status&& status) : data_(std::move(status)) {}
    StatusOr(const T& val) : data_(val) {}
    StatusOr(T&& val) : data_(std::move(val)) {}

    [[nodiscard]] bool is_ok() const noexcept {
        return std::holds_alternative<T>(data_);
    }

    [[nodiscard]] const Status& status() const noexcept {
        if (is_ok()) {
            static const Status kOk = Status::OK();
            return kOk;
        }
        return std::get<Status>(data_);
    }

    [[nodiscard]] const T& value() const & {
        if (!is_ok()) {
            throw std::runtime_error("Attempted to access value of failed StatusOr: " + status().to_string());
        }
        return std::get<T>(data_);
    }

    [[nodiscard]] T& value() & {
        if (!is_ok()) {
            throw std::runtime_error("Attempted to access value of failed StatusOr: " + status().to_string());
        }
        return std::get<T>(data_);
    }

    [[nodiscard]] T&& value() && {
        if (!is_ok()) {
            throw std::runtime_error("Attempted to access value of failed StatusOr: " + status().to_string());
        }
        return std::get<T>(std::move(data_));
    }

    [[nodiscard]] const T* operator->() const {
        return &value();
    }

    [[nodiscard]] T* operator->() {
        return &value();
    }

    [[nodiscard]] const T& operator*() const & {
        return value();
    }

    [[nodiscard]] T& operator*() & {
        return value();
    }

private:
    std::variant<Status, T> data_;
};

} // namespace pipepye

