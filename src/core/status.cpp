#include <pipepye/core/status.hpp>
#include <sstream>

namespace pipepye {

std::string_view status_code_to_string(StatusCode code) {
    switch (code) {
        case StatusCode::Success: return "Success";
        case StatusCode::InvalidArgument: return "InvalidArgument";
        case StatusCode::OutOfMemory: return "OutOfMemory";
        case StatusCode::CudaError: return "CudaError";
        case StatusCode::ConvergenceFailed: return "ConvergenceFailed";
        case StatusCode::NumericalFailure: return "NumericalFailure";
        case StatusCode::NotImplemented: return "NotImplemented";
        case StatusCode::UnknownError: return "UnknownError";
        default: return "InvalidStatusCode";
    }
}

std::string Status::to_string() const {
    std::ostringstream oss;
    oss << "[" << status_code_to_string(code_) << "]";
    if (!message_.empty()) {
        oss << " " << message_;
    }
    return oss.str();
}

std::ostream& operator<<(std::ostream& os, const Status& status) {
    return os << status.to_string();
}

} // namespace pipepye
