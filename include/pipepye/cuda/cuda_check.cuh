#pragma once

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <sstream>
#include <iostream>

namespace pipepye::cuda {

class CudaException : public std::runtime_error {
public:
    CudaException(cudaError_t code, const char* file, int line, const char* func, const std::string& msg)
        : std::runtime_error(format_message(code, file, line, func, msg)),
          error_code_(code), file_(file), line_(line), function_(func) {}

    [[nodiscard]] cudaError_t error_code() const noexcept { return error_code_; }
    [[nodiscard]] const char* file() const noexcept { return file_; }
    [[nodiscard]] int line() const noexcept { return line_; }
    [[nodiscard]] const char* function() const noexcept { return function_; }

private:
    static std::string format_message(cudaError_t code, const char* file, int line,
                                      const char* func, const std::string& msg) {
        std::ostringstream ss;
        ss << "[CUDA Error] " << file << ":" << line << " in " << func << "()\n"
           << "  Error code : " << static_cast<int>(code) << " (" << cudaGetErrorName(code) << ")\n"
           << "  Description: " << cudaGetErrorString(code) << "\n";
        if (!msg.empty()) {
            ss << "  Context    : " << msg << "\n";
        }
        return ss.str();
    }

    cudaError_t error_code_;
    const char* file_;
    int line_;
    const char* function_;
};

inline void cuda_check_impl(cudaError_t code, const char* file, int line, const char* func, const char* expr) {
    if (code != cudaSuccess) {
        throw CudaException(code, file, line, func, std::string("Failed expression: ") + expr);
    }
}

inline void cuda_check_last_error_impl(const char* file, int line, const char* func) {
    cudaError_t code = cudaGetLastError();
    if (code != cudaSuccess) {
        throw CudaException(code, file, line, func, "Kernel launch error detected by cudaGetLastError()");
    }
}

inline void cuda_sync_and_check_impl(const char* file, int line, const char* func) {
    cudaError_t code = cudaDeviceSynchronize();
    if (code != cudaSuccess) {
        throw CudaException(code, file, line, func, "Asynchronous kernel failure detected by cudaDeviceSynchronize()");
    }
}

} // namespace pipepye::cuda

#define CUDA_CHECK(expr) \
    ::pipepye::cuda::cuda_check_impl((expr), __FILE__, __LINE__, __func__, #expr)

#define CUDA_CHECK_LAST_ERROR() \
    ::pipepye::cuda::cuda_check_last_error_impl(__FILE__, __LINE__, __func__)

#define CUDA_SYNC_AND_CHECK() \
    ::pipepye::cuda::cuda_sync_and_check_impl(__FILE__, __LINE__, __func__)
