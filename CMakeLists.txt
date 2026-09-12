cmake_minimum_required(VERSION 3.24)

project(PipePye
    VERSION 0.1.0
    DESCRIPTION "High-Performance Sovereign Optimization Solver for Indian Industry"
    LANGUAGES CXX
)

# Standard C++20 requirement
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# User-configurable Options
option(PIPEPYE_ENABLE_CUDA "Enable CUDA acceleration" ON)
option(PIPEPYE_BUILD_TESTS "Build test suite with GoogleTest" ON)
option(PIPEPYE_BUILD_BENCHMARKS "Build micro-benchmarking harness" ON)

# Output directory hygiene
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Compiler warnings for C++ host code
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(
        $<$<COMPILE_LANGUAGE:CXX>:-Wall>
        $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
        $<$<COMPILE_LANGUAGE:CXX>:-Wpedantic>
        $<$<COMPILE_LANGUAGE:CXX>:-Wno-unused-parameter>
    )
endif()

# CUDA Detection & Configuration
if(PIPEPYE_ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        # Default to Ampere sm_86 (RTX 3050 Laptop GPU)
        if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES EQUAL 75)
            set(CMAKE_CUDA_ARCHITECTURES 86 CACHE STRING "CUDA architectures" FORCE)
        endif()

        enable_language(CUDA)
        find_package(CUDAToolkit QUIET)

        set(CMAKE_CUDA_STANDARD 20)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        set(CMAKE_CUDA_EXTENSIONS OFF)

        # Allow host code in CUDA to use modern flags
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wall,-Wextra")

        message(STATUS "PipePye: CUDA enabled using ${CMAKE_CUDA_COMPILER}")
        message(STATUS "PipePye: Targeting CUDA architecture(s): ${CMAKE_CUDA_ARCHITECTURES}")
    else()
        message(WARNING "PipePye: CUDA requested but nvcc compiler not found. Falling back to CPU-only build.")
        set(PIPEPYE_ENABLE_CUDA OFF)
    endif()
endif()

# Subdirectories
add_subdirectory(src)
add_subdirectory(cuda)

if(PIPEPYE_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(PIPEPYE_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()
