if(NOT PIPEPYE_ENABLE_CUDA)
    message(STATUS "CUDA disabled; skipping benchmarks")
    return()
endif()

add_executable(pipepye_microbench_cuda
    microbench_cuda.cu
)

set(BENCH_DEPS
    pipepye_cuda
    pipepye_core
    ${CMAKE_DL_LIBS}
)
if(TARGET CUDA::nvtx3)
    list(APPEND BENCH_DEPS CUDA::nvtx3)
endif()
if(TARGET CUDA::nvToolsExt)
    list(APPEND BENCH_DEPS CUDA::nvToolsExt)
endif()

target_link_libraries(pipepye_microbench_cuda
    PRIVATE
        ${BENCH_DEPS}
)

set_target_properties(pipepye_microbench_cuda PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
)
