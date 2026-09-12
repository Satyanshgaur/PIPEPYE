if(NOT PIPEPYE_ENABLE_CUDA)
    message(STATUS "CUDA disabled; skipping benchmarks")
    return()
endif()

add_executable(pipepye_microbench_cuda
    microbench_cuda.cu
)

target_link_libraries(pipepye_microbench_cuda
    PRIVATE
        pipepye_cuda
        pipepye_core
        ${CMAKE_DL_LIBS}
)

set_target_properties(pipepye_microbench_cuda PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
)
