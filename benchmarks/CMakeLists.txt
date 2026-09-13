# Standalone CPU Sparse Linear Algebra & Vector Primitives Benchmark
add_executable(pipepye_bench_sparse_cpu
    bench_sparse_cpu.cpp
)

target_link_libraries(pipepye_bench_sparse_cpu
    PRIVATE
        pipepye_core
)

set_target_properties(pipepye_bench_sparse_cpu PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

# Automated Before/After Presolve & Scaling Benchmark
add_executable(pipepye_bench_presolve_scaling
    bench_presolve_scaling.cpp
)
target_link_libraries(pipepye_bench_presolve_scaling
    PRIVATE
        pipepye_core
)
set_target_properties(pipepye_bench_presolve_scaling PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

# Numerical Robustness Benchmark (PDHG First-Order Simulation)
add_executable(pipepye_bench_numerical_robustness
    bench_numerical_robustness.cpp
)
target_link_libraries(pipepye_bench_numerical_robustness
    PRIVATE
        pipepye_core
)
set_target_properties(pipepye_bench_numerical_robustness PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

if(PIPEPYE_ENABLE_CUDA)
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

    add_executable(pipepye_bench_spmv_cuda
        bench_spmv_cuda.cu
    )

    target_link_libraries(pipepye_bench_spmv_cuda
        PRIVATE
            ${BENCH_DEPS}
    )

    set_target_properties(pipepye_bench_spmv_cuda PROPERTIES
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON
    )

    add_executable(pipepye_run_all_benchmarks
        run_all_benchmarks.cu
    )

    target_link_libraries(pipepye_run_all_benchmarks
        PRIVATE
            ${BENCH_DEPS}
    )

    set_target_properties(pipepye_run_all_benchmarks PROPERTIES
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON
    )

    add_executable(pipepye_bench_pdhg
        bench_pdhg.cu
    )

    target_link_libraries(pipepye_bench_pdhg
        PRIVATE
            ${BENCH_DEPS}
    )

    set_target_properties(pipepye_bench_pdhg PROPERTIES
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON
    )
endif()

