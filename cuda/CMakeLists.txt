if(NOT PIPEPYE_ENABLE_CUDA)
    message(STATUS "CUDA disabled; skipping pipepye_cuda target")
    return()
endif()

# Collect any available NVTX library targets from CUDAToolkit
set(PIPEPYE_CUDA_DEPENDENCIES ${CMAKE_DL_LIBS})
if(TARGET CUDA::nvtx3)
    list(APPEND PIPEPYE_CUDA_DEPENDENCIES CUDA::nvtx3)
endif()
if(TARGET CUDA::nvToolsExt)
    list(APPEND PIPEPYE_CUDA_DEPENDENCIES CUDA::nvToolsExt)
endif()

add_library(pipepye_cuda
    device_info.cu
    kernels/axpy.cu
    kernels/spmv.cu
    kernels/reductions.cu
)

target_include_directories(pipepye_cuda
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)

target_link_libraries(pipepye_cuda
    PUBLIC
        pipepye_core
        ${PIPEPYE_CUDA_DEPENDENCIES}
)

set_target_properties(pipepye_cuda PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
)

target_compile_definitions(pipepye_cuda PUBLIC PIPEPYE_HAS_CUDA=1)

# Standalone CUDA verification and hardware diagnostic executable
add_executable(pipepye_device_probe
    main_device_probe.cu
)

target_link_libraries(pipepye_device_probe
    PRIVATE
        pipepye_cuda
        pipepye_core
        ${PIPEPYE_CUDA_DEPENDENCIES}
)

set_target_properties(pipepye_device_probe PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
)
