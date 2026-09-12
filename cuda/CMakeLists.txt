if(NOT PIPEPYE_ENABLE_CUDA)
    message(STATUS "CUDA disabled; skipping pipepye_cuda target")
    return()
endif()

add_library(pipepye_cuda
    device_info.cu
    kernels/axpy.cu
)

target_include_directories(pipepye_cuda
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)

target_link_libraries(pipepye_cuda
    PUBLIC
        pipepye_core
        ${CMAKE_DL_LIBS}
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
        ${CMAKE_DL_LIBS}
)

set_target_properties(pipepye_device_probe PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
)
