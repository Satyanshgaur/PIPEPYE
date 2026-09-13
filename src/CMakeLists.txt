add_library(pipepye_core
    core/version.cpp
    core/status.cpp
    utils/timer.cpp
    utils/logger.cpp
    model/mps_parser.cpp
    sparse/coo_matrix.cpp
    sparse/csr_matrix.cpp
    sparse/csc_matrix.cpp
    sparse/dense_matrix.cpp
    sparse/cpu_ops.cpp
)

target_include_directories(pipepye_core
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_compile_features(pipepye_core PUBLIC cxx_std_20)

if(PIPEPYE_ENABLE_CUDA)
    target_compile_definitions(pipepye_core PUBLIC PIPEPYE_CUDA_ENABLED)
endif()
