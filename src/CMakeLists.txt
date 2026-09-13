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
    sparse/matrix_generator.cpp
    presolve/postsolve.cpp
    presolve/presolve_context.cpp
    presolve/presolve_pass.cpp
    presolve/presolve_pass_manager.cpp
)

target_include_directories(pipepye_core
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_compile_features(pipepye_core PUBLIC cxx_std_20)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(pipepye_core PUBLIC OpenMP::OpenMP_CXX)
endif()

if(PIPEPYE_ENABLE_CUDA)
    target_compile_definitions(pipepye_core PUBLIC PIPEPYE_CUDA_ENABLED)
endif()
