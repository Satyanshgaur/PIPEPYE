include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.15.2
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

include(GoogleTest)

# Core C++ unit tests
add_executable(test_core
    test_core_types.cpp
    test_timer.cpp
    test_mps_parser.cpp
)

target_link_libraries(test_core
    PRIVATE
        pipepye_core
        GTest::gtest_main
)

gtest_discover_tests(test_core)

# CUDA unit and integration tests
if(PIPEPYE_ENABLE_CUDA)
    add_executable(test_cuda
        test_cuda_ops.cu
    )

    set(TEST_CUDA_DEPS
        pipepye_cuda
        pipepye_core
        GTest::gtest_main
        ${CMAKE_DL_LIBS}
    )
    if(TARGET CUDA::nvtx3)
        list(APPEND TEST_CUDA_DEPS CUDA::nvtx3)
    endif()
    if(TARGET CUDA::nvToolsExt)
        list(APPEND TEST_CUDA_DEPS CUDA::nvToolsExt)
    endif()

    target_link_libraries(test_cuda
        PRIVATE
            ${TEST_CUDA_DEPS}
    )

    set_target_properties(test_cuda PROPERTIES
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON
    )

    gtest_discover_tests(test_cuda)
endif()
