add_executable(pipepye_inspect
    pipepye_inspect.cpp
)

target_link_libraries(pipepye_inspect
    PRIVATE
        pipepye_core
)

set_target_properties(pipepye_inspect PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

add_executable(pipepye_dashboard_runner
    pipepye_dashboard_runner.cpp
)

if(PIPEPYE_ENABLE_CUDA)
    set(RUNNER_DEPS
        pipepye_cuda
        pipepye_core
        ${CMAKE_DL_LIBS}
    )
    if(TARGET CUDA::nvtx3)
        list(APPEND RUNNER_DEPS CUDA::nvtx3)
    endif()
    if(TARGET CUDA::nvToolsExt)
        list(APPEND RUNNER_DEPS CUDA::nvToolsExt)
    endif()
    target_link_libraries(pipepye_dashboard_runner
        PRIVATE
            ${RUNNER_DEPS}
    )
else()
    target_link_libraries(pipepye_dashboard_runner
        PRIVATE
            pipepye_core
    )
endif()

set_target_properties(pipepye_dashboard_runner PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

add_executable(run_public_corpora
    run_public_corpora.cpp
)

target_link_libraries(run_public_corpora
    PRIVATE
        pipepye_core
)

set_target_properties(run_public_corpora PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)
