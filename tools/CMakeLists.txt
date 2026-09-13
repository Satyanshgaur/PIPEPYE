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
