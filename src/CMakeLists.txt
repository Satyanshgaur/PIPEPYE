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
    presolve/presolve_oracle.cpp
    scaling/equilibrator.cpp
    analysis/problem_analyzer.cpp
    analysis/conditioning_proxy.cpp
    pipeline/model_pipeline.cpp
    solver/solution_verifier.cpp
    solver/pdhg_solver_cpu.cpp
    solver/pdhg_solver.cpp
    factorization/dense_lu.cpp
    factorization/sparse_lu.cpp
    factorization/basis_factorization.cpp
    simplex/dual_simplex.cpp
    crossover/pdhg_crossover.cpp
    milp/branch_and_bound.cpp
    workloads/case_a_crude_blending.cpp
    workloads/case_b_multi_period_planning.cpp
    workloads/case_c_refinery_scheduling.cpp
    workloads/case_d_unit_commitment.cpp
    analysis/solver_selector.cpp
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
