if(USE_STRELA_CODEGEN)
    message(STATUS "Build with STRELA codegen")

    tvm_file_glob(GLOB COMPILER_STRELA_SRCS src/relax/backend/contrib/strela/*.cc)
    list(APPEND COMPILER_SRCS ${COMPILER_STRELA_SRCS})

    # This makes it so that the runtime exists (hence relax.transform.RunCodegen
    # can do its job) but without defining the GRAPH_EXECUTOR which enables the
    # code in the runtime that actually uses the accelerator.
    tvm_file_glob(GLOB RUNTIME_STRELA_SRCS src/runtime/extra/contrib/strela/*.cc)
    if(NOT USE_STRELA_RUNTIME)
        list(APPEND COMPILER_SRCS ${RUNTIME_STRELA_SRCS})
    endif()
endif()

if(USE_STRELA_RUNTIME)
    message(STATUS "Build with STRELA runtime")

    # TODO: this should be handled by something find_library/find_path
    if(NOT DEFINED STRELA_INCLUDE_DIR)
        message(FATAL_ERROR "STRELA_INCLUDE_DIR is not defined. Please provide the path to the STRELA headers.")
    endif()

    if(NOT DEFINED STRELA_LIB_DIR)
        message(FATAL_ERROR "STRELA_LIB_DIR is not defined. Please provide the path to the STRELA library directory.")
    endif()

    tvm_file_glob(GLOB RUNTIME_STRELA_SRCS src/runtime/extra/contrib/strela/*.cc)
    add_library(tvm_strela_objs OBJECT ${RUNTIME_STRELA_SRCS})
    target_include_directories(tvm_strela_objs PRIVATE ${STRELA_INCLUDE_DIR})
    target_link_directories(tvm_strela_objs PRIVATE ${STRELA_LIB_DIR})
    target_link_libraries(tvm_strela_objs PRIVATE tvm_runtime_extra_defs strela)

    target_link_libraries(tvm_runtime_extra PRIVATE tvm_strela_objs)

    add_definitions(-DTVM_GRAPH_EXECUTOR_STRELA)
endif()
