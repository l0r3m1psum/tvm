if(USE_STRELA_CODEGEN)
    message(STATUS "Build with STRELA codegen")

    tvm_file_glob(GLOB COMPILER_STRELA_SRCS src/relax/backend/contrib/strela/*.cc)
    list(APPEND COMPILER_SRCS ${COMPILER_STRELA_SRCS})

    tvm_file_glob(GLOB RUNTIME_STRELA_SRCS src/runtime/contrib/strela/*.cc)
    if(NOT USE_STRELA_RUNTIME)
        list(APPEND COMPILER_SRCS ${RUNTIME_STRELA_SRCS})
    endif()
endif()

if(USE_STRELA_RUNTIME)
    message(STATUS "Build with STRELA runtime")


    tvm_file_glob(GLOB RUNTIME_STRELA_SRCS src/runtime/contrib/strela/*.cc)
    list(APPEND RUNTIME_SRCS ${RUNTIME_STRELA_SRCS})

    if(NOT DEFINED STRELA_INCLUDE_DIR)
        message(FATAL_ERROR "STRELA_INCLUDE_DIR is not defined. Please provide the path to the STRELA headers.")
    endif()

    add_definitions(-DTVM_GRAPH_EXECUTOR_STRELA)
    add_compile_options(-I ${STRELA_INCLUDE_DIR})
endif()
