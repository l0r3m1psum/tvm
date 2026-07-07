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

    if(IS_DIRECTORY ${USE_STRELA_RUNTIME})
        set(STRELA_ROOT_DIR ${USE_STRELA_RUNTIME})
        message(STATUS "Using custom STRELA path: ${STRELA_ROOT_DIR}")
    elseif(DEFINED ENV{STRELA_HOME})
        set(STRELA_ROOT_DIR $ENV{STRELA_HOME})
    endif()

    find_path(STRELA_INCLUDE_DIR NAMES strela.h HINTS ${STRELA_ROOT_DIR} PATH_SUFFIXES include)
    find_library(STRELA_LIBRARY NAMES strela HINTS ${STRELA_ROOT_DIR} PATH_SUFFIXES lib)

    include(FindPackageHandleStandardArgs)
    find_package_handle_standard_args(STRELA DEFAULT_MSG STRELA_INCLUDE_DIR STRELA_LIBRARY)

    if(NOT STRELA_FOUND)
        message(FATAL_ERROR "Could not find STRELA. Please set USE_STRELA_RUNTIME to the installation directory or set the STRELA_HOME environment variable.")
    endif()

    tvm_file_glob(GLOB RUNTIME_STRELA_SRCS src/runtime/extra/contrib/strela/*.cc)
    add_library(tvm_strela_objs OBJECT ${RUNTIME_STRELA_SRCS})

    target_include_directories(tvm_strela_objs PRIVATE ${STRELA_INCLUDE_DIR})
    target_link_libraries(tvm_strela_objs PRIVATE tvm_runtime_extra_defs)

    target_link_libraries(tvm_runtime_extra PRIVATE tvm_strela_objs ${STRELA_LIBRARY})

    # FIXME: This part is a slight adaptation from what happens in CUDA.cmake
    # but for some reason the ext_dev is not registered by TVM-FFI...

    tvm_file_glob(GLOB BACKEND_RUNTIME_STRELA_SRCS src/backend/contrib/strela/runtime/*.cc)

    add_library(tvm_runtime_strela_objs OBJECT ${BACKEND_RUNTIME_STRELA_SRCS})
    target_include_directories(tvm_runtime_strela_objs PRIVATE ${STRELA_INCLUDE_DIR})
    target_link_libraries(tvm_runtime_strela_objs PUBLIC tvm_ffi_header)
    target_compile_definitions(tvm_runtime_strela_objs PRIVATE TVM_RUNTIME_EXPORTS TVM_FFI_EXPORTS)
    set_target_properties(tvm_runtime_strela_objs PROPERTIES POSITION_INDEPENDENT_CODE ON)
    if(TVM_VISIBILITY_FLAG)
      target_compile_options(tvm_runtime_strela_objs PRIVATE "${TVM_VISIBILITY_FLAG}")
    endif()
    add_library(tvm_runtime_strela SHARED $<TARGET_OBJECTS:tvm_runtime_strela_objs>)
    list(APPEND TVM_RUNTIME_BACKEND_LIBS tvm_runtime_strela)
    target_link_libraries(tvm_runtime_strela PUBLIC tvm_runtime ${STRELA_LIBRARY})
    tvm_configure_target_library(tvm_runtime_strela RUNTIME_MODULE)

    add_definitions(-DTVM_GRAPH_EXECUTOR_STRELA)
endif()
