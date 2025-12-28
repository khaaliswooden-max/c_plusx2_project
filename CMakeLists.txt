# CMakeLists.txt
# MicroGrad++ - Educational C++ Autodiff Tensor Library
cmake_minimum_required(VERSION 3.16)

project(micrograd-cpp
    VERSION 0.1.0
    DESCRIPTION "PhD-level educational C++ autodiff tensor library"
    LANGUAGES CXX
)

# =============================================================================
# Build Options
# =============================================================================
option(MICROGRAD_BUILD_TESTS "Build test executables" ON)
option(MICROGRAD_BUILD_BENCHMARKS "Build benchmark executables" ON)
option(MICROGRAD_ENABLE_SANITIZERS "Enable ASan/UBSan for debug builds" OFF)

# =============================================================================
# C++ Standard & Compiler Settings
# =============================================================================
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# =============================================================================
# Header-Only Library Target
# =============================================================================
add_library(micrograd INTERFACE)
add_library(micrograd::micrograd ALIAS micrograd)

target_include_directories(micrograd INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_compile_features(micrograd INTERFACE cxx_std_20)

# =============================================================================
# Compiler Warnings (applied to targets linking micrograd)
# =============================================================================
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(micrograd INTERFACE
        $<BUILD_INTERFACE:-Wall -Wextra -Wpedantic -Wconversion -Wshadow>
    )
elseif(MSVC)
    target_compile_options(micrograd INTERFACE
        $<BUILD_INTERFACE:/W4 /permissive->
    )
endif()

# =============================================================================
# Sanitizers (Debug builds only)
# =============================================================================
if(MICROGRAD_ENABLE_SANITIZERS AND CMAKE_BUILD_TYPE STREQUAL "Debug")
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
        add_link_options(-fsanitize=address,undefined)
    endif()
endif()

# =============================================================================
# Tests
# =============================================================================
if(MICROGRAD_BUILD_TESTS)
    enable_testing()
    
    # Test: tensor
    add_executable(test_tensor tests/test_tensor.cpp)
    target_link_libraries(test_tensor PRIVATE micrograd)
    add_test(NAME test_tensor COMMAND test_tensor)
    
    # Test: ops (if exists)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_ops.cpp")
        add_executable(test_ops tests/test_ops.cpp)
        target_link_libraries(test_ops PRIVATE micrograd)
        add_test(NAME test_ops COMMAND test_ops)
    endif()
endif()

# =============================================================================
# Benchmarks
# =============================================================================
if(MICROGRAD_BUILD_BENCHMARKS)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/bench_tensor.cpp")
        add_executable(bench_tensor benchmarks/bench_tensor.cpp)
        target_link_libraries(bench_tensor PRIVATE micrograd)
    endif()
endif()

# =============================================================================
# Installation
# =============================================================================
include(GNUInstallDirs)

install(TARGETS micrograd
    EXPORT micrograd-targets
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(DIRECTORY include/micrograd
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT micrograd-targets
    FILE micrograd-config.cmake
    NAMESPACE micrograd::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/micrograd
)
