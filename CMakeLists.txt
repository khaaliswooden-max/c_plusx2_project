# CMakeLists.txt - MicroGrad++ Build Configuration
# Repository: github.com/khaaliswooden-max/c_plusx2_project

cmake_minimum_required(VERSION 3.16)
project(micrograd_cpp 
    VERSION 0.1.0
    DESCRIPTION "Minimal autodiff tensor library for learning C++"
    LANGUAGES CXX
)

# ============================================================================
# Global Settings
# ============================================================================
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)  # For clangd/IDE support

# Build type default
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type" FORCE)
endif()

# ============================================================================
# Compiler Warnings (Strict)
# ============================================================================
add_library(project_warnings INTERFACE)
target_compile_options(project_warnings INTERFACE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:
        -Wall -Wextra -Wpedantic
        -Wshadow -Wnon-virtual-dtor -Wold-style-cast
        -Wcast-align -Wunused -Woverloaded-virtual
        -Wconversion -Wsign-conversion
        -Wnull-dereference -Wdouble-promotion
        -Wformat=2 -Wimplicit-fallthrough
    >
    $<$<CXX_COMPILER_ID:GNU>:
        -Wmisleading-indentation -Wduplicated-cond
        -Wduplicated-branches -Wlogical-op
    >
    $<$<CXX_COMPILER_ID:MSVC>:
        /W4 /permissive-
    >
)

# ============================================================================
# Sanitizers (Debug builds)
# ============================================================================
option(ENABLE_SANITIZERS "Enable AddressSanitizer and UndefinedBehaviorSanitizer" ON)

add_library(project_sanitizers INTERFACE)
if(ENABLE_SANITIZERS AND CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(project_sanitizers INTERFACE
        $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:
            -fsanitize=address,undefined
            -fno-omit-frame-pointer
        >
    )
    target_link_options(project_sanitizers INTERFACE
        $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:
            -fsanitize=address,undefined
        >
    )
endif()

# ============================================================================
# Main Library (Header-only)
# ============================================================================
add_library(micrograd INTERFACE)
target_include_directories(micrograd INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(micrograd INTERFACE
    project_warnings
)

# ============================================================================
# Benchmarks
# ============================================================================
option(BUILD_BENCHMARKS "Build benchmarks" ON)

if(BUILD_BENCHMARKS)
    # Expression templates benchmark
    add_executable(bench_expr benchmarks/bench_expr.cpp)
    target_link_libraries(bench_expr PRIVATE micrograd)
    target_compile_options(bench_expr PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-O3 -march=native>
        $<$<CXX_COMPILER_ID:MSVC>:/O2 /arch:AVX2>
    )
    
    # SIMD benchmark (Phase 4)
    add_executable(bench_simd benchmarks/bench_simd.cpp)
    target_link_libraries(bench_simd PRIVATE micrograd)
    target_compile_options(bench_simd PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-O3 -march=native -ffast-math>
        $<$<CXX_COMPILER_ID:MSVC>:/O2 /arch:AVX2 /fp:fast>
    )
    
    # OpenMP support for SIMD benchmark
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(bench_simd PRIVATE OpenMP::OpenMP_CXX)
    endif()
endif()

# ============================================================================
# Examples
# ============================================================================
option(BUILD_EXAMPLES "Build examples" ON)

if(BUILD_EXAMPLES)
    add_executable(xor_mlp examples/xor_mlp.cpp)
    target_link_libraries(xor_mlp PRIVATE micrograd)
endif()

# ============================================================================
# Testing
# ============================================================================
option(BUILD_TESTING "Build tests" ON)

if(BUILD_TESTING)
    enable_testing()
    
    # Try to find GoogleTest, or fetch it
    find_package(GTest QUIET)
    if(NOT GTest_FOUND)
        include(FetchContent)
        FetchContent_Declare(
            googletest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG v1.14.0
        )
        # Prevent GoogleTest from overriding our compiler/linker options
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(googletest)
    endif()
    
    # Test executable
    add_executable(test_tensor
        tests/test_main.cpp
        tests/test_tensor.cpp
        tests/test_shape.cpp
        tests/test_ops.cpp
        tests/test_expr.cpp
        tests/test_autograd.cpp
        tests/test_simd.cpp
    )
    target_link_libraries(test_tensor PRIVATE
        micrograd
        project_sanitizers
        GTest::gtest
        GTest::gtest_main
    )
    
    # Enable AVX2 for SIMD tests in Release mode
    target_compile_options(test_tensor PRIVATE
        $<$<AND:$<CONFIG:Release>,$<CXX_COMPILER_ID:GNU,Clang,AppleClang>>:-mavx2>
        $<$<AND:$<CONFIG:Release>,$<CXX_COMPILER_ID:MSVC>>:/arch:AVX2>
    )
    
    # Register tests with CTest
    include(GoogleTest)
    gtest_discover_tests(test_tensor)
endif()

# ============================================================================
# Install (for use as subproject)
# ============================================================================
install(DIRECTORY src/ DESTINATION include/micrograd
    FILES_MATCHING PATTERN "*.hpp"
)

# ============================================================================
# Summary
# ============================================================================
message(STATUS "")
message(STATUS "MicroGrad++ Configuration:")
message(STATUS "  Version:      ${PROJECT_VERSION}")
message(STATUS "  Build type:   ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Compiler:     ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  Sanitizers:   ${ENABLE_SANITIZERS}")
message(STATUS "  Testing:      ${BUILD_TESTING}")
message(STATUS "")
