# tests/CMakeLists.txt
# Test configuration for MicroGrad++

# Phase 1: Simple assertion-based tests
add_executable(test_tensor test_tensor.cpp)
target_link_libraries(test_tensor PRIVATE micrograd)
add_test(NAME test_tensor COMMAND test_tensor)

# Future phases will add more test executables:
# add_executable(test_expr test_expr.cpp)           # PHASE2
# add_executable(test_autograd test_autograd.cpp)   # PHASE3
# add_executable(test_simd test_simd.cpp)           # PHASE4
