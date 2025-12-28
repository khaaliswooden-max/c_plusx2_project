// tests/test_tensor.cpp
// PHASE1: Basic tests for Tensor class - no framework, just assertions
#include <micrograd/tensor.hpp>
#include <cassert>
#include <cmath>
#include <iostream>
#include <sstream>

using namespace micrograd;

// =============================================================================
// Test Utilities
// =============================================================================

#define TEST(name) void test_##name()
#define RUN_TEST(name) do { \
    std::cout << "Running " #name "... "; \
    test_##name(); \
    std::cout << "PASSED" << std::endl; \
} while(0)

template<typename T>
bool approx_equal(T a, T b, T eps = T{1e-5}) {
    return std::abs(a - b) < eps;
}

// =============================================================================
// Shape Tests
// =============================================================================

TEST(shape_basic) {
    Shape s{2, 3, 4};
    assert(s.rank() == 3);
    assert(s[0] == 2);
    assert(s[1] == 3);
    assert(s[2] == 4);
    assert(s.numel() == 24);
}

TEST(shape_empty) {
    Shape s;
    assert(s.rank() == 0);
    assert(s.numel() == 0);
}

TEST(shape_equality) {
    Shape s1{2, 3};
    Shape s2{2, 3};
    Shape s3{3, 2};
    assert(s1 == s2);
    assert(!(s1 == s3));
}

// =============================================================================
// Tensor Construction Tests
// =============================================================================

TEST(tensor_default) {
    Tensor<float> t;
    assert(t.empty());
    assert(t.size() == 0);
}

TEST(tensor_shape) {
    Tensor<float> t({2, 3});
    assert(t.size() == 6);
    assert(t.rank() == 2);
    assert(t.shape()[0] == 2);
    assert(t.shape()[1] == 3);
}

TEST(tensor_fill) {
    Tensor<float> t({2, 3}, 1.5f);
    for (size_t i = 0; i < t.size(); ++i) {
        assert(approx_equal(t[i], 1.5f));
    }
}

TEST(tensor_zeros) {
    auto t = Tensor<float>::zeros({3, 3});
    for (const auto& val : t) {
        assert(approx_equal(val, 0.0f));
    }
}

TEST(tensor_ones) {
    auto t = Tensor<float>::ones({2, 2});
    for (const auto& val : t) {
        assert(approx_equal(val, 1.0f));
    }
}

// =============================================================================
// Rule of 5 Tests
// =============================================================================

TEST(tensor_copy) {
    Tensor<float> a({2, 2}, 3.0f);
    Tensor<float> b = a;  // Copy constructor
    
    // Verify deep copy
    assert(a.data() != b.data());
    assert(a.shape() == b.shape());
    for (size_t i = 0; i < a.size(); ++i) {
        assert(approx_equal(a[i], b[i]));
    }
    
    // Modify original, copy should be unchanged
    a[0] = 999.0f;
    assert(approx_equal(b[0], 3.0f));
}

TEST(tensor_copy_assign) {
    Tensor<float> a({2, 2}, 3.0f);
    Tensor<float> b({3, 3}, 5.0f);
    
    b = a;  // Copy assignment
    
    assert(b.shape() == a.shape());
    assert(b.data() != a.data());
    for (size_t i = 0; i < b.size(); ++i) {
        assert(approx_equal(b[i], 3.0f));
    }
}

TEST(tensor_move) {
    Tensor<float> a({2, 2}, 3.0f);
    float* original_ptr = a.data();
    
    Tensor<float> b = std::move(a);  // Move constructor
    
    assert(b.data() == original_ptr);  // Same memory
    assert(b.size() == 4);
    assert(a.data() == nullptr);  // Moved-from state
}

TEST(tensor_move_assign) {
    Tensor<float> a({2, 2}, 3.0f);
    Tensor<float> b({3, 3}, 5.0f);
    float* original_ptr = a.data();
    
    b = std::move(a);  // Move assignment
    
    assert(b.data() == original_ptr);
    assert(b.size() == 4);
}

TEST(tensor_self_assign) {
    Tensor<float> a({2, 2}, 3.0f);
    a = a;  // Self-assignment
    assert(a.size() == 4);
    assert(approx_equal(a[0], 3.0f));
}

// =============================================================================
// Element Access Tests
// =============================================================================

TEST(tensor_indexing) {
    Tensor<float> t({2, 3});
    for (size_t i = 0; i < t.size(); ++i) {
        t[i] = static_cast<float>(i);
    }
    
    for (size_t i = 0; i < t.size(); ++i) {
        assert(approx_equal(t[i], static_cast<float>(i)));
    }
}

TEST(tensor_multidim_access) {
    Tensor<float> t({2, 3});
    // Row-major: t[row][col] = t[row * 3 + col]
    t.at({0, 0}) = 1.0f;
    t.at({0, 2}) = 2.0f;
    t.at({1, 1}) = 3.0f;
    
    assert(approx_equal(t[0], 1.0f));  // (0,0)
    assert(approx_equal(t[2], 2.0f));  // (0,2)
    assert(approx_equal(t[4], 3.0f));  // (1,1)
}

TEST(tensor_2d_indexing) {
    Tensor<float> t({2, 3});
    t(0, 0) = 1.0f;
    t(0, 2) = 2.0f;
    t(1, 1) = 3.0f;
    
    assert(approx_equal(t(0, 0), 1.0f));
    assert(approx_equal(t(0, 2), 2.0f));
    assert(approx_equal(t(1, 1), 3.0f));
}

TEST(tensor_iteration) {
    Tensor<float> t({2, 2}, 1.0f);
    float sum = 0.0f;
    for (const auto& val : t) {
        sum += val;
    }
    assert(approx_equal(sum, 4.0f));
}

// =============================================================================
// Operation Tests
// =============================================================================

TEST(tensor_add) {
    Tensor<float> a({2, 2}, 1.0f);
    Tensor<float> b({2, 2}, 2.0f);
    
    auto c = a + b;
    
    assert(c.shape() == a.shape());
    for (size_t i = 0; i < c.size(); ++i) {
        assert(approx_equal(c[i], 3.0f));
    }
}

TEST(tensor_sub) {
    Tensor<float> a({2, 2}, 5.0f);
    Tensor<float> b({2, 2}, 2.0f);
    
    auto c = a - b;
    
    for (size_t i = 0; i < c.size(); ++i) {
        assert(approx_equal(c[i], 3.0f));
    }
}

TEST(tensor_mul) {
    Tensor<float> a({2, 2}, 3.0f);
    Tensor<float> b({2, 2}, 4.0f);
    
    auto c = a * b;
    
    for (size_t i = 0; i < c.size(); ++i) {
        assert(approx_equal(c[i], 12.0f));
    }
}

TEST(tensor_div) {
    Tensor<float> a({2, 2}, 12.0f);
    Tensor<float> b({2, 2}, 4.0f);
    
    auto c = a / b;
    
    for (size_t i = 0; i < c.size(); ++i) {
        assert(approx_equal(c[i], 3.0f));
    }
}

TEST(tensor_scalar_add) {
    Tensor<float> a({2, 2}, 1.0f);
    auto b = a + 2.0f;
    
    for (size_t i = 0; i < b.size(); ++i) {
        assert(approx_equal(b[i], 3.0f));
    }
}

TEST(tensor_scalar_mul) {
    Tensor<float> a({2, 2}, 3.0f);
    auto b = a * 2.0f;
    
    for (size_t i = 0; i < b.size(); ++i) {
        assert(approx_equal(b[i], 6.0f));
    }
}

TEST(tensor_chained_ops) {
    // Note: In PHASE1, this creates intermediate tensors
    // PHASE2 will optimize this with expression templates
    Tensor<float> a({2, 2}, 1.0f);
    Tensor<float> b({2, 2}, 2.0f);
    Tensor<float> c({2, 2}, 3.0f);
    
    auto result = (a + b) * c;
    
    // (1 + 2) * 3 = 9
    for (size_t i = 0; i < result.size(); ++i) {
        assert(approx_equal(result[i], 9.0f));
    }
}

// =============================================================================
// Reduction Tests
// =============================================================================

TEST(tensor_sum) {
    auto t = Tensor<float>::ones({3, 3});
    assert(approx_equal(t.sum(), 9.0f));
}

TEST(tensor_mean) {
    Tensor<float> t({2, 2});
    t[0] = 1.0f; t[1] = 2.0f; t[2] = 3.0f; t[3] = 4.0f;
    assert(approx_equal(t.mean(), 2.5f));
}

// =============================================================================
// Factory Method Tests
// =============================================================================

TEST(tensor_from_list) {
    auto t = Tensor<float>::from_list({1.0f, 2.0f, 3.0f, 4.0f});
    assert(t.size() == 4);
    assert(approx_equal(t[0], 1.0f));
    assert(approx_equal(t[3], 4.0f));
}

// =============================================================================
// Edge Cases
// =============================================================================

TEST(tensor_single_element) {
    Tensor<float> t({1}, 42.0f);
    assert(t.size() == 1);
    assert(approx_equal(t[0], 42.0f));
}

TEST(tensor_large) {
    // Test larger tensor to verify no memory issues
    Tensor<float> t({100, 100, 100});
    assert(t.size() == 1000000);
    t.fill(1.0f);
    assert(approx_equal(t[999999], 1.0f));
}

TEST(tensor_print) {
    Tensor<float> t({2, 2}, 1.5f);
    std::stringstream ss;
    ss << t;
    std::string output = ss.str();
    assert(output.find("Tensor") != std::string::npos);
    assert(output.find("1.5") != std::string::npos);
}

// =============================================================================
// Main
// =============================================================================

int main() {
    std::cout << "=== MicroGrad++ Phase 1 Tests ===" << std::endl;
    
    // Shape tests
    RUN_TEST(shape_basic);
    RUN_TEST(shape_empty);
    RUN_TEST(shape_equality);
    
    // Construction tests
    RUN_TEST(tensor_default);
    RUN_TEST(tensor_shape);
    RUN_TEST(tensor_fill);
    RUN_TEST(tensor_zeros);
    RUN_TEST(tensor_ones);
    
    // Rule of 5 tests
    RUN_TEST(tensor_copy);
    RUN_TEST(tensor_copy_assign);
    RUN_TEST(tensor_move);
    RUN_TEST(tensor_move_assign);
    RUN_TEST(tensor_self_assign);
    
    // Access tests
    RUN_TEST(tensor_indexing);
    RUN_TEST(tensor_multidim_access);
    RUN_TEST(tensor_2d_indexing);
    RUN_TEST(tensor_iteration);
    
    // Operation tests
    RUN_TEST(tensor_add);
    RUN_TEST(tensor_sub);
    RUN_TEST(tensor_mul);
    RUN_TEST(tensor_div);
    RUN_TEST(tensor_scalar_add);
    RUN_TEST(tensor_scalar_mul);
    RUN_TEST(tensor_chained_ops);
    
    // Reduction tests
    RUN_TEST(tensor_sum);
    RUN_TEST(tensor_mean);
    
    // Factory tests
    RUN_TEST(tensor_from_list);
    
    // Edge cases
    RUN_TEST(tensor_single_element);
    RUN_TEST(tensor_large);
    RUN_TEST(tensor_print);
    
    std::cout << "\n=== All Phase 1 tests PASSED ===" << std::endl;
    return 0;
}

