// tests/test_ops.cpp
// Unit tests for SIMD operations

#include <gtest/gtest.h>
#include <simd/simd.hpp>
#include <cmath>
#include <vector>

using namespace micrograd::simd;

class SimdOpsTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Allocate aligned buffers
        a_ = static_cast<float*>(aligned_alloc(N * sizeof(float)));
        b_ = static_cast<float*>(aligned_alloc(N * sizeof(float)));
        c_ = static_cast<float*>(aligned_alloc(N * sizeof(float)));
        
        // Initialize with test data
        for (size_t i = 0; i < N; ++i) {
            a_[i] = static_cast<float>(i);
            b_[i] = static_cast<float>(i * 2);
        }
    }
    
    void TearDown() override {
        aligned_free(a_);
        aligned_free(b_);
        aligned_free(c_);
    }
    
    static constexpr size_t N = 256;
    float* a_;
    float* b_;
    float* c_;
};

TEST_F(SimdOpsTest, Add) {
    add_f32(c_, a_, b_, N);
    
    for (size_t i = 0; i < N; ++i) {
        EXPECT_FLOAT_EQ(c_[i], a_[i] + b_[i]) << "Mismatch at index " << i;
    }
}

TEST_F(SimdOpsTest, Sub) {
    sub_f32(c_, a_, b_, N);
    
    for (size_t i = 0; i < N; ++i) {
        EXPECT_FLOAT_EQ(c_[i], a_[i] - b_[i]) << "Mismatch at index " << i;
    }
}

TEST_F(SimdOpsTest, Mul) {
    mul_f32(c_, a_, b_, N);
    
    for (size_t i = 0; i < N; ++i) {
        EXPECT_FLOAT_EQ(c_[i], a_[i] * b_[i]) << "Mismatch at index " << i;
    }
}

TEST_F(SimdOpsTest, Scale) {
    scale_f32(c_, a_, 2.5f, N);
    
    for (size_t i = 0; i < N; ++i) {
        EXPECT_FLOAT_EQ(c_[i], a_[i] * 2.5f) << "Mismatch at index " << i;
    }
}

TEST_F(SimdOpsTest, ReLU) {
    // Mix of positive and negative values
    for (size_t i = 0; i < N; ++i) {
        a_[i] = static_cast<float>(i) - static_cast<float>(N / 2);
    }
    
    relu_f32(c_, a_, N);
    
    for (size_t i = 0; i < N; ++i) {
        float expected = std::max(0.0f, a_[i]);
        EXPECT_FLOAT_EQ(c_[i], expected) << "Mismatch at index " << i;
    }
}

TEST_F(SimdOpsTest, Sum) {
    float result = sum_f32(a_, N);
    
    float expected = 0.0f;
    for (size_t i = 0; i < N; ++i) {
        expected += a_[i];
    }
    
    EXPECT_NEAR(result, expected, 1e-3f);
}

TEST_F(SimdOpsTest, DotProduct) {
    float result = dot_f32(a_, b_, N);
    
    float expected = 0.0f;
    for (size_t i = 0; i < N; ++i) {
        expected += a_[i] * b_[i];
    }
    
    EXPECT_NEAR(result, expected, 1.0f);  // Allow some floating point error
}

