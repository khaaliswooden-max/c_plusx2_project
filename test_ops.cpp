// tests/test_ops.cpp
// Unit tests for tensor operations

#include <gtest/gtest.h>
#include <micrograd/micrograd.hpp>
#include <cmath>

using namespace micrograd;

// =============================================================================
// Binary Operations Tests
// =============================================================================

TEST(OpsTest, Addition) {
    auto a = Tensor<float>::from_list({1, 2, 3, 4});
    auto b = Tensor<float>::from_list({5, 6, 7, 8});
    
    auto c = a + b;
    
    EXPECT_FLOAT_EQ(c[0], 6);
    EXPECT_FLOAT_EQ(c[1], 8);
    EXPECT_FLOAT_EQ(c[2], 10);
    EXPECT_FLOAT_EQ(c[3], 12);
}

TEST(OpsTest, Subtraction) {
    auto a = Tensor<float>::from_list({5, 6, 7, 8});
    auto b = Tensor<float>::from_list({1, 2, 3, 4});
    
    auto c = a - b;
    
    EXPECT_FLOAT_EQ(c[0], 4);
    EXPECT_FLOAT_EQ(c[1], 4);
    EXPECT_FLOAT_EQ(c[2], 4);
    EXPECT_FLOAT_EQ(c[3], 4);
}

TEST(OpsTest, Multiplication) {
    auto a = Tensor<float>::from_list({2, 3, 4, 5});
    auto b = Tensor<float>::from_list({1, 2, 3, 4});
    
    auto c = a * b;
    
    EXPECT_FLOAT_EQ(c[0], 2);
    EXPECT_FLOAT_EQ(c[1], 6);
    EXPECT_FLOAT_EQ(c[2], 12);
    EXPECT_FLOAT_EQ(c[3], 20);
}

TEST(OpsTest, Division) {
    auto a = Tensor<float>::from_list({10, 20, 30, 40});
    auto b = Tensor<float>::from_list({2, 4, 5, 8});
    
    auto c = a / b;
    
    EXPECT_FLOAT_EQ(c[0], 5);
    EXPECT_FLOAT_EQ(c[1], 5);
    EXPECT_FLOAT_EQ(c[2], 6);
    EXPECT_FLOAT_EQ(c[3], 5);
}

// =============================================================================
// Scalar Operations Tests
// =============================================================================

TEST(OpsTest, AddScalar) {
    auto a = Tensor<float>::from_list({1, 2, 3});
    
    auto b = a + 10.0f;
    auto c = 10.0f + a;
    
    EXPECT_FLOAT_EQ(b[0], 11);
    EXPECT_FLOAT_EQ(b[2], 13);
    EXPECT_FLOAT_EQ(c[0], 11);  // Commutative
}

TEST(OpsTest, SubtractScalar) {
    auto a = Tensor<float>::from_list({10, 20, 30});
    
    auto b = a - 5.0f;
    auto c = 100.0f - a;
    
    EXPECT_FLOAT_EQ(b[0], 5);
    EXPECT_FLOAT_EQ(c[0], 90);
    EXPECT_FLOAT_EQ(c[2], 70);
}

TEST(OpsTest, MultiplyScalar) {
    auto a = Tensor<float>::from_list({1, 2, 3});
    
    auto b = a * 3.0f;
    auto c = 3.0f * a;
    
    EXPECT_FLOAT_EQ(b[0], 3);
    EXPECT_FLOAT_EQ(b[2], 9);
    EXPECT_FLOAT_EQ(c[0], 3);  // Commutative
}

TEST(OpsTest, DivideScalar) {
    auto a = Tensor<float>::from_list({10, 20, 30});
    
    auto b = a / 10.0f;
    auto c = 60.0f / a;
    
    EXPECT_FLOAT_EQ(b[0], 1);
    EXPECT_FLOAT_EQ(b[2], 3);
    EXPECT_FLOAT_EQ(c[0], 6);
    EXPECT_FLOAT_EQ(c[2], 2);
}

// =============================================================================
// Unary Operations Tests
// =============================================================================

TEST(OpsTest, Negation) {
    auto a = Tensor<float>::from_list({1, -2, 3, -4});
    
    auto b = -a;
    
    EXPECT_FLOAT_EQ(b[0], -1);
    EXPECT_FLOAT_EQ(b[1], 2);
    EXPECT_FLOAT_EQ(b[2], -3);
    EXPECT_FLOAT_EQ(b[3], 4);
}

TEST(OpsTest, Exp) {
    auto a = Tensor<float>::from_list({0, 1, 2});
    
    auto b = exp(a);
    
    EXPECT_NEAR(b[0], 1.0f, 1e-5);
    EXPECT_NEAR(b[1], std::exp(1.0f), 1e-5);
    EXPECT_NEAR(b[2], std::exp(2.0f), 1e-5);
}

TEST(OpsTest, Log) {
    auto a = Tensor<float>::from_list({1, 2.718281828f, 10});
    
    auto b = log(a);
    
    EXPECT_NEAR(b[0], 0.0f, 1e-5);
    EXPECT_NEAR(b[1], 1.0f, 1e-5);
    EXPECT_NEAR(b[2], std::log(10.0f), 1e-5);
}

TEST(OpsTest, Sqrt) {
    auto a = Tensor<float>::from_list({1, 4, 9, 16});
    
    auto b = sqrt(a);
    
    EXPECT_FLOAT_EQ(b[0], 1);
    EXPECT_FLOAT_EQ(b[1], 2);
    EXPECT_FLOAT_EQ(b[2], 3);
    EXPECT_FLOAT_EQ(b[3], 4);
}

TEST(OpsTest, Pow) {
    auto a = Tensor<float>::from_list({2, 3, 4});
    
    auto b = pow(a, 2.0f);
    
    EXPECT_FLOAT_EQ(b[0], 4);
    EXPECT_FLOAT_EQ(b[1], 9);
    EXPECT_FLOAT_EQ(b[2], 16);
}

TEST(OpsTest, Abs) {
    auto a = Tensor<float>::from_list({-1, 2, -3, 4});
    
    auto b = abs(a);
    
    EXPECT_FLOAT_EQ(b[0], 1);
    EXPECT_FLOAT_EQ(b[1], 2);
    EXPECT_FLOAT_EQ(b[2], 3);
    EXPECT_FLOAT_EQ(b[3], 4);
}

// =============================================================================
// Activation Functions Tests
// =============================================================================

TEST(OpsTest, ReLU) {
    auto a = Tensor<float>::from_list({-2, -1, 0, 1, 2});
    
    auto b = relu(a);
    
    EXPECT_FLOAT_EQ(b[0], 0);
    EXPECT_FLOAT_EQ(b[1], 0);
    EXPECT_FLOAT_EQ(b[2], 0);
    EXPECT_FLOAT_EQ(b[3], 1);
    EXPECT_FLOAT_EQ(b[4], 2);
}

TEST(OpsTest, Sigmoid) {
    auto a = Tensor<float>::from_list({-10, 0, 10});
    
    auto b = sigmoid(a);
    
    EXPECT_NEAR(b[0], 0.0f, 1e-4);  // Very close to 0
    EXPECT_FLOAT_EQ(b[1], 0.5f);    // Exactly 0.5
    EXPECT_NEAR(b[2], 1.0f, 1e-4);  // Very close to 1
}

TEST(OpsTest, Tanh) {
    auto a = Tensor<float>::from_list({-10, 0, 10});
    
    auto b = tanh(a);
    
    EXPECT_NEAR(b[0], -1.0f, 1e-4);
    EXPECT_FLOAT_EQ(b[1], 0.0f);
    EXPECT_NEAR(b[2], 1.0f, 1e-4);
}

// =============================================================================
// Compound Assignment Tests
// =============================================================================

TEST(OpsTest, AddAssign) {
    auto a = Tensor<float>::from_list({1, 2, 3});
    auto b = Tensor<float>::from_list({4, 5, 6});
    
    a += b;
    
    EXPECT_FLOAT_EQ(a[0], 5);
    EXPECT_FLOAT_EQ(a[1], 7);
    EXPECT_FLOAT_EQ(a[2], 9);
}

TEST(OpsTest, SubAssign) {
    auto a = Tensor<float>::from_list({10, 20, 30});
    auto b = Tensor<float>::from_list({1, 2, 3});
    
    a -= b;
    
    EXPECT_FLOAT_EQ(a[0], 9);
    EXPECT_FLOAT_EQ(a[1], 18);
    EXPECT_FLOAT_EQ(a[2], 27);
}

TEST(OpsTest, MulAssign) {
    auto a = Tensor<float>::from_list({2, 3, 4});
    auto b = Tensor<float>::from_list({2, 2, 2});
    
    a *= b;
    
    EXPECT_FLOAT_EQ(a[0], 4);
    EXPECT_FLOAT_EQ(a[1], 6);
    EXPECT_FLOAT_EQ(a[2], 8);
}

TEST(OpsTest, MulAssignScalar) {
    auto a = Tensor<float>::from_list({1, 2, 3});
    
    a *= 10.0f;
    
    EXPECT_FLOAT_EQ(a[0], 10);
    EXPECT_FLOAT_EQ(a[1], 20);
    EXPECT_FLOAT_EQ(a[2], 30);
}

// =============================================================================
// Matrix Operations Tests
// =============================================================================

TEST(OpsTest, MatMul) {
    // A: 2x3, B: 3x2 -> C: 2x2
    Tensor<float> a(Shape({2, 3}));
    Tensor<float> b(Shape({3, 2}));
    
    // A = [[1, 2, 3],
    //      [4, 5, 6]]
    a(0, 0) = 1; a(0, 1) = 2; a(0, 2) = 3;
    a(1, 0) = 4; a(1, 1) = 5; a(1, 2) = 6;
    
    // B = [[7, 8],
    //      [9, 10],
    //      [11, 12]]
    b(0, 0) = 7;  b(0, 1) = 8;
    b(1, 0) = 9;  b(1, 1) = 10;
    b(2, 0) = 11; b(2, 1) = 12;
    
    auto c = matmul(a, b);
    
    EXPECT_EQ(c.shape()[0], 2);
    EXPECT_EQ(c.shape()[1], 2);
    
    // C[0,0] = 1*7 + 2*9 + 3*11 = 7 + 18 + 33 = 58
    EXPECT_FLOAT_EQ(c(0, 0), 58);
    // C[0,1] = 1*8 + 2*10 + 3*12 = 8 + 20 + 36 = 64
    EXPECT_FLOAT_EQ(c(0, 1), 64);
    // C[1,0] = 4*7 + 5*9 + 6*11 = 28 + 45 + 66 = 139
    EXPECT_FLOAT_EQ(c(1, 0), 139);
    // C[1,1] = 4*8 + 5*10 + 6*12 = 32 + 50 + 72 = 154
    EXPECT_FLOAT_EQ(c(1, 1), 154);
}

TEST(OpsTest, Transpose) {
    Tensor<float> a(Shape({2, 3}));
    a(0, 0) = 1; a(0, 1) = 2; a(0, 2) = 3;
    a(1, 0) = 4; a(1, 1) = 5; a(1, 2) = 6;
    
    auto b = transpose(a);
    
    EXPECT_EQ(b.shape()[0], 3);
    EXPECT_EQ(b.shape()[1], 2);
    
    EXPECT_FLOAT_EQ(b(0, 0), 1);
    EXPECT_FLOAT_EQ(b(1, 0), 2);
    EXPECT_FLOAT_EQ(b(2, 0), 3);
    EXPECT_FLOAT_EQ(b(0, 1), 4);
    EXPECT_FLOAT_EQ(b(1, 1), 5);
    EXPECT_FLOAT_EQ(b(2, 1), 6);
}

TEST(OpsTest, TransposeTranspose) {
    Tensor<float> a(Shape({3, 4}));
    for (size_t i = 0; i < a.size(); ++i) {
        a[i] = static_cast<float>(i);
    }
    
    auto b = transpose(transpose(a));
    
    EXPECT_EQ(b.shape(), a.shape());
    for (size_t i = 0; i < a.size(); ++i) {
        EXPECT_FLOAT_EQ(b[i], a[i]);
    }
}

// =============================================================================
// Apply Function Tests
// =============================================================================

TEST(OpsTest, ApplyUnary) {
    auto a = Tensor<float>::from_list({1, 2, 3, 4});
    
    auto b = apply(a, [](float x) { return x * x; });
    
    EXPECT_FLOAT_EQ(b[0], 1);
    EXPECT_FLOAT_EQ(b[1], 4);
    EXPECT_FLOAT_EQ(b[2], 9);
    EXPECT_FLOAT_EQ(b[3], 16);
}

TEST(OpsTest, ApplyBinary) {
    auto a = Tensor<float>::from_list({1, 2, 3});
    auto b = Tensor<float>::from_list({4, 5, 6});
    
    auto c = apply(a, b, [](float x, float y) { return x * y + 1; });
    
    EXPECT_FLOAT_EQ(c[0], 5);   // 1*4 + 1
    EXPECT_FLOAT_EQ(c[1], 11);  // 2*5 + 1
    EXPECT_FLOAT_EQ(c[2], 19);  // 3*6 + 1
}

// =============================================================================
// Chained Operations Tests
// =============================================================================

TEST(OpsTest, ChainedArithmetic) {
    auto a = Tensor<float>::from_list({1, 2, 3});
    auto b = Tensor<float>::from_list({4, 5, 6});
    auto c = Tensor<float>::from_list({7, 8, 9});
    
    // (a + b) * c
    auto result = (a + b) * c;
    
    EXPECT_FLOAT_EQ(result[0], 35);   // (1+4)*7
    EXPECT_FLOAT_EQ(result[1], 56);   // (2+5)*8
    EXPECT_FLOAT_EQ(result[2], 81);   // (3+6)*9
}

TEST(OpsTest, NeuralNetLayerSimulation) {
    // Simulate: output = relu(W @ x + b)
    
    // Weight matrix: 2x3
    Tensor<float> W(Shape({2, 3}));
    W(0, 0) = 0.1f; W(0, 1) = 0.2f; W(0, 2) = 0.3f;
    W(1, 0) = 0.4f; W(1, 1) = 0.5f; W(1, 2) = 0.6f;
    
    // Input: 3x1
    Tensor<float> x(Shape({3, 1}));
    x(0, 0) = 1.0f;
    x(1, 0) = 2.0f;
    x(2, 0) = 3.0f;
    
    // Bias: 2x1
    Tensor<float> b(Shape({2, 1}));
    b(0, 0) = -1.0f;
    b(1, 0) = 0.0f;
    
    // Forward pass
    auto z = matmul(W, x) + b;
    auto output = relu(z);
    
    // Check shapes
    EXPECT_EQ(output.shape()[0], 2);
    EXPECT_EQ(output.shape()[1], 1);
    
    // z[0] = 0.1*1 + 0.2*2 + 0.3*3 - 1 = 0.1 + 0.4 + 0.9 - 1 = 0.4
    // z[1] = 0.4*1 + 0.5*2 + 0.6*3 + 0 = 0.4 + 1.0 + 1.8 = 3.2
    EXPECT_NEAR(output(0, 0), 0.4f, 1e-5);
    EXPECT_NEAR(output(1, 0), 3.2f, 1e-5);
}

// Main entry point
int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
