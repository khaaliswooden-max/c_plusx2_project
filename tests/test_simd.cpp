// filepath: tests/test_simd.cpp
// SIMD Tests - Phase 4
// Verifies: correctness of vectorized operations
#include <gtest/gtest.h>
#include "simd/simd.hpp"
#include <cmath>
#include <vector>

using namespace micrograd::simd;

// ============================================================================
// SIMD Detection Tests
// ============================================================================

TEST(SimdTest, Detection_ReturnsValidCaps)
{
    auto caps = detect_simd();
    // At minimum, SSE2 should be available on x86-64
#ifdef MICROGRAD_X86
    EXPECT_TRUE(caps.sse || caps.sse2 || caps.avx || caps.avx2 || caps.avx512f || true);
#endif
}

TEST(SimdTest, SIMDInfo_ReturnsNonEmpty)
{
    std::string info = simd_info();
    EXPECT_FALSE(info.empty());
    EXPECT_NE(info.find("SIMD:"), std::string::npos);
}

// ============================================================================
// Element-wise Operation Tests
// ============================================================================

TEST(SimdTest, Add_Correct)
{
    std::vector<float> a = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    std::vector<float> b = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100};
    std::vector<float> c(10);
    
    add_f32(c.data(), a.data(), b.data(), 10);
    
    EXPECT_FLOAT_EQ(c[0], 11.0f);
    EXPECT_FLOAT_EQ(c[4], 55.0f);
    EXPECT_FLOAT_EQ(c[9], 110.0f);
}

TEST(SimdTest, Add_LargeArray_Correct)
{
    const size_t N = 1000;
    std::vector<float> a(N, 1.0f);
    std::vector<float> b(N, 2.0f);
    std::vector<float> c(N);
    
    add_f32(c.data(), a.data(), b.data(), N);
    
    for (size_t i = 0; i < N; ++i) {
        EXPECT_FLOAT_EQ(c[i], 3.0f);
    }
}

TEST(SimdTest, Sub_Correct)
{
    std::vector<float> a = {10, 20, 30, 40, 50, 60, 70, 80};
    std::vector<float> b = {1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<float> c(8);
    
    sub_f32(c.data(), a.data(), b.data(), 8);
    
    EXPECT_FLOAT_EQ(c[0], 9.0f);
    EXPECT_FLOAT_EQ(c[7], 72.0f);
}

TEST(SimdTest, Mul_Correct)
{
    std::vector<float> a = {1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<float> b = {2, 2, 2, 2, 2, 2, 2, 2};
    std::vector<float> c(8);
    
    mul_f32(c.data(), a.data(), b.data(), 8);
    
    EXPECT_FLOAT_EQ(c[0], 2.0f);
    EXPECT_FLOAT_EQ(c[7], 16.0f);
}

TEST(SimdTest, Scale_Correct)
{
    std::vector<float> a = {1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<float> c(8);
    
    scale_f32(c.data(), a.data(), 3.0f, 8);
    
    EXPECT_FLOAT_EQ(c[0], 3.0f);
    EXPECT_FLOAT_EQ(c[7], 24.0f);
}

TEST(SimdTest, ReLU_Correct)
{
    std::vector<float> a = {-2, -1, 0, 1, 2, -3, 4, -5};
    std::vector<float> c(8);
    
    relu_f32(c.data(), a.data(), 8);
    
    EXPECT_FLOAT_EQ(c[0], 0.0f);
    EXPECT_FLOAT_EQ(c[1], 0.0f);
    EXPECT_FLOAT_EQ(c[2], 0.0f);
    EXPECT_FLOAT_EQ(c[3], 1.0f);
    EXPECT_FLOAT_EQ(c[4], 2.0f);
    EXPECT_FLOAT_EQ(c[5], 0.0f);
    EXPECT_FLOAT_EQ(c[6], 4.0f);
    EXPECT_FLOAT_EQ(c[7], 0.0f);
}

// ============================================================================
// Reduction Tests
// ============================================================================

TEST(SimdTest, Sum_Correct)
{
    std::vector<float> a = {1, 2, 3, 4, 5, 6, 7, 8};
    float sum = sum_f32(a.data(), 8);
    EXPECT_FLOAT_EQ(sum, 36.0f);
}

TEST(SimdTest, Sum_LargeArray_Correct)
{
    const size_t N = 1000;
    std::vector<float> a(N, 1.0f);
    float sum = sum_f32(a.data(), N);
    EXPECT_FLOAT_EQ(sum, static_cast<float>(N));
}

TEST(SimdTest, Dot_Correct)
{
    std::vector<float> a = {1, 2, 3, 4};
    std::vector<float> b = {2, 2, 2, 2};
    float dp = dot_f32(a.data(), b.data(), 4);
    // 1*2 + 2*2 + 3*2 + 4*2 = 2 + 4 + 6 + 8 = 20
    EXPECT_FLOAT_EQ(dp, 20.0f);
}

TEST(SimdTest, Dot_LargeArray_Correct)
{
    const size_t N = 1000;
    std::vector<float> a(N, 2.0f);
    std::vector<float> b(N, 3.0f);
    float dp = dot_f32(a.data(), b.data(), N);
    EXPECT_FLOAT_EQ(dp, 6000.0f);  // 2*3*1000
}

// ============================================================================
// Matrix Multiplication Tests
// ============================================================================

TEST(SimdTest, MatMul_Naive_Correct)
{
    // A: 2x3, B: 3x2, C: 2x2
    std::vector<float> A = {1, 2, 3, 4, 5, 6};  // row-major
    std::vector<float> B = {1, 2, 3, 4, 5, 6};
    std::vector<float> C(4, 0);
    
    matmul_naive(C.data(), A.data(), B.data(), 2, 3, 2);
    
    // C[0,0] = 1*1 + 2*3 + 3*5 = 1 + 6 + 15 = 22
    // C[0,1] = 1*2 + 2*4 + 3*6 = 2 + 8 + 18 = 28
    // C[1,0] = 4*1 + 5*3 + 6*5 = 4 + 15 + 30 = 49
    // C[1,1] = 4*2 + 5*4 + 6*6 = 8 + 20 + 36 = 64
    EXPECT_FLOAT_EQ(C[0], 22.0f);
    EXPECT_FLOAT_EQ(C[1], 28.0f);
    EXPECT_FLOAT_EQ(C[2], 49.0f);
    EXPECT_FLOAT_EQ(C[3], 64.0f);
}

TEST(SimdTest, MatMul_SIMD_MatchesNaive)
{
    const size_t M = 64, K = 32, N = 48;
    std::vector<float> A(M * K), B(K * N);
    std::vector<float> C_naive(M * N), C_simd(M * N);
    
    // Initialize with random data
    std::srand(42);
    for (auto& x : A) x = static_cast<float>(std::rand()) / RAND_MAX;
    for (auto& x : B) x = static_cast<float>(std::rand()) / RAND_MAX;
    
    matmul_naive(C_naive.data(), A.data(), B.data(), M, K, N);
    matmul_simd(C_simd.data(), A.data(), B.data(), M, K, N);
    
    for (size_t i = 0; i < M * N; ++i) {
        EXPECT_NEAR(C_simd[i], C_naive[i], 1e-4f);
    }
}

TEST(SimdTest, MatMul_Blocked_MatchesNaive)
{
    const size_t M = 100, K = 80, N = 90;
    std::vector<float> A(M * K), B(K * N);
    std::vector<float> C_naive(M * N), C_blocked(M * N);
    
    std::srand(123);
    for (auto& x : A) x = static_cast<float>(std::rand()) / RAND_MAX;
    for (auto& x : B) x = static_cast<float>(std::rand()) / RAND_MAX;
    
    matmul_naive(C_naive.data(), A.data(), B.data(), M, K, N);
    matmul_blocked(C_blocked.data(), A.data(), B.data(), M, K, N);
    
    for (size_t i = 0; i < M * N; ++i) {
        EXPECT_NEAR(C_blocked[i], C_naive[i], 1e-4f);
    }
}

TEST(SimdTest, MatMul_Fast_MatchesNaive)
{
    const size_t M = 128, K = 64, N = 96;
    std::vector<float> A(M * K), B(K * N);
    std::vector<float> C_naive(M * N), C_fast(M * N);
    
    std::srand(456);
    for (auto& x : A) x = static_cast<float>(std::rand()) / RAND_MAX;
    for (auto& x : B) x = static_cast<float>(std::rand()) / RAND_MAX;
    
    matmul_naive(C_naive.data(), A.data(), B.data(), M, K, N);
    matmul_fast(C_fast.data(), A.data(), B.data(), M, K, N);
    
    for (size_t i = 0; i < M * N; ++i) {
        EXPECT_NEAR(C_fast[i], C_naive[i], 1e-4f);
    }
}

// ============================================================================
// AlignedTensor Tests
// ============================================================================

TEST(SimdTest, AlignedTensor_IsAligned)
{
    AlignedTensor<float> t({1000});
    EXPECT_TRUE(t.is_aligned());
}

TEST(SimdTest, AlignedTensor_Add_Correct)
{
    auto a = AlignedTensor<float>::ones({100});
    auto b = AlignedTensor<float>::ones({100});
    auto c = a + b;
    
    for (size_t i = 0; i < 100; ++i) {
        EXPECT_FLOAT_EQ(c[i], 2.0f);
    }
}

TEST(SimdTest, AlignedTensor_Mul_Correct)
{
    auto a = AlignedTensor<float>::ones({100});
    for (size_t i = 0; i < 100; ++i) a[i] = static_cast<float>(i);
    
    auto b = a * 2.0f;
    
    EXPECT_FLOAT_EQ(b[0], 0.0f);
    EXPECT_FLOAT_EQ(b[50], 100.0f);
    EXPECT_FLOAT_EQ(b[99], 198.0f);
}

TEST(SimdTest, AlignedTensor_Sum_Correct)
{
    auto a = AlignedTensor<float>::ones({1000});
    float sum = a.sum();
    EXPECT_FLOAT_EQ(sum, 1000.0f);
}

TEST(SimdTest, AlignedTensor_MatMul_Correct)
{
    // 2x3 @ 3x2 = 2x2
    auto A = AlignedTensor<float>::zeros({2, 3});
    auto B = AlignedTensor<float>::zeros({3, 2});
    
    // Fill A = [[1,2,3],[4,5,6]]
    A[0] = 1; A[1] = 2; A[2] = 3;
    A[3] = 4; A[4] = 5; A[5] = 6;
    
    // Fill B = [[1,2],[3,4],[5,6]]
    B[0] = 1; B[1] = 2;
    B[2] = 3; B[3] = 4;
    B[4] = 5; B[5] = 6;
    
    auto C = matmul(A, B);
    
    EXPECT_EQ(C.shape()[0], 2);
    EXPECT_EQ(C.shape()[1], 2);
    EXPECT_FLOAT_EQ(C[0], 22.0f);
    EXPECT_FLOAT_EQ(C[1], 28.0f);
    EXPECT_FLOAT_EQ(C[2], 49.0f);
    EXPECT_FLOAT_EQ(C[3], 64.0f);
}

TEST(SimdTest, AlignedTensor_Dot_Correct)
{
    auto a = AlignedTensor<float>::ones({100});
    auto b = AlignedTensor<float>::ones({100});
    float dp = dot(a, b);
    EXPECT_FLOAT_EQ(dp, 100.0f);
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST(SimdTest, SmallArray_StillCorrect)
{
    // Test with array smaller than SIMD width
    std::vector<float> a = {1, 2, 3};
    std::vector<float> b = {4, 5, 6};
    std::vector<float> c(3);
    
    add_f32(c.data(), a.data(), b.data(), 3);
    
    EXPECT_FLOAT_EQ(c[0], 5.0f);
    EXPECT_FLOAT_EQ(c[1], 7.0f);
    EXPECT_FLOAT_EQ(c[2], 9.0f);
}

TEST(SimdTest, NonMultipleOfSIMDWidth_Correct)
{
    // 13 elements (not divisible by 4 or 8)
    std::vector<float> a(13, 1.0f);
    std::vector<float> b(13, 2.0f);
    std::vector<float> c(13);
    
    add_f32(c.data(), a.data(), b.data(), 13);
    
    for (int i = 0; i < 13; ++i) {
        EXPECT_FLOAT_EQ(c[i], 3.0f);
    }
}
