// filepath: src/simd/simd_ops.hpp
// SIMD Vectorized Operations
// Phase 4: AVX2/SSE Optimized Tensor Operations
#pragma once

#include "simd_detect.hpp"
#include <cstddef>
#include <cmath>
#include <algorithm>

namespace micrograd {
namespace simd {

// ============================================================================
// Vectorized Element-wise Operations
// ============================================================================

/// @brief Vectorized addition: dst = a + b
/// @param dst Output array (must be aligned)
/// @param a First input array
/// @param b Second input array  
/// @param n Number of elements
/// 
/// Uses AVX2 to process 8 floats per iteration.
/// Falls back to scalar for non-AVX2 or unaligned data.
inline void add_f32(float* dst, const float* a, const float* b, size_t n)
{
#ifdef MICROGRAD_AVX2
    size_t i = 0;
    
    // Process 8 floats at a time with AVX2
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 vc = _mm256_add_ps(va, vb);
        _mm256_storeu_ps(dst + i, vc);
    }
    
    // Handle remaining elements
    for (; i < n; ++i) {
        dst[i] = a[i] + b[i];
    }
#elif defined(MICROGRAD_SSE)
    size_t i = 0;
    
    // Process 4 floats at a time with SSE
    for (; i + 4 <= n; i += 4) {
        __m128 va = _mm_loadu_ps(a + i);
        __m128 vb = _mm_loadu_ps(b + i);
        __m128 vc = _mm_add_ps(va, vb);
        _mm_storeu_ps(dst + i, vc);
    }
    
    for (; i < n; ++i) {
        dst[i] = a[i] + b[i];
    }
#else
    // Scalar fallback
    for (size_t i = 0; i < n; ++i) {
        dst[i] = a[i] + b[i];
    }
#endif
}

/// @brief Vectorized subtraction: dst = a - b
inline void sub_f32(float* dst, const float* a, const float* b, size_t n)
{
#ifdef MICROGRAD_AVX2
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 vc = _mm256_sub_ps(va, vb);
        _mm256_storeu_ps(dst + i, vc);
    }
    for (; i < n; ++i) {
        dst[i] = a[i] - b[i];
    }
#elif defined(MICROGRAD_SSE)
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m128 va = _mm_loadu_ps(a + i);
        __m128 vb = _mm_loadu_ps(b + i);
        __m128 vc = _mm_sub_ps(va, vb);
        _mm_storeu_ps(dst + i, vc);
    }
    for (; i < n; ++i) {
        dst[i] = a[i] - b[i];
    }
#else
    for (size_t i = 0; i < n; ++i) {
        dst[i] = a[i] - b[i];
    }
#endif
}

/// @brief Vectorized multiplication: dst = a * b
inline void mul_f32(float* dst, const float* a, const float* b, size_t n)
{
#ifdef MICROGRAD_AVX2
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 vc = _mm256_mul_ps(va, vb);
        _mm256_storeu_ps(dst + i, vc);
    }
    for (; i < n; ++i) {
        dst[i] = a[i] * b[i];
    }
#elif defined(MICROGRAD_SSE)
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m128 va = _mm_loadu_ps(a + i);
        __m128 vb = _mm_loadu_ps(b + i);
        __m128 vc = _mm_mul_ps(va, vb);
        _mm_storeu_ps(dst + i, vc);
    }
    for (; i < n; ++i) {
        dst[i] = a[i] * b[i];
    }
#else
    for (size_t i = 0; i < n; ++i) {
        dst[i] = a[i] * b[i];
    }
#endif
}

/// @brief Vectorized division: dst = a / b
inline void div_f32(float* dst, const float* a, const float* b, size_t n)
{
#ifdef MICROGRAD_AVX2
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 vc = _mm256_div_ps(va, vb);
        _mm256_storeu_ps(dst + i, vc);
    }
    for (; i < n; ++i) {
        dst[i] = a[i] / b[i];
    }
#elif defined(MICROGRAD_SSE)
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m128 va = _mm_loadu_ps(a + i);
        __m128 vb = _mm_loadu_ps(b + i);
        __m128 vc = _mm_div_ps(va, vb);
        _mm_storeu_ps(dst + i, vc);
    }
    for (; i < n; ++i) {
        dst[i] = a[i] / b[i];
    }
#else
    for (size_t i = 0; i < n; ++i) {
        dst[i] = a[i] / b[i];
    }
#endif
}

/// @brief Vectorized scalar multiply: dst = a * scalar
inline void scale_f32(float* dst, const float* a, float scalar, size_t n)
{
#ifdef MICROGRAD_AVX2
    size_t i = 0;
    __m256 vs = _mm256_set1_ps(scalar);
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vc = _mm256_mul_ps(va, vs);
        _mm256_storeu_ps(dst + i, vc);
    }
    for (; i < n; ++i) {
        dst[i] = a[i] * scalar;
    }
#elif defined(MICROGRAD_SSE)
    size_t i = 0;
    __m128 vs = _mm_set1_ps(scalar);
    for (; i + 4 <= n; i += 4) {
        __m128 va = _mm_loadu_ps(a + i);
        __m128 vc = _mm_mul_ps(va, vs);
        _mm_storeu_ps(dst + i, vc);
    }
    for (; i < n; ++i) {
        dst[i] = a[i] * scalar;
    }
#else
    for (size_t i = 0; i < n; ++i) {
        dst[i] = a[i] * scalar;
    }
#endif
}

// ============================================================================
// Vectorized Activation Functions
// ============================================================================

/// @brief Vectorized ReLU: dst = max(0, a)
inline void relu_f32(float* dst, const float* a, size_t n)
{
#ifdef MICROGRAD_AVX2
    size_t i = 0;
    __m256 zero = _mm256_setzero_ps();
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vc = _mm256_max_ps(zero, va);
        _mm256_storeu_ps(dst + i, vc);
    }
    for (; i < n; ++i) {
        dst[i] = std::max(0.0f, a[i]);
    }
#elif defined(MICROGRAD_SSE)
    size_t i = 0;
    __m128 zero = _mm_setzero_ps();
    for (; i + 4 <= n; i += 4) {
        __m128 va = _mm_loadu_ps(a + i);
        __m128 vc = _mm_max_ps(zero, va);
        _mm_storeu_ps(dst + i, vc);
    }
    for (; i < n; ++i) {
        dst[i] = std::max(0.0f, a[i]);
    }
#else
    for (size_t i = 0; i < n; ++i) {
        dst[i] = std::max(0.0f, a[i]);
    }
#endif
}

/// @brief Vectorized sigmoid approximation: dst = 1 / (1 + exp(-a))
/// Uses fast polynomial approximation for exp
inline void sigmoid_f32(float* dst, const float* a, size_t n)
{
    // Sigmoid is complex to vectorize efficiently
    // Use scalar with potential auto-vectorization
    for (size_t i = 0; i < n; ++i) {
        dst[i] = 1.0f / (1.0f + std::exp(-a[i]));
    }
}

/// @brief Vectorized tanh: dst = tanh(a)
inline void tanh_f32(float* dst, const float* a, size_t n)
{
    for (size_t i = 0; i < n; ++i) {
        dst[i] = std::tanh(a[i]);
    }
}

// ============================================================================
// Vectorized Reductions
// ============================================================================

/// @brief Vectorized sum: return sum of all elements
[[nodiscard]] inline float sum_f32(const float* a, size_t n)
{
#ifdef MICROGRAD_AVX2
    size_t i = 0;
    __m256 vsum = _mm256_setzero_ps();
    
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        vsum = _mm256_add_ps(vsum, va);
    }
    
    // Horizontal sum of 8 floats
    __m128 vlow = _mm256_castps256_ps128(vsum);
    __m128 vhigh = _mm256_extractf128_ps(vsum, 1);
    __m128 vsum128 = _mm_add_ps(vlow, vhigh);
    vsum128 = _mm_hadd_ps(vsum128, vsum128);
    vsum128 = _mm_hadd_ps(vsum128, vsum128);
    float sum = _mm_cvtss_f32(vsum128);
    
    // Handle remaining elements
    for (; i < n; ++i) {
        sum += a[i];
    }
    return sum;
#elif defined(MICROGRAD_SSE)
    size_t i = 0;
    __m128 vsum = _mm_setzero_ps();
    
    for (; i + 4 <= n; i += 4) {
        __m128 va = _mm_loadu_ps(a + i);
        vsum = _mm_add_ps(vsum, va);
    }
    
    // Horizontal sum
    vsum = _mm_hadd_ps(vsum, vsum);
    vsum = _mm_hadd_ps(vsum, vsum);
    float sum = _mm_cvtss_f32(vsum);
    
    for (; i < n; ++i) {
        sum += a[i];
    }
    return sum;
#else
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        sum += a[i];
    }
    return sum;
#endif
}

/// @brief Vectorized dot product: return sum(a * b)
[[nodiscard]] inline float dot_f32(const float* a, const float* b, size_t n)
{
#ifdef MICROGRAD_AVX2
    size_t i = 0;
    __m256 vsum = _mm256_setzero_ps();
    
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        #ifdef MICROGRAD_FMA
            vsum = _mm256_fmadd_ps(va, vb, vsum);
        #else
            __m256 vprod = _mm256_mul_ps(va, vb);
            vsum = _mm256_add_ps(vsum, vprod);
        #endif
    }
    
    // Horizontal sum
    __m128 vlow = _mm256_castps256_ps128(vsum);
    __m128 vhigh = _mm256_extractf128_ps(vsum, 1);
    __m128 vsum128 = _mm_add_ps(vlow, vhigh);
    vsum128 = _mm_hadd_ps(vsum128, vsum128);
    vsum128 = _mm_hadd_ps(vsum128, vsum128);
    float sum = _mm_cvtss_f32(vsum128);
    
    for (; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
#elif defined(MICROGRAD_SSE)
    size_t i = 0;
    __m128 vsum = _mm_setzero_ps();
    
    for (; i + 4 <= n; i += 4) {
        __m128 va = _mm_loadu_ps(a + i);
        __m128 vb = _mm_loadu_ps(b + i);
        __m128 vprod = _mm_mul_ps(va, vb);
        vsum = _mm_add_ps(vsum, vprod);
    }
    
    vsum = _mm_hadd_ps(vsum, vsum);
    vsum = _mm_hadd_ps(vsum, vsum);
    float sum = _mm_cvtss_f32(vsum);
    
    for (; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
#else
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
#endif
}

// ============================================================================
// Fused Multiply-Add Operations
// ============================================================================

/// @brief Vectorized FMA: dst = a * b + c
inline void fma_f32(float* dst, const float* a, const float* b, const float* c, size_t n)
{
#if defined(MICROGRAD_AVX2) && defined(__FMA__)
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 vc = _mm256_loadu_ps(c + i);
        __m256 vr = _mm256_fmadd_ps(va, vb, vc);
        _mm256_storeu_ps(dst + i, vr);
    }
    for (; i < n; ++i) {
        dst[i] = a[i] * b[i] + c[i];
    }
#elif defined(MICROGRAD_AVX2)
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 vc = _mm256_loadu_ps(c + i);
        __m256 vprod = _mm256_mul_ps(va, vb);
        __m256 vr = _mm256_add_ps(vprod, vc);
        _mm256_storeu_ps(dst + i, vr);
    }
    for (; i < n; ++i) {
        dst[i] = a[i] * b[i] + c[i];
    }
#else
    for (size_t i = 0; i < n; ++i) {
        dst[i] = a[i] * b[i] + c[i];
    }
#endif
}

/// @brief Vectorized AXPY: y = a*x + y (BLAS-like)
inline void axpy_f32(float* y, float a, const float* x, size_t n)
{
#ifdef MICROGRAD_AVX2
    size_t i = 0;
    __m256 va = _mm256_set1_ps(a);
    for (; i + 8 <= n; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        __m256 vy = _mm256_loadu_ps(y + i);
        #ifdef __FMA__
            __m256 vr = _mm256_fmadd_ps(va, vx, vy);
        #else
            __m256 vprod = _mm256_mul_ps(va, vx);
            __m256 vr = _mm256_add_ps(vprod, vy);
        #endif
        _mm256_storeu_ps(y + i, vr);
    }
    for (; i < n; ++i) {
        y[i] += a * x[i];
    }
#else
    for (size_t i = 0; i < n; ++i) {
        y[i] += a * x[i];
    }
#endif
}

} // namespace simd
} // namespace micrograd
