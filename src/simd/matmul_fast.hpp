// filepath: src/simd/matmul_fast.hpp
// Fast Matrix Multiplication - Cache Blocking + SIMD + OpenMP
// Phase 4: Performance Optimization
//
// Techniques applied:
// 1. Cache blocking (tiling) for L1/L2 cache efficiency
// 2. SIMD vectorization (AVX2) for 8 floats per instruction
// 3. OpenMP parallelization for multi-core
// 4. Loop reordering (ikj) for better memory access patterns
#pragma once

#include "simd_detect.hpp"
#include "simd_ops.hpp"
#include <cstddef>
#include <algorithm>
#include <cstring>

// OpenMP support (optional)
#ifdef _OPENMP
    #include <omp.h>
    #define MICROGRAD_OPENMP 1
#endif

namespace micrograd {
namespace simd {

// ============================================================================
// Block Size Tuning
// ============================================================================

/// @brief Optimal block size for L1 cache (~32KB)
/// Block should fit: 3 * BLOCK_SIZE^2 * sizeof(float) < L1_SIZE
/// For L1 = 32KB: sqrt(32KB / (3 * 4)) ≈ 52, round to 64 for alignment
constexpr size_t MATMUL_BLOCK_M = 64;  // Rows of A block
constexpr size_t MATMUL_BLOCK_N = 64;  // Cols of B block  
constexpr size_t MATMUL_BLOCK_K = 64;  // Shared dimension block

// Smaller micro-kernel for register blocking
constexpr size_t MICRO_M = 4;
constexpr size_t MICRO_N = 8;  // AVX2 register width

// ============================================================================
// Naive Matrix Multiplication (Baseline)
// ============================================================================

/// @brief Naive O(n³) matmul: C = A @ B
/// @param C Output matrix (M x N), must be zero-initialized
/// @param A Left matrix (M x K)
/// @param B Right matrix (K x N)
/// @param M Rows of A and C
/// @param K Cols of A, Rows of B
/// @param N Cols of B and C
inline void matmul_naive(
    float* C, const float* A, const float* B,
    size_t M, size_t K, size_t N)
{
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// ============================================================================
// Loop Reordered (ikj order for better cache access)
// ============================================================================

/// @brief Loop-reordered matmul (ikj order)
/// Better memory access pattern: B accessed contiguously
inline void matmul_ikj(
    float* C, const float* A, const float* B,
    size_t M, size_t K, size_t N)
{
    // Zero C first
    std::memset(C, 0, M * N * sizeof(float));
    
    for (size_t i = 0; i < M; ++i) {
        for (size_t k = 0; k < K; ++k) {
            float a_ik = A[i * K + k];
            for (size_t j = 0; j < N; ++j) {
                C[i * N + j] += a_ik * B[k * N + j];
            }
        }
    }
}

// ============================================================================
// SIMD Vectorized (AVX2)
// ============================================================================

/// @brief SIMD-vectorized matmul using AVX2
/// Processes 8 columns at a time
inline void matmul_simd(
    float* C, const float* A, const float* B,
    size_t M, size_t K, size_t N)
{
    std::memset(C, 0, M * N * sizeof(float));
    
#ifdef MICROGRAD_AVX2
    for (size_t i = 0; i < M; ++i) {
        for (size_t k = 0; k < K; ++k) {
            __m256 va = _mm256_set1_ps(A[i * K + k]);
            
            size_t j = 0;
            // Process 8 columns at a time
            for (; j + 8 <= N; j += 8) {
                __m256 vb = _mm256_loadu_ps(&B[k * N + j]);
                __m256 vc = _mm256_loadu_ps(&C[i * N + j]);
                #ifdef __FMA__
                    vc = _mm256_fmadd_ps(va, vb, vc);
                #else
                    vc = _mm256_add_ps(vc, _mm256_mul_ps(va, vb));
                #endif
                _mm256_storeu_ps(&C[i * N + j], vc);
            }
            
            // Handle remaining columns
            float a_ik = A[i * K + k];
            for (; j < N; ++j) {
                C[i * N + j] += a_ik * B[k * N + j];
            }
        }
    }
#else
    matmul_ikj(C, A, B, M, K, N);
#endif
}

// ============================================================================
// Cache-Blocked (Tiled) Matrix Multiplication
// ============================================================================

/// @brief Cache-blocked matmul with configurable tile size
/// @tparam BM Block size for M dimension
/// @tparam BK Block size for K dimension
/// @tparam BN Block size for N dimension
template<size_t BM = MATMUL_BLOCK_M, 
         size_t BK = MATMUL_BLOCK_K, 
         size_t BN = MATMUL_BLOCK_N>
inline void matmul_blocked(
    float* C, const float* A, const float* B,
    size_t M, size_t K, size_t N)
{
    std::memset(C, 0, M * N * sizeof(float));
    
    // Iterate over blocks
    for (size_t ii = 0; ii < M; ii += BM) {
        for (size_t kk = 0; kk < K; kk += BK) {
            for (size_t jj = 0; jj < N; jj += BN) {
                
                // Process block
                size_t i_end = std::min(ii + BM, M);
                size_t k_end = std::min(kk + BK, K);
                size_t j_end = std::min(jj + BN, N);
                
                for (size_t i = ii; i < i_end; ++i) {
                    for (size_t k = kk; k < k_end; ++k) {
                        float a_ik = A[i * K + k];
                        
#ifdef MICROGRAD_AVX2
                        size_t j = jj;
                        __m256 va = _mm256_set1_ps(a_ik);
                        
                        for (; j + 8 <= j_end; j += 8) {
                            __m256 vb = _mm256_loadu_ps(&B[k * N + j]);
                            __m256 vc = _mm256_loadu_ps(&C[i * N + j]);
                            #ifdef __FMA__
                                vc = _mm256_fmadd_ps(va, vb, vc);
                            #else
                                vc = _mm256_add_ps(vc, _mm256_mul_ps(va, vb));
                            #endif
                            _mm256_storeu_ps(&C[i * N + j], vc);
                        }
                        
                        for (; j < j_end; ++j) {
                            C[i * N + j] += a_ik * B[k * N + j];
                        }
#else
                        for (size_t j = jj; j < j_end; ++j) {
                            C[i * N + j] += a_ik * B[k * N + j];
                        }
#endif
                    }
                }
            }
        }
    }
}

// ============================================================================
// OpenMP Parallelized + SIMD + Blocked
// ============================================================================

/// @brief Fully optimized matmul: OpenMP + Cache Blocking + SIMD
/// This is the fastest version for large matrices
inline void matmul_fast(
    float* C, const float* A, const float* B,
    size_t M, size_t K, size_t N)
{
    std::memset(C, 0, M * N * sizeof(float));
    
    constexpr size_t BM = MATMUL_BLOCK_M;
    constexpr size_t BK = MATMUL_BLOCK_K;
    constexpr size_t BN = MATMUL_BLOCK_N;
    
    // Convert to signed for OpenMP compatibility (MSVC requires signed loop vars)
    const ptrdiff_t sM = static_cast<ptrdiff_t>(M);
    const ptrdiff_t sN = static_cast<ptrdiff_t>(N);
    const ptrdiff_t sBM = static_cast<ptrdiff_t>(BM);
    const ptrdiff_t sBN = static_cast<ptrdiff_t>(BN);
    
#ifdef MICROGRAD_OPENMP
    #pragma omp parallel for schedule(dynamic)
#endif
    for (ptrdiff_t ii = 0; ii < sM; ii += sBM) {
        for (ptrdiff_t jj = 0; jj < sN; jj += sBN) {
            const size_t uii = static_cast<size_t>(ii);
            const size_t ujj = static_cast<size_t>(jj);
            
            // Local accumulator for this block (better cache locality)
            alignas(32) float local_c[BM * BN] = {0};
            
            size_t i_end = std::min(uii + BM, M);
            size_t j_end = std::min(ujj + BN, N);
            size_t block_m = i_end - uii;
            size_t block_n = j_end - ujj;
            
            for (size_t kk = 0; kk < K; kk += BK) {
                size_t k_end = std::min(kk + BK, K);
                
                for (size_t i = 0; i < block_m; ++i) {
                    for (size_t k = kk; k < k_end; ++k) {
                        float a_ik = A[(uii + i) * K + k];
                        
#ifdef MICROGRAD_AVX2
                        __m256 va = _mm256_set1_ps(a_ik);
                        size_t j = 0;
                        
                        for (; j + 8 <= block_n; j += 8) {
                            __m256 vb = _mm256_loadu_ps(&B[k * N + ujj + j]);
                            __m256 vc = _mm256_loadu_ps(&local_c[i * BN + j]);
                            #ifdef __FMA__
                                vc = _mm256_fmadd_ps(va, vb, vc);
                            #else
                                vc = _mm256_add_ps(vc, _mm256_mul_ps(va, vb));
                            #endif
                            _mm256_storeu_ps(&local_c[i * BN + j], vc);
                        }
                        
                        for (; j < block_n; ++j) {
                            local_c[i * BN + j] += a_ik * B[k * N + ujj + j];
                        }
#else
                        for (size_t j = 0; j < block_n; ++j) {
                            local_c[i * BN + j] += a_ik * B[k * N + ujj + j];
                        }
#endif
                    }
                }
            }
            
            // Write back to C
            for (size_t i = 0; i < block_m; ++i) {
                for (size_t j = 0; j < block_n; ++j) {
                    C[(uii + i) * N + ujj + j] = local_c[i * BN + j];
                }
            }
        }
    }
}

// ============================================================================
// Auto-dispatch based on matrix size
// ============================================================================

/// @brief Automatically select best matmul algorithm based on size
inline void matmul_auto(
    float* C, const float* A, const float* B,
    size_t M, size_t K, size_t N)
{
    const size_t total_ops = M * K * N;
    
    // Small matrices: overhead of blocking not worth it
    if (total_ops < 4096) {  // ~16x16x16
        matmul_simd(C, A, B, M, K, N);
    }
    // Medium matrices: use blocking
    else if (total_ops < 1000000) {  // ~100x100x100
        matmul_blocked(C, A, B, M, K, N);
    }
    // Large matrices: full optimization
    else {
        matmul_fast(C, A, B, M, K, N);
    }
}

} // namespace simd
} // namespace micrograd
