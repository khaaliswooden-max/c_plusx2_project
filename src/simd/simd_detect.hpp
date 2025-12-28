// filepath: src/simd/simd_detect.hpp
// SIMD Detection and Platform Abstraction
// Phase 4: Performance Optimization
#pragma once

#include <cstdint>
#include <string>

// ============================================================================
// Platform Detection
// ============================================================================

// Compiler detection
#if defined(_MSC_VER)
    #define MICROGRAD_MSVC 1
    #include <intrin.h>
#elif defined(__GNUC__) || defined(__clang__)
    #define MICROGRAD_GCC_CLANG 1
    #include <cpuid.h>
#endif

// Architecture detection
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    #define MICROGRAD_X86 1
#elif defined(__aarch64__) || defined(_M_ARM64)
    #define MICROGRAD_ARM64 1
#endif

// SIMD instruction set detection (compile-time)
#if defined(__AVX512F__)
    #define MICROGRAD_AVX512 1
    #define MICROGRAD_AVX2 1
    #define MICROGRAD_SSE 1
#elif defined(__AVX2__)
    #define MICROGRAD_AVX2 1
    #define MICROGRAD_SSE 1
#elif defined(__AVX__)
    #define MICROGRAD_AVX 1
    #define MICROGRAD_SSE 1
#elif defined(__SSE4_2__) || defined(__SSE4_1__) || defined(__SSE3__) || defined(__SSE2__) || defined(__SSE__)
    #define MICROGRAD_SSE 1
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
    #define MICROGRAD_NEON 1
#endif

// Include appropriate headers
#ifdef MICROGRAD_X86
    #ifdef MICROGRAD_AVX512
        #include <immintrin.h>
    #elif defined(MICROGRAD_AVX2) || defined(MICROGRAD_AVX)
        #include <immintrin.h>
    #elif defined(MICROGRAD_SSE)
        #include <xmmintrin.h>
        #include <emmintrin.h>
        #include <pmmintrin.h>
        #include <smmintrin.h>
    #endif
#endif

#ifdef MICROGRAD_NEON
    #include <arm_neon.h>
#endif

namespace micrograd {
namespace simd {

// ============================================================================
// SIMD Width Constants
// ============================================================================

/// @brief Number of floats per SIMD register
#ifdef MICROGRAD_AVX512
    constexpr size_t SIMD_WIDTH_FLOAT = 16;  // 512 bits / 32 bits
    constexpr size_t SIMD_WIDTH_DOUBLE = 8;  // 512 bits / 64 bits
#elif defined(MICROGRAD_AVX2) || defined(MICROGRAD_AVX)
    constexpr size_t SIMD_WIDTH_FLOAT = 8;   // 256 bits / 32 bits
    constexpr size_t SIMD_WIDTH_DOUBLE = 4;  // 256 bits / 64 bits
#elif defined(MICROGRAD_SSE) || defined(MICROGRAD_NEON)
    constexpr size_t SIMD_WIDTH_FLOAT = 4;   // 128 bits / 32 bits
    constexpr size_t SIMD_WIDTH_DOUBLE = 2;  // 128 bits / 64 bits
#else
    constexpr size_t SIMD_WIDTH_FLOAT = 1;   // Scalar fallback
    constexpr size_t SIMD_WIDTH_DOUBLE = 1;
#endif

/// @brief Alignment requirement for SIMD operations (bytes)
#ifdef MICROGRAD_AVX512
    constexpr size_t SIMD_ALIGNMENT = 64;
#elif defined(MICROGRAD_AVX2) || defined(MICROGRAD_AVX)
    constexpr size_t SIMD_ALIGNMENT = 32;
#else
    constexpr size_t SIMD_ALIGNMENT = 16;
#endif

// ============================================================================
// Runtime SIMD Detection
// ============================================================================

/// @brief SIMD capability flags
struct SimdCapabilities
{
    bool sse = false;
    bool sse2 = false;
    bool sse3 = false;
    bool ssse3 = false;
    bool sse41 = false;
    bool sse42 = false;
    bool avx = false;
    bool avx2 = false;
    bool fma = false;
    bool avx512f = false;
    bool avx512dq = false;
    bool avx512vl = false;
    bool neon = false;
};

/// @brief Detect CPU SIMD capabilities at runtime
[[nodiscard]] inline SimdCapabilities detect_simd()
{
    SimdCapabilities caps;

#ifdef MICROGRAD_X86
    #ifdef MICROGRAD_MSVC
        int cpuInfo[4];
        __cpuid(cpuInfo, 0);
        int nIds = cpuInfo[0];

        if (nIds >= 1) {
            __cpuid(cpuInfo, 1);
            caps.sse    = (cpuInfo[3] & (1 << 25)) != 0;
            caps.sse2   = (cpuInfo[3] & (1 << 26)) != 0;
            caps.sse3   = (cpuInfo[2] & (1 << 0)) != 0;
            caps.ssse3  = (cpuInfo[2] & (1 << 9)) != 0;
            caps.sse41  = (cpuInfo[2] & (1 << 19)) != 0;
            caps.sse42  = (cpuInfo[2] & (1 << 20)) != 0;
            caps.avx    = (cpuInfo[2] & (1 << 28)) != 0;
            caps.fma    = (cpuInfo[2] & (1 << 12)) != 0;
        }

        if (nIds >= 7) {
            __cpuidex(cpuInfo, 7, 0);
            caps.avx2     = (cpuInfo[1] & (1 << 5)) != 0;
            caps.avx512f  = (cpuInfo[1] & (1 << 16)) != 0;
            caps.avx512dq = (cpuInfo[1] & (1 << 17)) != 0;
            caps.avx512vl = (cpuInfo[1] & (1 << 31)) != 0;
        }
    #else
        unsigned int eax, ebx, ecx, edx;
        if (__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
            caps.sse    = (edx & (1 << 25)) != 0;
            caps.sse2   = (edx & (1 << 26)) != 0;
            caps.sse3   = (ecx & (1 << 0)) != 0;
            caps.ssse3  = (ecx & (1 << 9)) != 0;
            caps.sse41  = (ecx & (1 << 19)) != 0;
            caps.sse42  = (ecx & (1 << 20)) != 0;
            caps.avx    = (ecx & (1 << 28)) != 0;
            caps.fma    = (ecx & (1 << 12)) != 0;
        }

        if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
            caps.avx2     = (ebx & (1 << 5)) != 0;
            caps.avx512f  = (ebx & (1 << 16)) != 0;
            caps.avx512dq = (ebx & (1 << 17)) != 0;
            caps.avx512vl = (ebx & (1 << 31)) != 0;
        }
    #endif
#endif

#ifdef MICROGRAD_NEON
    caps.neon = true;  // NEON always available on ARM64
#endif

    return caps;
}

/// @brief Get human-readable SIMD capability string
[[nodiscard]] inline std::string simd_info()
{
    auto caps = detect_simd();
    std::string info = "SIMD: ";
    
    if (caps.avx512f) info += "AVX-512 ";
    else if (caps.avx2) info += "AVX2 ";
    else if (caps.avx) info += "AVX ";
    else if (caps.sse42) info += "SSE4.2 ";
    else if (caps.sse2) info += "SSE2 ";
    else if (caps.neon) info += "NEON ";
    else info += "Scalar ";
    
    if (caps.fma) info += "FMA ";
    
    info += "| Width: " + std::to_string(SIMD_WIDTH_FLOAT) + " floats";
    info += " | Align: " + std::to_string(SIMD_ALIGNMENT) + " bytes";
    
    return info;
}

// ============================================================================
// Alignment Helpers
// ============================================================================

/// @brief Check if pointer is aligned to SIMD boundary
template<typename T>
[[nodiscard]] inline bool is_aligned(const T* ptr, size_t alignment = SIMD_ALIGNMENT)
{
    return (reinterpret_cast<uintptr_t>(ptr) % alignment) == 0;
}

/// @brief Round up to next multiple of alignment
[[nodiscard]] inline size_t align_up(size_t n, size_t alignment = SIMD_ALIGNMENT)
{
    return (n + alignment - 1) & ~(alignment - 1);
}

} // namespace simd
} // namespace micrograd
