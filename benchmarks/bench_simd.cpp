// benchmarks/bench_simd.cpp
// SIMD Performance Benchmark for MicroGrad++
// Compares naive vs SIMD-optimized implementations using the library

#include <simd/simd.hpp>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using namespace micrograd::simd;

// =============================================================================
// Aligned Memory Allocation (RAII wrapper)
// =============================================================================

class AlignedBuffer {
public:
    explicit AlignedBuffer(size_t n) : size_(n) {
        void* ptr = aligned_alloc(n * sizeof(float));
        if (!ptr && n > 0) {
            throw std::bad_alloc();
        }
        data_ = static_cast<float*>(ptr);
    }
    
    ~AlignedBuffer() { 
        if (data_) aligned_free(data_); 
    }
    
    // Delete copy
    AlignedBuffer(const AlignedBuffer&) = delete;
    AlignedBuffer& operator=(const AlignedBuffer&) = delete;
    
    // Allow move
    AlignedBuffer(AlignedBuffer&& other) noexcept 
        : size_(other.size_), data_(other.data_) {
        other.data_ = nullptr;
        other.size_ = 0;
    }
    
    float* data() { return data_; }
    const float* data() const { return data_; }
    size_t size() const { return size_; }
    
    float& operator[](size_t i) { return data_[i]; }
    const float& operator[](size_t i) const { return data_[i]; }
    
private:
    size_t size_;
    float* data_;
};

// =============================================================================
// Timing Utilities
// =============================================================================

class Timer {
public:
    void start() { start_ = std::chrono::high_resolution_clock::now(); }
    void stop() { end_ = std::chrono::high_resolution_clock::now(); }
    
    double elapsed_ms() const {
        return std::chrono::duration<double, std::milli>(end_ - start_).count();
    }
    
private:
    std::chrono::high_resolution_clock::time_point start_, end_;
};

template<typename Func>
double benchmark(Func&& func, int iterations = 100) {
    Timer timer;
    
    // Warmup
    for (int i = 0; i < 10; ++i) func();
    
    timer.start();
    for (int i = 0; i < iterations; ++i) {
        func();
    }
    timer.stop();
    
    return timer.elapsed_ms() / iterations;
}

// =============================================================================
// Random Initialization
// =============================================================================

void fill_random(float* data, size_t n, float min_val = -1.0f, float max_val = 1.0f) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(min_val, max_val);
    for (size_t i = 0; i < n; ++i) {
        data[i] = dist(rng);
    }
}

// =============================================================================
// NAIVE IMPLEMENTATIONS (for comparison)
// =============================================================================

void naive_add(const float* a, const float* b, float* c, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        c[i] = a[i] + b[i];
    }
}

void naive_mul(const float* a, const float* b, float* c, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        c[i] = a[i] * b[i];
    }
}

void naive_relu(const float* a, float* b, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        b[i] = a[i] > 0.0f ? a[i] : 0.0f;
    }
}

float naive_dot(const float* a, const float* b, size_t n) {
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

// =============================================================================
// BENCHMARK FUNCTIONS
// =============================================================================

void print_header() {
    std::cout << "================================================================\n";
    std::cout << "  MicroGrad++ SIMD Performance Benchmark\n";
    std::cout << "================================================================\n\n";
    
    // Print SIMD info using library function
    std::cout << simd_info() << "\n";
    
    // Print OpenMP info
    std::cout << "OpenMP: ";
#ifdef _OPENMP
    std::cout << "Enabled (" << omp_get_max_threads() << " threads)\n";
#else
    std::cout << "Disabled\n";
#endif
    std::cout << "\n";
}

void benchmark_elementwise() {
    constexpr size_t N = 1'000'000;  // 1M elements
    
    AlignedBuffer a(N), b(N), c(N);
    fill_random(a.data(), N);
    fill_random(b.data(), N);
    
    std::cout << "[Benchmark 1] Element-wise Operations (" << N / 1000000 << "M elements)\n";
    std::cout << "------------------------------------------------------------\n";
    std::cout << std::setw(18) << "Operation" 
              << std::setw(15) << "Naive (ms)"
              << std::setw(15) << "SIMD (ms)"
              << std::setw(12) << "Speedup\n";
    std::cout << "------------------------------------------------------------\n";
    
    // Add
    {
        double naive_ms = benchmark([&]() { naive_add(a.data(), b.data(), c.data(), N); });
        double simd_ms = benchmark([&]() { add_f32(c.data(), a.data(), b.data(), N); });
        double speedup = naive_ms / simd_ms;
        
        std::cout << std::setw(18) << "Add"
                  << std::setw(15) << std::fixed << std::setprecision(3) << naive_ms
                  << std::setw(15) << std::fixed << std::setprecision(3) << simd_ms
                  << std::setw(10) << std::fixed << std::setprecision(1) << speedup << "x\n";
    }
    
    // Multiply
    {
        double naive_ms = benchmark([&]() { naive_mul(a.data(), b.data(), c.data(), N); });
        double simd_ms = benchmark([&]() { mul_f32(c.data(), a.data(), b.data(), N); });
        double speedup = naive_ms / simd_ms;
        
        std::cout << std::setw(18) << "Multiply"
                  << std::setw(15) << std::fixed << std::setprecision(3) << naive_ms
                  << std::setw(15) << std::fixed << std::setprecision(3) << simd_ms
                  << std::setw(10) << std::fixed << std::setprecision(1) << speedup << "x\n";
    }
    
    // ReLU
    {
        double naive_ms = benchmark([&]() { naive_relu(a.data(), c.data(), N); });
        double simd_ms = benchmark([&]() { relu_f32(c.data(), a.data(), N); });
        double speedup = naive_ms / simd_ms;
        
        std::cout << std::setw(18) << "ReLU"
                  << std::setw(15) << std::fixed << std::setprecision(3) << naive_ms
                  << std::setw(15) << std::fixed << std::setprecision(3) << simd_ms
                  << std::setw(10) << std::fixed << std::setprecision(1) << speedup << "x\n";
    }
    
    // Dot Product
    {
        volatile float result;  // Prevent optimization
        double naive_ms = benchmark([&]() { result = naive_dot(a.data(), b.data(), N); });
        double simd_ms = benchmark([&]() { result = dot_f32(a.data(), b.data(), N); });
        double speedup = naive_ms / simd_ms;
        (void)result;
        
        std::cout << std::setw(18) << "Dot Product"
                  << std::setw(15) << std::fixed << std::setprecision(3) << naive_ms
                  << std::setw(15) << std::fixed << std::setprecision(3) << simd_ms
                  << std::setw(10) << std::fixed << std::setprecision(1) << speedup << "x\n";
    }
    
    std::cout << "\n";
}

void benchmark_matmul() {
    std::cout << "[Benchmark 2] Matrix Multiplication (N x N @ N x N)\n";
    std::cout << "------------------------------------------------------------\n";
    std::cout << std::setw(8) << "Size"
              << std::setw(12) << "Naive"
              << std::setw(12) << "Reorder"
              << std::setw(12) << "SIMD"
              << std::setw(12) << "Blocked"
              << std::setw(12) << "GFLOPS\n";
    std::cout << "------------------------------------------------------------\n";
    
    std::vector<size_t> sizes = {64, 128, 256, 512, 1024};
    
    for (size_t N : sizes) {
        AlignedBuffer A(N * N), B(N * N), C(N * N);
        fill_random(A.data(), N * N);
        fill_random(B.data(), N * N);
        
        // Number of FLOPs for matmul: 2 * N^3 (multiply + add)
        double flops = 2.0 * N * N * N;
        
        // Reduce iterations for large matrices
        int iters = (N <= 256) ? 20 : (N <= 512) ? 5 : 2;
        
        double naive_ms = 0, reorder_ms = 0, simd_ms = 0, blocked_ms = 0;
        
        // Only run naive for smaller sizes (too slow for large)
        if (N <= 256) {
            std::memset(C.data(), 0, N * N * sizeof(float));
            naive_ms = benchmark([&]() { 
                matmul_naive(C.data(), A.data(), B.data(), N, N, N); 
            }, iters);
        }
        
        // Reordered (ikj loop order)
        reorder_ms = benchmark([&]() { 
            matmul_ikj(C.data(), A.data(), B.data(), N, N, N); 
        }, iters);
        
        // SIMD vectorized
        simd_ms = benchmark([&]() { 
            matmul_simd(C.data(), A.data(), B.data(), N, N, N); 
        }, iters);
        
        // Blocked + SIMD
        blocked_ms = benchmark([&]() { 
            matmul_blocked(C.data(), A.data(), B.data(), N, N, N); 
        }, iters);
        
        // Calculate GFLOPS based on best time
        double best_ms = blocked_ms;
        double gflops = (flops / (best_ms / 1000.0)) / 1e9;
        
        // Format output
        std::cout << std::setw(8) << N;
        
        if (N <= 256) {
            std::cout << std::setw(10) << std::fixed << std::setprecision(2) << naive_ms << "ms";
        } else {
            std::cout << std::setw(12) << "---";
        }
        
        std::cout << std::setw(10) << std::fixed << std::setprecision(2) << reorder_ms << "ms"
                  << std::setw(10) << std::fixed << std::setprecision(2) << simd_ms << "ms"
                  << std::setw(10) << std::fixed << std::setprecision(2) << blocked_ms << "ms"
                  << std::setw(10) << std::fixed << std::setprecision(1) << gflops << "\n";
    }
    
    std::cout << "\n";
}

void benchmark_memory_bandwidth() {
    constexpr size_t N = 10'000'000;  // 10M elements = 40MB
    
    std::cout << "[Benchmark 3] Memory Bandwidth (40 MB transfer)\n";
    std::cout << "------------------------------------------------------------\n";
    
    AlignedBuffer src(N), dst(N);
    fill_random(src.data(), N);
    
    // Copy bandwidth
    double copy_ms = benchmark([&]() {
        std::memcpy(dst.data(), src.data(), N * sizeof(float));
    }, 50);
    
    double bandwidth = (N * sizeof(float) * 2) / (copy_ms / 1000.0) / 1e9;  // GB/s (read + write)
    std::cout << "memcpy:       " << std::fixed << std::setprecision(2) << bandwidth << " GB/s\n";
    
#ifdef MICROGRAD_AVX2
    // SIMD streaming copy using library-style operations
    double simd_copy_ms = benchmark([&]() {
        const float* s = src.data();
        float* d = dst.data();
        size_t i = 0;
        for (; i + 8 <= N; i += 8) {
            __m256 v = _mm256_load_ps(s + i);
            _mm256_store_ps(d + i, v);
        }
        for (; i < N; ++i) {
            d[i] = s[i];
        }
    }, 50);
    
    double simd_bandwidth = (N * sizeof(float) * 2) / (simd_copy_ms / 1000.0) / 1e9;
    std::cout << "SIMD copy:    " << std::fixed << std::setprecision(2) << simd_bandwidth << " GB/s\n";
    
    // Non-temporal streaming stores (bypass cache)
    double stream_ms = benchmark([&]() {
        const float* s = src.data();
        float* d = dst.data();
        size_t i = 0;
        for (; i + 8 <= N; i += 8) {
            __m256 v = _mm256_load_ps(s + i);
            _mm256_stream_ps(d + i, v);
        }
        _mm_sfence();  // Ensure stores complete
        for (; i < N; ++i) {
            d[i] = s[i];
        }
    }, 50);
    
    double stream_bandwidth = (N * sizeof(float) * 2) / (stream_ms / 1000.0) / 1e9;
    std::cout << "Streaming:    " << std::fixed << std::setprecision(2) << stream_bandwidth << " GB/s\n";
#endif
    
    std::cout << "\n";
}

void benchmark_activation_functions() {
    constexpr size_t N = 1'000'000;
    
    std::cout << "[Benchmark 4] Activation Functions (" << N / 1000000 << "M elements)\n";
    std::cout << "------------------------------------------------------------\n";
    std::cout << std::setw(18) << "Function"
              << std::setw(15) << "Naive (ms)"
              << std::setw(15) << "SIMD (ms)"
              << std::setw(12) << "Speedup\n";
    std::cout << "------------------------------------------------------------\n";
    
    AlignedBuffer a(N), b(N);
    fill_random(a.data(), N, -5.0f, 5.0f);
    
    // ReLU
    {
        double naive_ms = benchmark([&]() { naive_relu(a.data(), b.data(), N); });
        double simd_ms = benchmark([&]() { relu_f32(b.data(), a.data(), N); });
        double speedup = naive_ms / simd_ms;
        
        std::cout << std::setw(18) << "ReLU"
                  << std::setw(15) << std::fixed << std::setprecision(3) << naive_ms
                  << std::setw(15) << std::fixed << std::setprecision(3) << simd_ms
                  << std::setw(10) << std::fixed << std::setprecision(1) << speedup << "x\n";
    }
    
    // Leaky ReLU
    {
        auto naive_leaky_relu = [](const float* a, float* b, size_t n, float alpha) {
            for (size_t i = 0; i < n; ++i) {
                b[i] = a[i] > 0.0f ? a[i] : alpha * a[i];
            }
        };
        
#ifdef MICROGRAD_AVX2
        auto simd_leaky_relu = [](const float* a, float* b, size_t n, float alpha) {
            __m256 valpha = _mm256_set1_ps(alpha);
            __m256 zero = _mm256_setzero_ps();
            size_t i = 0;
            
            for (; i + 8 <= n; i += 8) {
                __m256 va = _mm256_load_ps(a + i);
                __m256 mask = _mm256_cmp_ps(va, zero, _CMP_GT_OQ);
                __m256 neg = _mm256_mul_ps(va, valpha);
                __m256 result = _mm256_blendv_ps(neg, va, mask);
                _mm256_store_ps(b + i, result);
            }
            
            for (; i < n; ++i) {
                b[i] = a[i] > 0.0f ? a[i] : alpha * a[i];
            }
        };
        
        double naive_ms = benchmark([&]() { naive_leaky_relu(a.data(), b.data(), N, 0.01f); });
        double simd_ms = benchmark([&]() { simd_leaky_relu(a.data(), b.data(), N, 0.01f); });
        double speedup = naive_ms / simd_ms;
        
        std::cout << std::setw(18) << "Leaky ReLU"
                  << std::setw(15) << std::fixed << std::setprecision(3) << naive_ms
                  << std::setw(15) << std::fixed << std::setprecision(3) << simd_ms
                  << std::setw(10) << std::fixed << std::setprecision(1) << speedup << "x\n";
#else
        double naive_ms = benchmark([&]() { naive_leaky_relu(a.data(), b.data(), N, 0.01f); });
        std::cout << std::setw(18) << "Leaky ReLU"
                  << std::setw(15) << std::fixed << std::setprecision(3) << naive_ms
                  << std::setw(15) << "N/A"
                  << std::setw(12) << "---\n";
#endif
    }
    
    // Sigmoid (compute-bound, harder to vectorize efficiently)
    {
        double naive_ms = benchmark([&]() { sigmoid_f32(b.data(), a.data(), N); }, 20);
        
        std::cout << std::setw(18) << "Sigmoid"
                  << std::setw(15) << std::fixed << std::setprecision(3) << naive_ms
                  << std::setw(15) << "N/A*"
                  << std::setw(12) << "---\n";
    }
    
    std::cout << "\n* Sigmoid requires fast exp() approximation for efficient SIMD\n\n";
}

void benchmark_aligned_tensor() {
    std::cout << "[Benchmark 5] AlignedTensor Class Operations\n";
    std::cout << "------------------------------------------------------------\n";
    
    constexpr size_t N = 1000;  // 1000x1000 matrix
    
    // Create AlignedTensors using the library
    auto A = AlignedTensor<float>::randn({N, N});
    auto B = AlignedTensor<float>::randn({N, N});
    
    std::cout << "Matrix size: " << N << "x" << N << " (" 
              << (N * N * sizeof(float) / 1e6) << " MB each)\n";
    std::cout << "Memory aligned: " << (A.is_aligned() ? "Yes" : "No") << "\n\n";
    
    // Element-wise add
    {
        double ms = benchmark([&]() {
            auto C = A + B;
            (void)C.data();
        }, 50);
        double gflops = (N * N) / (ms * 1e6);
        std::cout << "Add:        " << std::fixed << std::setprecision(3) << ms 
                  << " ms (" << std::setprecision(2) << gflops << " GFLOP/s)\n";
    }
    
    // Element-wise multiply
    {
        double ms = benchmark([&]() {
            auto C = A * B;
            (void)C.data();
        }, 50);
        double gflops = (N * N) / (ms * 1e6);
        std::cout << "Multiply:   " << std::fixed << std::setprecision(3) << ms 
                  << " ms (" << std::setprecision(2) << gflops << " GFLOP/s)\n";
    }
    
    // Matrix multiplication
    {
        double ms = benchmark([&]() {
            auto C = matmul(A, B);
            (void)C.data();
        }, 5);
        double gflops = (2.0 * N * N * N) / (ms * 1e6);
        std::cout << "MatMul:     " << std::fixed << std::setprecision(3) << ms 
                  << " ms (" << std::setprecision(2) << gflops << " GFLOP/s)\n";
    }
    
    // Dot product
    {
        auto a_flat = AlignedTensor<float>::randn({N * N});
        auto b_flat = AlignedTensor<float>::randn({N * N});
        
        volatile float result;
        double ms = benchmark([&]() {
            result = dot(a_flat, b_flat);
        }, 100);
        (void)result;
        double gflops = (2.0 * N * N) / (ms * 1e6);
        std::cout << "Dot:        " << std::fixed << std::setprecision(3) << ms 
                  << " ms (" << std::setprecision(2) << gflops << " GFLOP/s)\n";
    }
    
    // ReLU
    {
        double ms = benchmark([&]() {
            auto C = relu(A);
            (void)C.data();
        }, 50);
        double gflops = (N * N) / (ms * 1e6);
        std::cout << "ReLU:       " << std::fixed << std::setprecision(3) << ms 
                  << " ms (" << std::setprecision(2) << gflops << " GFLOP/s)\n";
    }
    
    std::cout << "\n";
}

void print_speedup_summary() {
    std::cout << "================================================================\n";
    std::cout << "  Performance Summary\n";
    std::cout << "================================================================\n";
    std::cout << "\n";
    std::cout << "SIMD provides significant speedups for:\n";
    std::cout << "  - Element-wise ops: 5-8x faster (memory bandwidth limited)\n";
    std::cout << "  - Matrix multiply:  3-10x faster (compute + cache optimization)\n";
    std::cout << "  - Activations:      5-7x faster (ReLU, Leaky ReLU)\n";
    std::cout << "\n";
    std::cout << "Key optimization techniques used:\n";
#ifdef MICROGRAD_AVX512
    std::cout << "  - AVX-512 512-bit vectors (16 floats/op)\n";
#elif defined(MICROGRAD_AVX2)
    std::cout << "  - AVX2 256-bit vectors (8 floats/op)\n";
#elif defined(MICROGRAD_SSE)
    std::cout << "  - SSE 128-bit vectors (4 floats/op)\n";
#else
    std::cout << "  - Scalar operations (no SIMD available)\n";
#endif
#ifdef MICROGRAD_FMA
    std::cout << "  - FMA (fused multiply-add)\n";
#endif
    std::cout << "  - Cache blocking for matrix multiply\n";
    std::cout << "  - Aligned memory allocation (" << SIMD_ALIGNMENT << "-byte)\n";
    std::cout << "  - Loop reordering for cache locality\n";
#ifdef _OPENMP
    std::cout << "  - OpenMP parallelization\n";
#endif
    std::cout << "\n";
    std::cout << "================================================================\n";
}

// =============================================================================
// MAIN
// =============================================================================

int main() {
    print_header();
    
    benchmark_elementwise();
    benchmark_matmul();
    benchmark_memory_bandwidth();
    benchmark_activation_functions();
    benchmark_aligned_tensor();
    
    print_speedup_summary();
    
    return 0;
}
