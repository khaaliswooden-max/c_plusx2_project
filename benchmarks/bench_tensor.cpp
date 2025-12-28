// benchmarks/bench_tensor.cpp
// Simple performance benchmarks for tensor operations (no external dependencies)

#include <micrograd/micrograd.hpp>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <random>

using namespace micrograd;

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
    
    double elapsed_us() const {
        return std::chrono::duration<double, std::micro>(end_ - start_).count();
    }
    
private:
    std::chrono::high_resolution_clock::time_point start_, end_;
};

template<typename Func>
double benchmark(Func&& func, int iterations = 100) {
    Timer timer;
    
    // Warmup
    for (int i = 0; i < 5; ++i) func();
    
    timer.start();
    for (int i = 0; i < iterations; ++i) {
        func();
    }
    timer.stop();
    
    return timer.elapsed_ms() / iterations;
}

// =============================================================================
// Benchmarks
// =============================================================================

void bench_creation(size_t size) {
    double time_ms = benchmark([&]() {
        Tensor<float> t(Shape({size, size}));
        (void)t.data();  // Prevent optimization
    });
    
    std::cout << "Creation " << size << "x" << size << ": " 
              << std::fixed << std::setprecision(3) << time_ms << " ms" << std::endl;
}

void bench_zeros(size_t size) {
    double time_ms = benchmark([&]() {
        auto t = Tensor<float>::zeros(Shape({size, size}));
        (void)t.data();
    });
    
    std::cout << "Zeros " << size << "x" << size << ": " 
              << std::fixed << std::setprecision(3) << time_ms << " ms" << std::endl;
}

void bench_addition(size_t size) {
    auto a = Tensor<float>::ones(Shape({size, size}));
    auto b = Tensor<float>::ones(Shape({size, size}));
    
    double time_ms = benchmark([&]() {
        auto c = a + b;
        (void)c.data();
    });
    
    double gflops = (size * size) / (time_ms * 1e6);
    std::cout << "Addition " << size << "x" << size << ": " 
              << std::fixed << std::setprecision(3) << time_ms << " ms "
              << "(" << std::setprecision(2) << gflops << " GFLOP/s)" << std::endl;
}

void bench_multiplication(size_t size) {
    auto a = Tensor<float>::ones(Shape({size, size}));
    auto b = Tensor<float>::ones(Shape({size, size}));
    
    double time_ms = benchmark([&]() {
        auto c = a * b;
        (void)c.data();
    });
    
    double gflops = (size * size) / (time_ms * 1e6);
    std::cout << "Element-wise Mul " << size << "x" << size << ": " 
              << std::fixed << std::setprecision(3) << time_ms << " ms "
              << "(" << std::setprecision(2) << gflops << " GFLOP/s)" << std::endl;
}

void bench_chained_ops(size_t size) {
    auto a = Tensor<float>::ones(Shape({size, size}));
    auto b = Tensor<float>::ones(Shape({size, size}));
    auto c = Tensor<float>::ones(Shape({size, size}));
    
    double time_ms = benchmark([&]() {
        auto result = (a + b) * c;
        (void)result.data();
    });
    
    std::cout << "Chained (a+b)*c " << size << "x" << size << ": " 
              << std::fixed << std::setprecision(3) << time_ms << " ms" << std::endl;
}

void bench_relu(size_t size) {
    Tensor<float> a(Shape({size, size}));
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < a.size(); ++i) {
        a[i] = dist(rng);
    }
    
    double time_ms = benchmark([&]() {
        auto b = relu(a);
        (void)b.data();
    });
    
    std::cout << "ReLU " << size << "x" << size << ": " 
              << std::fixed << std::setprecision(3) << time_ms << " ms" << std::endl;
}

void bench_sigmoid(size_t size) {
    Tensor<float> a(Shape({size, size}));
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-5.0f, 5.0f);
    for (size_t i = 0; i < a.size(); ++i) {
        a[i] = dist(rng);
    }
    
    double time_ms = benchmark([&]() {
        auto b = sigmoid(a);
        (void)b.data();
    });
    
    std::cout << "Sigmoid " << size << "x" << size << ": " 
              << std::fixed << std::setprecision(3) << time_ms << " ms" << std::endl;
}

void bench_matmul(size_t size) {
    auto a = Tensor<float>::ones(Shape({size, size}));
    auto b = Tensor<float>::ones(Shape({size, size}));
    
    double time_ms = benchmark([&]() {
        auto c = matmul(a, b);
        (void)c.data();
    }, 10);  // Fewer iterations for expensive op
    
    // MatMul is 2*n³ operations (multiply + add)
    double gflops = (2.0 * size * size * size) / (time_ms * 1e6);
    std::cout << "MatMul " << size << "x" << size << ": " 
              << std::fixed << std::setprecision(3) << time_ms << " ms "
              << "(" << std::setprecision(2) << gflops << " GFLOP/s)" << std::endl;
}

void bench_transpose(size_t size) {
    auto a = Tensor<float>::ones(Shape({size, size}));
    
    double time_ms = benchmark([&]() {
        auto b = transpose(a);
        (void)b.data();
    });
    
    std::cout << "Transpose " << size << "x" << size << ": " 
              << std::fixed << std::setprecision(3) << time_ms << " ms" << std::endl;
}

void bench_sum(size_t size) {
    auto a = Tensor<float>::ones(Shape({size, size}));
    
    double time_ms = benchmark([&]() {
        float s = a.sum();
        (void)s;
    });
    
    std::cout << "Sum " << size << "x" << size << ": " 
              << std::fixed << std::setprecision(3) << time_ms << " ms" << std::endl;
}

void bench_deep_copy(size_t size) {
    auto a = Tensor<float>::ones(Shape({size, size}));
    
    double time_ms = benchmark([&]() {
        Tensor<float> b(a);
        (void)b.data();
    });
    
    double bandwidth = (size * size * sizeof(float)) / (time_ms * 1e6);  // GB/s
    std::cout << "Deep Copy " << size << "x" << size << ": " 
              << std::fixed << std::setprecision(3) << time_ms << " ms "
              << "(" << std::setprecision(2) << bandwidth << " GB/s)" << std::endl;
}

// =============================================================================
// Main
// =============================================================================

int main() {
    std::cout << "=== MicroGrad++ Benchmarks ===" << std::endl;
    std::cout << "(Average over 100 iterations, 5 warmup)" << std::endl;
    std::cout << std::endl;
    
    const std::vector<size_t> sizes = {64, 128, 256, 512};
    
    std::cout << "--- Tensor Creation ---" << std::endl;
    for (size_t s : sizes) bench_creation(s);
    std::cout << std::endl;
    
    std::cout << "--- Zeros Initialization ---" << std::endl;
    for (size_t s : sizes) bench_zeros(s);
    std::cout << std::endl;
    
    std::cout << "--- Element-wise Addition ---" << std::endl;
    for (size_t s : sizes) bench_addition(s);
    std::cout << std::endl;
    
    std::cout << "--- Element-wise Multiplication ---" << std::endl;
    for (size_t s : sizes) bench_multiplication(s);
    std::cout << std::endl;
    
    std::cout << "--- Chained Operations ---" << std::endl;
    for (size_t s : sizes) bench_chained_ops(s);
    std::cout << std::endl;
    
    std::cout << "--- ReLU Activation ---" << std::endl;
    for (size_t s : sizes) bench_relu(s);
    std::cout << std::endl;
    
    std::cout << "--- Sigmoid Activation ---" << std::endl;
    for (size_t s : sizes) bench_sigmoid(s);
    std::cout << std::endl;
    
    std::cout << "--- Matrix Multiplication (O(n³)) ---" << std::endl;
    for (size_t s : {64ul, 128ul, 256ul}) bench_matmul(s);  // Skip 512 - too slow
    std::cout << std::endl;
    
    std::cout << "--- Transpose ---" << std::endl;
    for (size_t s : sizes) bench_transpose(s);
    std::cout << std::endl;
    
    std::cout << "--- Sum Reduction ---" << std::endl;
    for (size_t s : sizes) bench_sum(s);
    std::cout << std::endl;
    
    std::cout << "--- Deep Copy ---" << std::endl;
    for (size_t s : sizes) bench_deep_copy(s);
    std::cout << std::endl;
    
    std::cout << "=== Benchmarks Complete ===" << std::endl;
    return 0;
}

