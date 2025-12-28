// benchmarks/bench_tensor.cpp
// Performance benchmarks for tensor operations

#include <benchmark/benchmark.h>
#include <micrograd/micrograd.hpp>
#include <random>

using namespace micrograd;

// =============================================================================
// Tensor Creation Benchmarks
// =============================================================================

static void BM_TensorCreation(benchmark::State& state) {
    const size_t size = static_cast<size_t>(state.range(0));
    
    for (auto _ : state) {
        Tensor<float> t(Shape({size, size}));
        benchmark::DoNotOptimize(t.data());
    }
    
    state.SetItemsProcessed(state.iterations() * size * size);
}
BENCHMARK(BM_TensorCreation)->Range(8, 1024);

static void BM_TensorZeros(benchmark::State& state) {
    const size_t size = static_cast<size_t>(state.range(0));
    
    for (auto _ : state) {
        auto t = Tensor<float>::zeros(Shape({size, size}));
        benchmark::DoNotOptimize(t.data());
    }
    
    state.SetItemsProcessed(state.iterations() * size * size);
}
BENCHMARK(BM_TensorZeros)->Range(8, 1024);

// =============================================================================
// Element-wise Operation Benchmarks
// =============================================================================

static void BM_Addition(benchmark::State& state) {
    const size_t size = static_cast<size_t>(state.range(0));
    
    auto a = Tensor<float>::ones(Shape({size, size}));
    auto b = Tensor<float>::ones(Shape({size, size}));
    
    for (auto _ : state) {
        auto c = a + b;
        benchmark::DoNotOptimize(c.data());
    }
    
    state.SetItemsProcessed(state.iterations() * size * size);
    state.SetBytesProcessed(state.iterations() * size * size * sizeof(float) * 3);
}
BENCHMARK(BM_Addition)->Range(8, 1024);

static void BM_Multiplication(benchmark::State& state) {
    const size_t size = static_cast<size_t>(state.range(0));
    
    auto a = Tensor<float>::ones(Shape({size, size}));
    auto b = Tensor<float>::ones(Shape({size, size}));
    
    for (auto _ : state) {
        auto c = a * b;
        benchmark::DoNotOptimize(c.data());
    }
    
    state.SetItemsProcessed(state.iterations() * size * size);
}
BENCHMARK(BM_Multiplication)->Range(8, 1024);

static void BM_ChainedOps(benchmark::State& state) {
    const size_t size = static_cast<size_t>(state.range(0));
    
    auto a = Tensor<float>::ones(Shape({size, size}));
    auto b = Tensor<float>::ones(Shape({size, size}));
    auto c = Tensor<float>::ones(Shape({size, size}));
    
    for (auto _ : state) {
        auto result = (a + b) * c;
        benchmark::DoNotOptimize(result.data());
    }
    
    state.SetItemsProcessed(state.iterations() * size * size);
}
BENCHMARK(BM_ChainedOps)->Range(8, 512);

// =============================================================================
// Activation Function Benchmarks
// =============================================================================

static void BM_ReLU(benchmark::State& state) {
    const size_t size = static_cast<size_t>(state.range(0));
    
    Tensor<float> a(Shape({size, size}));
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < a.size(); ++i) {
        a[i] = dist(rng);
    }
    
    for (auto _ : state) {
        auto b = relu(a);
        benchmark::DoNotOptimize(b.data());
    }
    
    state.SetItemsProcessed(state.iterations() * size * size);
}
BENCHMARK(BM_ReLU)->Range(8, 1024);

static void BM_Sigmoid(benchmark::State& state) {
    const size_t size = static_cast<size_t>(state.range(0));
    
    Tensor<float> a(Shape({size, size}));
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-5.0f, 5.0f);
    for (size_t i = 0; i < a.size(); ++i) {
        a[i] = dist(rng);
    }
    
    for (auto _ : state) {
        auto b = sigmoid(a);
        benchmark::DoNotOptimize(b.data());
    }
    
    state.SetItemsProcessed(state.iterations() * size * size);
}
BENCHMARK(BM_Sigmoid)->Range(8, 512);

// =============================================================================
// Matrix Operation Benchmarks
// =============================================================================

static void BM_MatMul(benchmark::State& state) {
    const size_t size = static_cast<size_t>(state.range(0));
    
    auto a = Tensor<float>::ones(Shape({size, size}));
    auto b = Tensor<float>::ones(Shape({size, size}));
    
    for (auto _ : state) {
        auto c = matmul(a, b);
        benchmark::DoNotOptimize(c.data());
    }
    
    // MatMul is O(n³) operations
    state.SetItemsProcessed(state.iterations() * size * size * size * 2);  // 2 for mul + add
}
BENCHMARK(BM_MatMul)->Range(8, 256);

static void BM_Transpose(benchmark::State& state) {
    const size_t size = static_cast<size_t>(state.range(0));
    
    auto a = Tensor<float>::ones(Shape({size, size}));
    
    for (auto _ : state) {
        auto b = transpose(a);
        benchmark::DoNotOptimize(b.data());
    }
    
    state.SetItemsProcessed(state.iterations() * size * size);
}
BENCHMARK(BM_Transpose)->Range(8, 1024);

// =============================================================================
// Reduction Benchmarks
// =============================================================================

static void BM_Sum(benchmark::State& state) {
    const size_t size = static_cast<size_t>(state.range(0));
    
    auto a = Tensor<float>::ones(Shape({size, size}));
    
    for (auto _ : state) {
        float sum = a.sum();
        benchmark::DoNotOptimize(sum);
    }
    
    state.SetItemsProcessed(state.iterations() * size * size);
}
BENCHMARK(BM_Sum)->Range(8, 1024);

// =============================================================================
// Copy Benchmarks
// =============================================================================

static void BM_DeepCopy(benchmark::State& state) {
    const size_t size = static_cast<size_t>(state.range(0));
    
    auto a = Tensor<float>::ones(Shape({size, size}));
    
    for (auto _ : state) {
        Tensor<float> b(a);  // Copy constructor
        benchmark::DoNotOptimize(b.data());
    }
    
    state.SetBytesProcessed(state.iterations() * size * size * sizeof(float));
}
BENCHMARK(BM_DeepCopy)->Range(8, 1024);

static void BM_Move(benchmark::State& state) {
    const size_t size = static_cast<size_t>(state.range(0));
    
    for (auto _ : state) {
        auto a = Tensor<float>::ones(Shape({size, size}));
        Tensor<float> b(std::move(a));
        benchmark::DoNotOptimize(b.data());
    }
}
BENCHMARK(BM_Move)->Range(8, 1024);

// Entry point
BENCHMARK_MAIN();
