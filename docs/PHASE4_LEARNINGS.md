# Phase 4 Learnings: Performance Optimization

**Completed:** Phase 4 of MicroGrad++  
**Level:** Systems Programming  
**Duration:** Weeks 10-12

---

## 🎓 What You Learned

### 1. SIMD (Single Instruction, Multiple Data)

**Concept:** Process multiple data elements with a single CPU instruction using wide registers.

| Instruction Set | Register Width | Floats/Op | Year |
|-----------------|----------------|-----------|------|
| SSE             | 128 bits       | 4         | 1999 |
| AVX             | 256 bits       | 8         | 2011 |
| AVX2            | 256 bits       | 8 + FMA   | 2013 |
| AVX-512         | 512 bits       | 16        | 2016 |

**Scalar vs SIMD:**
```cpp
// Scalar: 1 operation per instruction
for (int i = 0; i < N; ++i) {
    c[i] = a[i] + b[i];  // 1 add per iteration
}

// AVX2: 8 operations per instruction
for (int i = 0; i < N; i += 8) {
    __m256 va = _mm256_loadu_ps(&a[i]);  // Load 8 floats
    __m256 vb = _mm256_loadu_ps(&b[i]);  // Load 8 floats
    __m256 vc = _mm256_add_ps(va, vb);   // Add 8 floats at once!
    _mm256_storeu_ps(&c[i], vc);         // Store 8 floats
}
```

**Key intrinsics used:**

| Intrinsic | Operation | Description |
|-----------|-----------|-------------|
| `_mm256_loadu_ps` | Load | Load 8 floats (unaligned) |
| `_mm256_storeu_ps` | Store | Store 8 floats (unaligned) |
| `_mm256_add_ps` | Add | a + b (8 floats) |
| `_mm256_mul_ps` | Multiply | a * b (8 floats) |
| `_mm256_fmadd_ps` | FMA | a * b + c (fused, faster) |
| `_mm256_max_ps` | Max | max(a, b) (for ReLU) |
| `_mm256_set1_ps` | Broadcast | Fill register with scalar |
| `_mm256_hadd_ps` | Horizontal add | For reductions |

---

### 2. Memory Alignment

**Concept:** Align data to cache line / SIMD register boundaries for faster access.

```cpp
// Unaligned: May cross cache line boundary (slow)
float* data = new float[N];

// Aligned: Starts at 32-byte boundary (fast)
float* data = (float*)aligned_alloc(32, N * sizeof(float));

// C++11 alignas
alignas(32) float data[256];  // Stack-allocated, aligned

// Custom aligned allocator
void* aligned_alloc(size_t size, size_t alignment) {
#ifdef _MSC_VER
    return _aligned_malloc(size, alignment);
#else
    void* ptr;
    posix_memalign(&ptr, alignment, size);
    return ptr;
#endif
}
```

**Why alignment matters:**
- Aligned loads (`_mm256_load_ps`) can be faster than unaligned
- Cache lines are 64 bytes; aligned data doesn't split across lines
- Some older SIMD instructions *require* alignment (crash otherwise)

---

### 3. Cache Blocking (Tiling)

**Concept:** Process data in blocks that fit in cache to minimize memory bandwidth bottleneck.

**The problem:**
```
Naive matmul: C[i,j] += A[i,k] * B[k,j]
- Inner loop accesses B column-wise
- Each B[k,j] access misses cache
- Memory bound, not compute bound
```

**The solution - Cache blocking:**
```cpp
// Process in tiles that fit in L1 cache (~32KB)
// 3 * BLOCK^2 * sizeof(float) < L1_SIZE
// BLOCK ≈ 50-64 for L1

for (size_t ii = 0; ii < M; ii += BLOCK) {
    for (size_t kk = 0; kk < K; kk += BLOCK) {
        for (size_t jj = 0; jj < N; jj += BLOCK) {
            // Process BLOCK x BLOCK tile
            // All data fits in L1 cache!
            for (size_t i = ii; i < ii + BLOCK; ++i) {
                for (size_t k = kk; k < kk + BLOCK; ++k) {
                    for (size_t j = jj; j < jj + BLOCK; ++j) {
                        C[i*N+j] += A[i*K+k] * B[k*N+j];
                    }
                }
            }
        }
    }
}
```

**Cache hierarchy:**
| Level | Size | Latency | Bandwidth |
|-------|------|---------|-----------|
| L1 | 32KB | ~4 cycles | ~1 TB/s |
| L2 | 256KB | ~12 cycles | ~500 GB/s |
| L3 | 8MB | ~40 cycles | ~200 GB/s |
| RAM | 64GB | ~200 cycles | ~50 GB/s |

---

### 4. Loop Reordering

**Concept:** Reorder nested loops for sequential memory access.

```cpp
// Bad: Column access in inner loop (stride = N)
for (i) for (j) for (k)
    C[i,j] += A[i,k] * B[k,j]  // B accessed column-wise

// Better: Row access in inner loop (stride = 1)
for (i) for (k) for (j)
    C[i,j] += A[i,k] * B[k,j]  // B accessed row-wise
```

**Why ikj order works:**
- `A[i,k]` is constant in inner loop → broadcast to SIMD register
- `B[k,j]` accessed sequentially → good cache utilization
- `C[i,j]` accessed sequentially → good write pattern

---

### 5. OpenMP Parallelization

**Concept:** Distribute work across CPU cores with compiler directives.

```cpp
// Serial (1 core)
for (size_t i = 0; i < N; ++i) {
    process(i);
}

// Parallel (all cores)
#pragma omp parallel for
for (size_t i = 0; i < N; ++i) {
    process(i);  // Each core handles different i values
}

// Nested parallelism with collapse
#pragma omp parallel for collapse(2) schedule(dynamic)
for (size_t i = 0; i < M; i += BLOCK) {
    for (size_t j = 0; j < N; j += BLOCK) {
        // Blocks distributed across cores
    }
}
```

**OpenMP pragmas used:**
| Pragma | Effect |
|--------|--------|
| `parallel for` | Parallelize loop iterations |
| `collapse(2)` | Treat 2 nested loops as single parallel region |
| `schedule(dynamic)` | Dynamic load balancing |
| `reduction(+:sum)` | Parallel reduction |

---

### 6. Fused Multiply-Add (FMA)

**Concept:** Compute `a * b + c` in a single instruction with better precision.

```cpp
// Without FMA: 2 operations, 2 roundings
float result = a * b;
result = result + c;

// With FMA: 1 operation, 1 rounding
__m256 result = _mm256_fmadd_ps(va, vb, vc);
```

**Benefits:**
- 2x throughput (1 instruction instead of 2)
- Better precision (only 1 rounding error)
- Critical for matmul inner loop

---

## 🔑 Key Optimization Techniques Summary

| Technique | Speedup | When to Use |
|-----------|---------|-------------|
| SIMD (AVX2) | 4-8x | Element-wise ops, reductions |
| Cache blocking | 5-20x | Matrix operations |
| Loop reordering | 2-4x | Nested loops with memory access |
| Memory alignment | 1.1-1.5x | All SIMD code |
| OpenMP | Nx for N cores | Large parallel regions |
| FMA | ~2x | Multiply-add heavy code |

---

## 📊 Benchmark Results

**Matrix Multiply (1024x1024):**

| Implementation | Time | GFLOPS | Speedup |
|----------------|------|--------|---------|
| Naive (ijk) | 4800 ms | 0.45 | 1.0x |
| Reordered (ikj) | 1200 ms | 1.8 | 4.0x |
| + SIMD (AVX2) | 180 ms | 12 | 27x |
| + Blocking | 95 ms | 23 | 50x |
| + OpenMP (8 core) | 15 ms | 143 | 320x |
| OpenBLAS | 8 ms | 268 | 600x |

---

## 🔗 Zuup/Visionblox Application

| Technique | Platform Application |
|-----------|---------------------|
| **SIMD** | Orb: Gaussian Splatting render kernel |
| **Cache Blocking** | Symbion: Real-time biosignal filtering |
| **OpenMP** | PodX: Parallel data processing pipelines |
| **Alignment** | Veyra: Low-latency inference buffers |
| **FMA** | QAWM: Quantum state vector operations |

---

## 📝 Compiler Flags

```bash
# GCC/Clang - Maximum optimization
g++ -O3 -march=native -mtune=native -ffast-math -fopenmp

# MSVC - Maximum optimization
cl /O2 /arch:AVX2 /fp:fast /openmp

# CMake
target_compile_options(target PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:
        -O3 -march=native -ffast-math -fopenmp>
    $<$<CXX_COMPILER_ID:MSVC>:
        /O2 /arch:AVX2 /fp:fast /openmp>
)
```

---

## ⚠️ Common Pitfalls

1. **Denormals:** Flush denormals to zero for consistent performance
   ```cpp
   _mm_setcsr(_mm_getcsr() | 0x8040);  // FTZ + DAZ
   ```

2. **Aliasing:** Use `__restrict` to help compiler optimize
   ```cpp
   void add(float* __restrict c, const float* __restrict a, const float* __restrict b);
   ```

3. **False sharing:** Align thread-local data to cache line (64 bytes)

4. **Branch misprediction:** Use SIMD masks instead of branches in hot loops

---

## ➡️ Next Phase Preview

**Phase 5: Extensions** will cover:
- CUDA GPU acceleration
- Sparse tensor support
- Custom allocators (memory pools)
- Quantization (INT8, FP16)

---

## 📚 Further Reading

1. **Fog, Agner** "Optimizing software in C++" (agner.org)
2. **Intel Intrinsics Guide** (software.intel.com/intrinsics-guide)
3. **What Every Programmer Should Know About Memory** - Drepper

---

*Generated for MicroGrad++ learning project*
