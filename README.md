# MicroGrad++

A PhD-level educational C++ autodiff tensor library. Learn C++ techniques from undergrad to PhD by building a functional ML framework from scratch.

## Quick Start

```bash
# Clone and build
git clone https://github.com/<you>/micrograd-cpp
cd micrograd-cpp

# Configure and build
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# Run tests
ctest --test-dir build --output-on-failure

# Run benchmarks
./build/benchmarks/bench_tensor
```

## Project Structure

```
micrograd-cpp/
├── include/micrograd/
│   ├── micrograd.hpp      # Umbrella header
│   ├── tensor.hpp         # Core tensor class
│   ├── shape.hpp          # Shape utilities
│   └── ops.hpp            # Operations
├── tests/                 # Google Test unit tests
├── benchmarks/            # Google Benchmark performance tests
├── docs/
│   └── learning-journey.html  # Learning documentation
├── .cursorrules           # Cursor AI configuration
├── CLAUDE.md              # Claude AI context
└── CMakeLists.txt         # Build configuration
```

## Learning Phases

| Phase | Level | Focus |
|-------|-------|-------|
| 1 | Undergrad | RAII, Rule of Five, Move Semantics, Templates |
| 2 | Graduate | CRTP, Expression Templates, SFINAE/Concepts |
| 3 | PhD | Autodiff, Computational Graphs, Backpropagation |
| 4 | Grad/PhD | SIMD, Cache Optimization, Multithreading |
| 5 | PhD | GPU/Sparse/Quantization (choose one) |

## Usage

```cpp
#include <micrograd/micrograd.hpp>

using namespace micrograd;

int main() {
    // Create tensors
    auto a = Tensor<float>::ones(Shape({2, 3}));
    auto b = Tensor<float>::ones(Shape({2, 3}));
    
    // Element-wise operations
    auto c = a + b;
    auto d = relu(c * 2.0f);
    
    // Matrix operations
    auto W = Tensor<float>::ones(Shape({3, 4}));
    auto x = Tensor<float>::ones(Shape({2, 3}));
    auto y = matmul(x, W);  // 2x4 result
    
    std::cout << "Result: " << y << std::endl;
    return 0;
}
```

## AI-Assisted Development

This project is optimized for Cursor Ultra and Claude Max:

- `.cursorrules` - Cursor AI configuration with project context
- `CLAUDE.md` - Claude context file with architecture overview

## Requirements

- C++20 compiler (GCC 13+, Clang 17+)
- CMake 3.16+
- (Optional) Ninja build system

## License

MIT License - Educational use encouraged.

## Zuup Innovation Lab

This project demonstrates techniques applicable to:
- **Aureon™** - Procurement AI embedding systems
- **Orb™** - 3D Gaussian Splatting optimization
- **Symbion™** - Biosignal processing pipelines
- **Veyra™** - Reinforcement learning frameworks
