# CLAUDE.md - AI Assistant Context for MicroGrad++

## Project Purpose
MicroGrad++ is a **learning project** that teaches C++ from undergrad to PhD level
through building an autodiff tensor library. It is NOT production ML code—it's
pedagogical code optimized for learning modern C++ idioms.

## Quick Start
```bash
# Build with sanitizers (recommended for development)
cmake -B build -DCMAKE_BUILD_TYPE=Debug -DMICROGRAD_ENABLE_SANITIZERS=ON
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure

# Build release
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

## Repository Layout
```
micrograd-cpp/
├── include/micrograd/    # Header-only library
│   ├── tensor.hpp        # PHASE1: Core tensor class
│   ├── view.hpp          # PHASE1: Non-owning views
│   ├── shape.hpp         # PHASE1: Shape utilities
│   ├── expr.hpp          # PHASE2: Expression templates
│   ├── ops/              # PHASE2: Lazy operations
│   ├── autograd/         # PHASE3: Autodiff engine
│   └── optim/            # PHASE4: SIMD/threading
├── tests/                # Test files per module
├── benchmarks/           # Performance benchmarks
├── docs/                 # Learning documentation
│   └── learning_journal.html  # Phase outcomes + Zuup mappings
├── examples/             # Usage examples
├── .github/workflows/    # CI configuration
├── .cursorrules          # Cursor IDE AI rules
└── CLAUDE.md             # This file
```

## Learning Phases

### Phase 1: Foundations (Undergrad)
**Files**: `tensor.hpp`, `view.hpp`, `shape.hpp`
**Concepts**: RAII, Rule of 5, templates, operator overloading
**Test**: `Tensor<float> a({2,3}); a.fill(1.0f);`

### Phase 2: Expression Templates (Grad)
**Files**: `expr.hpp`, `ops/*.hpp`
**Concepts**: CRTP, lazy evaluation, zero-copy operations
**Test**: `a + b + c` compiles to single loop

### Phase 3: Autodiff (PhD Core)
**Files**: `autograd/*.hpp`
**Concepts**: Computational graphs, reverse-mode AD, chain rule
**Test**: Train MLP on XOR, gradients match PyTorch ±1e-5

### Phase 4: Performance (Grad/PhD)
**Files**: `optim/*.hpp`
**Concepts**: SIMD intrinsics, threading, custom allocators
**Test**: 10x speedup on matmul vs naive

### Phase 5: Extensions (PhD Novel)
**Options**: CUDA backend, sparse tensors, quantization, graph optimization

## Key Design Decisions

### Why Header-Only?
- Simplifies build for learners
- Enables aggressive inlining
- No link-time issues with templates

### Why Expression Templates?
- Eliminates temporary allocations
- Teaches advanced template metaprogramming
- Used in Eigen, Blaze (industry standard)

### Why Tape-Based Autodiff?
- Clearer than operator overloading approaches
- Matches PyTorch's design philosophy
- Easier to debug and extend

## Common Tasks

### Add a New Operation
1. Create header in `include/micrograd/ops/`
2. Inherit from `Expr<YourOp>`
3. Implement `operator[]` and `size()`
4. Add `backward_fn` for autodiff support
5. Add tests in `tests/test_ops.cpp`

### Add SIMD Kernel
1. Add to `include/micrograd/optim/simd.hpp`
2. Guard with `#ifdef MICROGRAD_HAS_AVX2`
3. Include scalar fallback
4. Benchmark in `benchmarks/`

### Debug Memory Issues
```bash
cmake -B build -DMICROGRAD_ENABLE_SANITIZERS=ON
cmake --build build
./build/tests/test_tensor  # ASan will report issues
```

## Zuup Platform Applications

This project teaches patterns directly applicable to the Zuup ecosystem:

| C++ Technique | Zuup Platform | Application |
|---------------|---------------|-------------|
| RAII + Rule of 5 | All platforms | Resource lifecycle management |
| Expression templates | Orb | Deferred 3DGS computations |
| Custom allocators | Symbion | Real-time memory pools |
| SIMD optimization | Orb, QAWM | Vectorized world model ops |
| Tape-based graphs | Veyra | Distributed computation DAGs |
| Type-safe IDs | Aureon | Procurement entity references |
| Immutable structures | Civium | Audit trail attestations |
| Lock-free queues | PodX | Offline-first message passing |

## AI Assistant Guidelines

### When Helping with This Project
1. **Explain the "why"** - This is a learning project
2. **Show assembly** when demonstrating optimization
3. **Reference phases** - Tag suggestions with PHASE1/2/3/4/5
4. **Connect to Zuup** - Note when techniques apply to specific platforms
5. **Prefer correctness** over cleverness for Phase 1-2

### Code Generation Preferences
- Use `[[nodiscard]]` on value-returning functions
- Use `constexpr` where possible
- Prefer concepts over SFINAE
- Add `static_assert` for invariants
- Include minimal but complete examples

### Don't Assume
- User has debugger experience (teach GDB basics if needed)
- User knows CMake deeply (explain generator expressions)
- User understands template instantiation (explain when relevant)

## External Resources
- [cppreference.com](https://en.cppreference.com) - C++ standard reference
- [Compiler Explorer](https://godbolt.org) - View generated assembly
- [micrograd](https://github.com/karpathy/micrograd) - Python reference impl
- [Eigen docs](https://eigen.tuxfamily.org/dox/) - Expression template reference

## Contact
Part of the Zuup Innovation Lab ecosystem.
Owner: Khaalis (Visionblox LLC)
