// include/micrograd/micrograd.hpp
// MicroGrad++ - Educational C++ autodiff tensor library
// 
// Umbrella header: includes all public API components
#pragma once

#include "shape.hpp"
#include "tensor.hpp"
#include "ops.hpp"

// Phase 2: Expression templates
// #include "expr.hpp"

// Phase 3: Autodiff
// #include "autograd/variable.hpp"
// #include "autograd/tape.hpp"

// Phase 4: Performance
// #include "simd/dispatch.hpp"

namespace micrograd {

/// Library version
inline constexpr struct {
    int major = 0;
    int minor = 1;
    int patch = 0;
} version;

} // namespace micrograd
