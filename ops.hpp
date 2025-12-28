// include/micrograd/ops.hpp
// Element-wise operations for tensors
#pragma once

#include "tensor.hpp"
#include <cmath>
#include <functional>

namespace micrograd {

// =============================================================================
// Binary element-wise operations
// =============================================================================

/// Element-wise addition
template<typename T>
[[nodiscard]] Tensor<T> operator+(const Tensor<T>& a, const Tensor<T>& b) {
    assert(a.shape() == b.shape() && "Shapes must match for addition");
    
    Tensor<T> result(a.shape());
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] + b[i];
    }
    return result;
}

/// Element-wise subtraction
template<typename T>
[[nodiscard]] Tensor<T> operator-(const Tensor<T>& a, const Tensor<T>& b) {
    assert(a.shape() == b.shape() && "Shapes must match for subtraction");
    
    Tensor<T> result(a.shape());
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] - b[i];
    }
    return result;
}

/// Element-wise multiplication (Hadamard product)
template<typename T>
[[nodiscard]] Tensor<T> operator*(const Tensor<T>& a, const Tensor<T>& b) {
    assert(a.shape() == b.shape() && "Shapes must match for multiplication");
    
    Tensor<T> result(a.shape());
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] * b[i];
    }
    return result;
}

/// Element-wise division
template<typename T>
[[nodiscard]] Tensor<T> operator/(const Tensor<T>& a, const Tensor<T>& b) {
    assert(a.shape() == b.shape() && "Shapes must match for division");
    
    Tensor<T> result(a.shape());
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] / b[i];
    }
    return result;
}

// =============================================================================
// Scalar operations
// =============================================================================

/// Tensor + scalar
template<typename T>
[[nodiscard]] Tensor<T> operator+(const Tensor<T>& a, T scalar) {
    Tensor<T> result(a.shape());
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] + scalar;
    }
    return result;
}

template<typename T>
[[nodiscard]] Tensor<T> operator+(T scalar, const Tensor<T>& a) {
    return a + scalar;
}

/// Tensor - scalar
template<typename T>
[[nodiscard]] Tensor<T> operator-(const Tensor<T>& a, T scalar) {
    Tensor<T> result(a.shape());
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] - scalar;
    }
    return result;
}

template<typename T>
[[nodiscard]] Tensor<T> operator-(T scalar, const Tensor<T>& a) {
    Tensor<T> result(a.shape());
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = scalar - a[i];
    }
    return result;
}

/// Tensor * scalar
template<typename T>
[[nodiscard]] Tensor<T> operator*(const Tensor<T>& a, T scalar) {
    Tensor<T> result(a.shape());
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] * scalar;
    }
    return result;
}

template<typename T>
[[nodiscard]] Tensor<T> operator*(T scalar, const Tensor<T>& a) {
    return a * scalar;
}

/// Tensor / scalar
template<typename T>
[[nodiscard]] Tensor<T> operator/(const Tensor<T>& a, T scalar) {
    Tensor<T> result(a.shape());
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] / scalar;
    }
    return result;
}

template<typename T>
[[nodiscard]] Tensor<T> operator/(T scalar, const Tensor<T>& a) {
    Tensor<T> result(a.shape());
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = scalar / a[i];
    }
    return result;
}

// =============================================================================
// Unary operations
// =============================================================================

/// Negation
template<typename T>
[[nodiscard]] Tensor<T> operator-(const Tensor<T>& a) {
    Tensor<T> result(a.shape());
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = -a[i];
    }
    return result;
}

/// Exponential
template<typename T>
[[nodiscard]] Tensor<T> exp(const Tensor<T>& a) {
    Tensor<T> result(a.shape());
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = std::exp(a[i]);
    }
    return result;
}

/// Natural logarithm
template<typename T>
[[nodiscard]] Tensor<T> log(const Tensor<T>& a) {
    Tensor<T> result(a.shape());
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = std::log(a[i]);
    }
    return result;
}

/// Square root
template<typename T>
[[nodiscard]] Tensor<T> sqrt(const Tensor<T>& a) {
    Tensor<T> result(a.shape());
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = std::sqrt(a[i]);
    }
    return result;
}

/// Power
template<typename T>
[[nodiscard]] Tensor<T> pow(const Tensor<T>& a, T exponent) {
    Tensor<T> result(a.shape());
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = std::pow(a[i], exponent);
    }
    return result;
}

/// Absolute value
template<typename T>
[[nodiscard]] Tensor<T> abs(const Tensor<T>& a) {
    Tensor<T> result(a.shape());
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = std::abs(a[i]);
    }
    return result;
}

// =============================================================================
// Activation functions
// =============================================================================

/// ReLU: max(0, x)
template<typename T>
[[nodiscard]] Tensor<T> relu(const Tensor<T>& a) {
    Tensor<T> result(a.shape());
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = std::max(T{0}, a[i]);
    }
    return result;
}

/// Sigmoid: 1 / (1 + exp(-x))
template<typename T>
[[nodiscard]] Tensor<T> sigmoid(const Tensor<T>& a) {
    Tensor<T> result(a.shape());
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = T{1} / (T{1} + std::exp(-a[i]));
    }
    return result;
}

/// Tanh
template<typename T>
[[nodiscard]] Tensor<T> tanh(const Tensor<T>& a) {
    Tensor<T> result(a.shape());
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = std::tanh(a[i]);
    }
    return result;
}

// =============================================================================
// Compound assignment operators
// =============================================================================

template<typename T>
Tensor<T>& operator+=(Tensor<T>& a, const Tensor<T>& b) {
    assert(a.shape() == b.shape());
    for (size_t i = 0; i < a.size(); ++i) {
        a[i] += b[i];
    }
    return a;
}

template<typename T>
Tensor<T>& operator-=(Tensor<T>& a, const Tensor<T>& b) {
    assert(a.shape() == b.shape());
    for (size_t i = 0; i < a.size(); ++i) {
        a[i] -= b[i];
    }
    return a;
}

template<typename T>
Tensor<T>& operator*=(Tensor<T>& a, const Tensor<T>& b) {
    assert(a.shape() == b.shape());
    for (size_t i = 0; i < a.size(); ++i) {
        a[i] *= b[i];
    }
    return a;
}

template<typename T>
Tensor<T>& operator*=(Tensor<T>& a, T scalar) {
    for (size_t i = 0; i < a.size(); ++i) {
        a[i] *= scalar;
    }
    return a;
}

// =============================================================================
// Comparison (element-wise, returns bool tensor)
// =============================================================================

template<typename T>
[[nodiscard]] Tensor<bool> operator==(const Tensor<T>& a, const Tensor<T>& b) {
    assert(a.shape() == b.shape());
    Tensor<bool> result(a.shape());
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = (a[i] == b[i]);
    }
    return result;
}

template<typename T>
[[nodiscard]] Tensor<bool> operator<(const Tensor<T>& a, const Tensor<T>& b) {
    assert(a.shape() == b.shape());
    Tensor<bool> result(a.shape());
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = (a[i] < b[i]);
    }
    return result;
}

// =============================================================================
// Apply arbitrary function
// =============================================================================

/// Apply unary function element-wise
template<typename T, typename Func>
[[nodiscard]] Tensor<T> apply(const Tensor<T>& a, Func&& func) {
    Tensor<T> result(a.shape());
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = func(a[i]);
    }
    return result;
}

/// Apply binary function element-wise
template<typename T, typename Func>
[[nodiscard]] Tensor<T> apply(const Tensor<T>& a, const Tensor<T>& b, Func&& func) {
    assert(a.shape() == b.shape());
    Tensor<T> result(a.shape());
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = func(a[i], b[i]);
    }
    return result;
}

// =============================================================================
// Matrix operations (basic, 2D only)
// =============================================================================

/// Matrix multiplication for 2D tensors
template<typename T>
[[nodiscard]] Tensor<T> matmul(const Tensor<T>& a, const Tensor<T>& b) {
    assert(a.ndim() == 2 && b.ndim() == 2 && "matmul requires 2D tensors");
    assert(a.shape()[1] == b.shape()[0] && "Inner dimensions must match");
    
    const size_t M = a.shape()[0];
    const size_t K = a.shape()[1];
    const size_t N = b.shape()[1];
    
    Tensor<T> result(Shape({M, N}));
    result.zero();
    
    // Naive O(n³) implementation - will optimize in Phase 4
    for (size_t i = 0; i < M; ++i) {
        for (size_t k = 0; k < K; ++k) {
            for (size_t j = 0; j < N; ++j) {
                result(i, j) += a(i, k) * b(k, j);
            }
        }
    }
    return result;
}

/// Transpose 2D tensor
template<typename T>
[[nodiscard]] Tensor<T> transpose(const Tensor<T>& a) {
    assert(a.ndim() == 2 && "transpose requires 2D tensor");
    
    const size_t M = a.shape()[0];
    const size_t N = a.shape()[1];
    
    Tensor<T> result(Shape({N, M}));
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            result(j, i) = a(i, j);
        }
    }
    return result;
}

} // namespace micrograd
