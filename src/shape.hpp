// src/shape.hpp
// Shape and stride utilities for N-dimensional tensors
// Used by the SIMD module
#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <initializer_list>
#include <numeric>
#include <string>
#include <vector>

namespace micrograd {
namespace simd {

// =============================================================================
// Shape Type
// =============================================================================

using Shape = std::vector<size_t>;
using Strides = std::vector<size_t>;

/// @brief Compute total number of elements in shape
[[nodiscard]] inline size_t shape_size(const Shape& shape) {
    if (shape.empty()) return 0;
    return std::accumulate(shape.begin(), shape.end(), size_t{1}, std::multiplies<>{});
}

/// @brief Compute strides for row-major (C-style) layout
[[nodiscard]] inline Strides shape_strides(const Shape& shape) {
    if (shape.empty()) return {};
    Strides strides(shape.size());
    strides.back() = 1;
    for (size_t i = shape.size() - 1; i > 0; --i) {
        strides[i - 1] = strides[i] * shape[i];
    }
    return strides;
}

/// @brief Convert multi-dimensional indices to flat index
[[nodiscard]] inline size_t ravel_index(const std::vector<size_t>& indices, const Shape& shape) {
    assert(indices.size() == shape.size());
    Strides strides = shape_strides(shape);
    size_t flat = 0;
    for (size_t i = 0; i < indices.size(); ++i) {
        flat += indices[i] * strides[i];
    }
    return flat;
}

/// @brief Convert shape to string representation
[[nodiscard]] inline std::string shape_to_string(const Shape& shape) {
    std::string s = "(";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) s += ", ";
        s += std::to_string(shape[i]);
    }
    s += ")";
    return s;
}

} // namespace simd
} // namespace micrograd

