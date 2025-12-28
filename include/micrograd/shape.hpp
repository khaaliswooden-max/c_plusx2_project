// include/micrograd/shape.hpp
// Shape and stride utilities for N-dimensional tensors
#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <initializer_list>
#include <numeric>
#include <ostream>
#include <span>
#include <vector>

namespace micrograd {

inline constexpr size_t MAX_DIMS = 8;

class Shape {
public:
    using Dims = std::vector<size_t>;
    
    Shape() = default;
    explicit Shape(std::initializer_list<size_t> dims) : dims_(dims) {}
    explicit Shape(std::span<const size_t> dims) : dims_(dims.begin(), dims.end()) {}
    explicit Shape(Dims dims) : dims_(std::move(dims)) {}
    
    [[nodiscard]] size_t ndim() const noexcept { return dims_.size(); }
    [[nodiscard]] size_t numel() const noexcept {
        if (dims_.empty()) return 0;
        return std::accumulate(dims_.begin(), dims_.end(), size_t{1}, std::multiplies<>{});
    }
    
    [[nodiscard]] size_t operator[](size_t i) const {
        assert(i < dims_.size());
        return dims_[i];
    }
    
    [[nodiscard]] size_t at(int i) const {
        if (i < 0) i += static_cast<int>(dims_.size());
        return dims_[static_cast<size_t>(i)];
    }
    
    [[nodiscard]] const Dims& dims() const noexcept { return dims_; }
    [[nodiscard]] std::span<const size_t> span() const noexcept { return {dims_.data(), dims_.size()}; }
    
    [[nodiscard]] auto begin() const noexcept { return dims_.begin(); }
    [[nodiscard]] auto end() const noexcept { return dims_.end(); }
    [[nodiscard]] bool operator==(const Shape& other) const noexcept { return dims_ == other.dims_; }
    
    [[nodiscard]] bool is_broadcastable_with(const Shape& other) const noexcept {
        auto it1 = dims_.rbegin(), it2 = other.dims_.rbegin();
        while (it1 != dims_.rend() && it2 != other.dims_.rend()) {
            if (*it1 != *it2 && *it1 != 1 && *it2 != 1) return false;
            ++it1; ++it2;
        }
        return true;
    }
    
    [[nodiscard]] Shape broadcast_with(const Shape& other) const {
        const size_t max_ndim = std::max(ndim(), other.ndim());
        Dims result(max_ndim);
        for (size_t i = 0; i < max_ndim; ++i) {
            const size_t d1 = (i < ndim()) ? dims_[ndim() - 1 - i] : 1;
            const size_t d2 = (i < other.ndim()) ? other.dims_[other.ndim() - 1 - i] : 1;
            result[max_ndim - 1 - i] = std::max(d1, d2);
        }
        return Shape(std::move(result));
    }

private:
    Dims dims_;
};

[[nodiscard]] inline std::vector<size_t> compute_strides(const Shape& shape) {
    if (shape.ndim() == 0) return {};
    std::vector<size_t> strides(shape.ndim());
    strides.back() = 1;
    for (size_t i = shape.ndim() - 1; i > 0; --i) {
        strides[i - 1] = strides[i] * shape[i];
    }
    return strides;
}

[[nodiscard]] inline size_t ravel_index(std::span<const size_t> indices, std::span<const size_t> strides) noexcept {
    size_t offset = 0;
    for (size_t i = 0; i < indices.size(); ++i) offset += indices[i] * strides[i];
    return offset;
}

[[nodiscard]] inline std::vector<size_t> unravel_index(size_t flat_index, const Shape& shape) {
    std::vector<size_t> indices(shape.ndim());
    for (size_t i = shape.ndim(); i-- > 0;) {
        indices[i] = flat_index % shape[i];
        flat_index /= shape[i];
    }
    return indices;
}

inline std::ostream& operator<<(std::ostream& os, const Shape& shape) {
    os << "(";
    for (size_t i = 0; i < shape.ndim(); ++i) {
        if (i > 0) os << ", ";
        os << shape[i];
    }
    return os << ")";
}

} // namespace micrograd

