// include/micrograd/tensor.hpp
// PHASE1: Core tensor class - RAII, Rule of 5, templates, operator overloading
#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <initializer_list>
#include <memory>
#include <numeric>
#include <ostream>
#include <span>
#include <stdexcept>
#include <vector>

namespace micrograd {

// =============================================================================
// Shape: Represents tensor dimensions
// PHASE1: Value semantics, STL container usage
// =============================================================================
class Shape {
public:
    using Dims = std::vector<size_t>;

    Shape() = default;
    Shape(std::initializer_list<size_t> dims) : dims_(dims) {}
    explicit Shape(Dims dims) : dims_(std::move(dims)) {}

    [[nodiscard]] size_t rank() const noexcept { return dims_.size(); }
    [[nodiscard]] size_t ndim() const noexcept { return dims_.size(); }
    [[nodiscard]] size_t operator[](size_t i) const { return dims_.at(i); }
    
    [[nodiscard]] size_t numel() const noexcept {
        if (dims_.empty()) return 0;
        return std::accumulate(dims_.begin(), dims_.end(), 
                               size_t{1}, std::multiplies<>());
    }

    [[nodiscard]] const Dims& dims() const noexcept { return dims_; }
    
    [[nodiscard]] bool operator==(const Shape& other) const {
        return dims_ == other.dims_;
    }

    friend std::ostream& operator<<(std::ostream& os, const Shape& s) {
        os << "(";
        for (size_t i = 0; i < s.dims_.size(); ++i) {
            if (i > 0) os << ", ";
            os << s.dims_[i];
        }
        return os << ")";
    }

private:
    Dims dims_;
};

// =============================================================================
// Tensor: Dense N-dimensional array
// PHASE1: RAII, Rule of 5, operator overloading
// =============================================================================
template<typename T = float>
class Tensor {
public:
    // -------------------------------------------------------------------------
    // Constructors (Rule of 5)
    // -------------------------------------------------------------------------
    
    // Default constructor: empty tensor
    Tensor() = default;

    // Construct with shape, uninitialized data
    explicit Tensor(Shape shape) 
        : shape_(std::move(shape))
        , data_(std::make_unique<T[]>(shape_.numel())) 
    {}

    // Construct with shape and fill value
    Tensor(Shape shape, T fill_value) 
        : shape_(std::move(shape))
        , data_(std::make_unique<T[]>(shape_.numel())) 
    {
        fill(fill_value);
    }

    // Destructor: unique_ptr handles cleanup (RAII)
    ~Tensor() = default;

    // Copy constructor: deep copy
    Tensor(const Tensor& other) 
        : shape_(other.shape_)
        , data_(std::make_unique<T[]>(other.size())) 
    {
        std::copy(other.data_.get(), other.data_.get() + other.size(), data_.get());
    }

    // Copy assignment: copy-and-swap idiom
    Tensor& operator=(const Tensor& other) {
        if (this != &other) {
            Tensor tmp(other);
            swap(*this, tmp);
        }
        return *this;
    }

    // Move constructor: transfer ownership
    Tensor(Tensor&& other) noexcept 
        : shape_(std::move(other.shape_))
        , data_(std::move(other.data_)) 
    {}

    // Move assignment
    Tensor& operator=(Tensor&& other) noexcept {
        if (this != &other) {
            shape_ = std::move(other.shape_);
            data_ = std::move(other.data_);
        }
        return *this;
    }

    // Friend swap for copy-and-swap idiom
    friend void swap(Tensor& a, Tensor& b) noexcept {
        using std::swap;
        swap(a.shape_, b.shape_);
        swap(a.data_, b.data_);
    }

    // -------------------------------------------------------------------------
    // Accessors
    // -------------------------------------------------------------------------
    
    [[nodiscard]] size_t size() const noexcept { return shape_.numel(); }
    [[nodiscard]] const Shape& shape() const noexcept { return shape_; }
    [[nodiscard]] size_t rank() const noexcept { return shape_.rank(); }
    [[nodiscard]] size_t ndim() const noexcept { return shape_.ndim(); }
    [[nodiscard]] bool empty() const noexcept { return size() == 0; }
    
    // Raw pointer access
    [[nodiscard]] T* data() noexcept { return data_.get(); }
    [[nodiscard]] const T* data() const noexcept { return data_.get(); }
    
    // Span view (C++20)
    [[nodiscard]] std::span<T> span() noexcept { return {data_.get(), size()}; }
    [[nodiscard]] std::span<const T> span() const noexcept { return {data_.get(), size()}; }

    // -------------------------------------------------------------------------
    // Element Access
    // -------------------------------------------------------------------------
    
    // Flat indexing
    [[nodiscard]] T& operator[](size_t i) { 
        assert(i < size() && "Index out of bounds");
        return data_[i]; 
    }
    
    [[nodiscard]] const T& operator[](size_t i) const { 
        assert(i < size() && "Index out of bounds");
        return data_[i]; 
    }

    // Multi-dimensional indexing (row-major)
    [[nodiscard]] T& at(std::initializer_list<size_t> indices) {
        return data_[flat_index(indices)];
    }

    [[nodiscard]] const T& at(std::initializer_list<size_t> indices) const {
        return data_[flat_index(indices)];
    }

    // 2D indexing via operator() for convenience (row-major)
    [[nodiscard]] T& operator()(size_t i, size_t j) {
        assert(shape_.rank() == 2 && "operator(i,j) requires 2D tensor");
        assert(i < shape_[0] && j < shape_[1] && "Index out of bounds");
        return data_[i * shape_[1] + j];
    }

    [[nodiscard]] const T& operator()(size_t i, size_t j) const {
        assert(shape_.rank() == 2 && "operator(i,j) requires 2D tensor");
        assert(i < shape_[0] && j < shape_[1] && "Index out of bounds");
        return data_[i * shape_[1] + j];
    }

    // -------------------------------------------------------------------------
    // Modifiers
    // -------------------------------------------------------------------------
    
    void fill(T value) {
        std::fill(data_.get(), data_.get() + size(), value);
    }

    void zero() { fill(T{0}); }

    // -------------------------------------------------------------------------
    // Reductions
    // -------------------------------------------------------------------------

    [[nodiscard]] T sum() const {
        return std::accumulate(begin(), end(), T{0});
    }

    [[nodiscard]] T mean() const {
        return sum() / static_cast<T>(size());
    }

    [[nodiscard]] T max() const {
        return *std::max_element(begin(), end());
    }

    [[nodiscard]] T min() const {
        return *std::min_element(begin(), end());
    }

    // -------------------------------------------------------------------------
    // Factory Methods
    // -------------------------------------------------------------------------
    
    [[nodiscard]] static Tensor zeros(Shape shape) {
        return Tensor(std::move(shape), T{0});
    }

    [[nodiscard]] static Tensor ones(Shape shape) {
        return Tensor(std::move(shape), T{1});
    }

    [[nodiscard]] static Tensor from_list(std::initializer_list<T> values) {
        Tensor t(Shape({values.size()}));
        std::copy(values.begin(), values.end(), t.data());
        return t;
    }

    [[nodiscard]] static Tensor from_list(std::initializer_list<T> values, Shape shape) {
        assert(values.size() == shape.numel() && "Size mismatch");
        Tensor t(std::move(shape));
        std::copy(values.begin(), values.end(), t.data());
        return t;
    }

    // -------------------------------------------------------------------------
    // Iterators (for range-based for loops)
    // -------------------------------------------------------------------------
    
    [[nodiscard]] T* begin() noexcept { return data_.get(); }
    [[nodiscard]] T* end() noexcept { return data_.get() + size(); }
    [[nodiscard]] const T* begin() const noexcept { return data_.get(); }
    [[nodiscard]] const T* end() const noexcept { return data_.get() + size(); }

private:
    // Compute flat index from multi-dimensional indices (row-major order)
    [[nodiscard]] size_t flat_index(std::initializer_list<size_t> indices) const {
        if (indices.size() != shape_.rank()) {
            throw std::out_of_range("Index dimension mismatch");
        }
        
        size_t flat = 0;
        size_t stride = 1;
        auto idx_it = std::rbegin(indices);
        
        for (size_t d = shape_.rank(); d-- > 0; ++idx_it) {
            if (*idx_it >= shape_[d]) {
                throw std::out_of_range("Index out of bounds");
            }
            flat += *idx_it * stride;
            stride *= shape_[d];
        }
        return flat;
    }

    Shape shape_;
    std::unique_ptr<T[]> data_;
};

// =============================================================================
// Element-wise Operations (Eager)
// PHASE1: Operator overloading, creates new tensors
// Note: PHASE2 replaces these with expression templates for zero-copy
// =============================================================================

template<typename T>
[[nodiscard]] Tensor<T> operator+(const Tensor<T>& a, const Tensor<T>& b) {
    assert(a.shape() == b.shape() && "Shape mismatch");
    Tensor<T> result(a.shape());
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] + b[i];
    }
    return result;
}

template<typename T>
[[nodiscard]] Tensor<T> operator-(const Tensor<T>& a, const Tensor<T>& b) {
    assert(a.shape() == b.shape() && "Shape mismatch");
    Tensor<T> result(a.shape());
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] - b[i];
    }
    return result;
}

template<typename T>
[[nodiscard]] Tensor<T> operator*(const Tensor<T>& a, const Tensor<T>& b) {
    assert(a.shape() == b.shape() && "Shape mismatch");
    Tensor<T> result(a.shape());
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] * b[i];
    }
    return result;
}

template<typename T>
[[nodiscard]] Tensor<T> operator/(const Tensor<T>& a, const Tensor<T>& b) {
    assert(a.shape() == b.shape() && "Shape mismatch");
    Tensor<T> result(a.shape());
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] / b[i];
    }
    return result;
}

// Scalar operations
template<typename T>
[[nodiscard]] Tensor<T> operator+(const Tensor<T>& a, T scalar) {
    Tensor<T> result(a.shape());
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] + scalar;
    }
    return result;
}

template<typename T>
[[nodiscard]] Tensor<T> operator*(const Tensor<T>& a, T scalar) {
    Tensor<T> result(a.shape());
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] * scalar;
    }
    return result;
}

// Printing
template<typename T>
std::ostream& operator<<(std::ostream& os, const Tensor<T>& t) {
    os << "Tensor" << t.shape() << "[";
    const size_t max_print = 10;
    for (size_t i = 0; i < std::min(t.size(), max_print); ++i) {
        if (i > 0) os << ", ";
        os << t[i];
    }
    if (t.size() > max_print) os << ", ...";
    return os << "]";
}

} // namespace micrograd

