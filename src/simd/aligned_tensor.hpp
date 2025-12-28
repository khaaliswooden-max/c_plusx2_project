// filepath: src/simd/aligned_tensor.hpp
// AlignedTensor - SIMD-Optimized Tensor with Aligned Memory
// Phase 4: Performance Optimization
#pragma once

#include "simd_detect.hpp"
#include "simd_ops.hpp"
#include "matmul_fast.hpp"
#include "../shape.hpp"

#include <memory>
#include <cstdlib>
#include <new>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <cassert>

namespace micrograd {
namespace simd {

// ============================================================================
// Aligned Memory Allocator
// ============================================================================

/// @brief Allocate aligned memory
/// @param size Number of bytes
/// @param alignment Alignment boundary (default: SIMD_ALIGNMENT)
/// @return Pointer to aligned memory (nullptr on failure)
[[nodiscard]] inline void* aligned_alloc(size_t size, size_t alignment = SIMD_ALIGNMENT)
{
#ifdef _MSC_VER
    return _aligned_malloc(size, alignment);
#else
    void* ptr = nullptr;
    if (posix_memalign(&ptr, alignment, size) != 0) {
        return nullptr;
    }
    return ptr;
#endif
}

/// @brief Free aligned memory
inline void aligned_free(void* ptr)
{
#ifdef _MSC_VER
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

/// @brief Custom deleter for aligned memory
struct AlignedDeleter
{
    void operator()(float* ptr) const
    {
        aligned_free(ptr);
    }
};

// ============================================================================
// AlignedTensor Class
// ============================================================================

/// @brief Tensor with SIMD-aligned memory and vectorized operations
/// @tparam T Scalar element type (float, double)
/// 
/// Key differences from regular Tensor:
/// - Memory aligned to SIMD boundary (32 bytes for AVX2)
/// - Operations use SIMD intrinsics
/// - Optimal for numerical computation
/// 
/// @example
/// ```cpp
/// AlignedTensor<float> a({1024, 1024});
/// AlignedTensor<float> b({1024, 1024});
/// a.randn();
/// b.randn();
/// 
/// // Uses SIMD+blocked matmul (~10-50x faster)
/// AlignedTensor<float> c = matmul(a, b);
/// ```
template<typename T = float>
class AlignedTensor
{
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                  "AlignedTensor only supports float or double");

public:
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;
    using size_type = size_t;

    // ========================================================================
    // Constructors
    // ========================================================================

    /// @brief Default constructor - empty tensor
    AlignedTensor() noexcept = default;

    /// @brief Construct with shape (uninitialized)
    explicit AlignedTensor(Shape shape)
        : shape_(std::move(shape))
        , strides_(shape_strides(shape_))
        , size_(shape_size(shape_))
    {
        allocate();
    }

    /// @brief Construct from initializer list shape
    explicit AlignedTensor(std::initializer_list<size_t> shape)
        : AlignedTensor(Shape(shape))
    {}

    // ========================================================================
    // Rule of 5
    // ========================================================================

    ~AlignedTensor() = default;  // unique_ptr handles cleanup

    AlignedTensor(const AlignedTensor& other)
        : shape_(other.shape_)
        , strides_(other.strides_)
        , size_(other.size_)
    {
        if (other.data_) {
            allocate();
            std::copy(other.data_.get(), other.data_.get() + size_, data_.get());
        }
    }

    AlignedTensor(AlignedTensor&& other) noexcept
        : shape_(std::move(other.shape_))
        , strides_(std::move(other.strides_))
        , size_(other.size_)
        , data_(std::move(other.data_))
    {
        other.size_ = 0;
    }

    AlignedTensor& operator=(const AlignedTensor& other)
    {
        if (this != &other) {
            AlignedTensor tmp(other);
            swap(*this, tmp);
        }
        return *this;
    }

    AlignedTensor& operator=(AlignedTensor&& other) noexcept
    {
        if (this != &other) {
            shape_ = std::move(other.shape_);
            strides_ = std::move(other.strides_);
            size_ = other.size_;
            data_ = std::move(other.data_);
            other.size_ = 0;
        }
        return *this;
    }

    friend void swap(AlignedTensor& a, AlignedTensor& b) noexcept
    {
        using std::swap;
        swap(a.shape_, b.shape_);
        swap(a.strides_, b.strides_);
        swap(a.size_, b.size_);
        swap(a.data_, b.data_);
    }

    // ========================================================================
    // Factory Methods
    // ========================================================================

    [[nodiscard]] static AlignedTensor zeros(Shape shape)
    {
        AlignedTensor t(std::move(shape));
        t.fill(T{0});
        return t;
    }

    [[nodiscard]] static AlignedTensor ones(Shape shape)
    {
        AlignedTensor t(std::move(shape));
        t.fill(T{1});
        return t;
    }

    [[nodiscard]] static AlignedTensor randn(Shape shape)
    {
        AlignedTensor t(std::move(shape));
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<T> dist(T{0}, T{1});
        for (size_t i = 0; i < t.size_; ++i) {
            t.data_[i] = dist(gen);
        }
        return t;
    }

    [[nodiscard]] static AlignedTensor rand(Shape shape)
    {
        AlignedTensor t(std::move(shape));
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<T> dist(T{0}, T{1});
        for (size_t i = 0; i < t.size_; ++i) {
            t.data_[i] = dist(gen);
        }
        return t;
    }

    // ========================================================================
    // Accessors
    // ========================================================================

    [[nodiscard]] const Shape& shape() const noexcept { return shape_; }
    [[nodiscard]] const Strides& strides() const noexcept { return strides_; }
    [[nodiscard]] size_t size() const noexcept { return size_; }
    [[nodiscard]] size_t ndim() const noexcept { return shape_.size(); }
    [[nodiscard]] bool empty() const noexcept { return size_ == 0; }

    [[nodiscard]] T* data() noexcept { return data_.get(); }
    [[nodiscard]] const T* data() const noexcept { return data_.get(); }

    [[nodiscard]] bool is_aligned() const noexcept
    {
        return micrograd::simd::is_aligned(data_.get());
    }

    // ========================================================================
    // Element Access
    // ========================================================================

    [[nodiscard]] T& operator[](size_t i)
    {
        assert(i < size_);
        return data_[i];
    }

    [[nodiscard]] const T& operator[](size_t i) const
    {
        assert(i < size_);
        return data_[i];
    }

    [[nodiscard]] T& at(const std::vector<size_t>& indices)
    {
        return data_[ravel_index(indices, shape_)];
    }

    [[nodiscard]] const T& at(const std::vector<size_t>& indices) const
    {
        return data_[ravel_index(indices, shape_)];
    }

    // ========================================================================
    // Mutators
    // ========================================================================

    void fill(T value) noexcept
    {
        std::fill(data_.get(), data_.get() + size_, value);
    }

    void zero() noexcept { fill(T{0}); }

    // ========================================================================
    // SIMD-Optimized Operations
    // ========================================================================

    /// @brief SIMD sum
    [[nodiscard]] T sum() const
    {
        if constexpr (std::is_same_v<T, float>) {
            return sum_f32(data_.get(), size_);
        } else {
            // Scalar fallback for double
            T result{0};
            for (size_t i = 0; i < size_; ++i) {
                result += data_[i];
            }
            return result;
        }
    }

    [[nodiscard]] T mean() const
    {
        return size_ > 0 ? sum() / static_cast<T>(size_) : T{0};
    }

    // ========================================================================
    // I/O
    // ========================================================================

    friend std::ostream& operator<<(std::ostream& os, const AlignedTensor& t)
    {
        os << "AlignedTensor" << shape_to_string(t.shape_);
        os << " (aligned=" << (t.is_aligned() ? "yes" : "no") << "):\n  [";
        const size_t max_show = 10;
        for (size_t i = 0; i < std::min(t.size_, max_show); ++i) {
            if (i > 0) os << ", ";
            os << std::setprecision(4) << t[i];
        }
        if (t.size_ > max_show) os << ", ...";
        os << "]\n";
        return os;
    }

private:
    Shape shape_;
    Strides strides_;
    size_t size_{0};
    std::unique_ptr<T[], AlignedDeleter> data_;

    void allocate()
    {
        if (size_ == 0) return;
        
        // Round up to alignment
        size_t aligned_size = align_up(size_ * sizeof(T), SIMD_ALIGNMENT);
        void* ptr = aligned_alloc(aligned_size);
        if (!ptr) {
            throw std::bad_alloc();
        }
        data_.reset(static_cast<T*>(ptr));
    }
};

// ============================================================================
// SIMD-Optimized Free Functions
// ============================================================================

/// @brief SIMD element-wise addition
template<typename T>
[[nodiscard]] AlignedTensor<T> operator+(const AlignedTensor<T>& a, const AlignedTensor<T>& b)
{
    assert(a.shape() == b.shape());
    AlignedTensor<T> result(a.shape());
    
    if constexpr (std::is_same_v<T, float>) {
        add_f32(result.data(), a.data(), b.data(), a.size());
    } else {
        for (size_t i = 0; i < a.size(); ++i) {
            result[i] = a[i] + b[i];
        }
    }
    return result;
}

/// @brief SIMD element-wise subtraction
template<typename T>
[[nodiscard]] AlignedTensor<T> operator-(const AlignedTensor<T>& a, const AlignedTensor<T>& b)
{
    assert(a.shape() == b.shape());
    AlignedTensor<T> result(a.shape());
    
    if constexpr (std::is_same_v<T, float>) {
        sub_f32(result.data(), a.data(), b.data(), a.size());
    } else {
        for (size_t i = 0; i < a.size(); ++i) {
            result[i] = a[i] - b[i];
        }
    }
    return result;
}

/// @brief SIMD element-wise multiplication
template<typename T>
[[nodiscard]] AlignedTensor<T> operator*(const AlignedTensor<T>& a, const AlignedTensor<T>& b)
{
    assert(a.shape() == b.shape());
    AlignedTensor<T> result(a.shape());
    
    if constexpr (std::is_same_v<T, float>) {
        mul_f32(result.data(), a.data(), b.data(), a.size());
    } else {
        for (size_t i = 0; i < a.size(); ++i) {
            result[i] = a[i] * b[i];
        }
    }
    return result;
}

/// @brief SIMD scalar multiplication
template<typename T>
[[nodiscard]] AlignedTensor<T> operator*(const AlignedTensor<T>& a, T scalar)
{
    AlignedTensor<T> result(a.shape());
    
    if constexpr (std::is_same_v<T, float>) {
        scale_f32(result.data(), a.data(), scalar, a.size());
    } else {
        for (size_t i = 0; i < a.size(); ++i) {
            result[i] = a[i] * scalar;
        }
    }
    return result;
}

template<typename T>
[[nodiscard]] AlignedTensor<T> operator*(T scalar, const AlignedTensor<T>& a)
{
    return a * scalar;
}

/// @brief SIMD ReLU
template<typename T>
[[nodiscard]] AlignedTensor<T> relu(const AlignedTensor<T>& a)
{
    AlignedTensor<T> result(a.shape());
    
    if constexpr (std::is_same_v<T, float>) {
        relu_f32(result.data(), a.data(), a.size());
    } else {
        for (size_t i = 0; i < a.size(); ++i) {
            result[i] = std::max(T{0}, a[i]);
        }
    }
    return result;
}

/// @brief Fast matrix multiplication using SIMD + blocking
/// @param a Left matrix (M x K)
/// @param b Right matrix (K x N)
/// @return Result matrix (M x N)
template<typename T>
[[nodiscard]] AlignedTensor<T> matmul(const AlignedTensor<T>& a, const AlignedTensor<T>& b)
{
    assert(a.ndim() == 2 && b.ndim() == 2);
    assert(a.shape()[1] == b.shape()[0]);
    
    size_t M = a.shape()[0];
    size_t K = a.shape()[1];
    size_t N = b.shape()[1];
    
    AlignedTensor<T> result({M, N});
    
    if constexpr (std::is_same_v<T, float>) {
        matmul_auto(result.data(), a.data(), b.data(), M, K, N);
    } else {
        // Scalar fallback for double
        matmul_naive(
            reinterpret_cast<float*>(result.data()),
            reinterpret_cast<const float*>(a.data()),
            reinterpret_cast<const float*>(b.data()),
            M, K, N);
    }
    
    return result;
}

/// @brief SIMD dot product
template<typename T>
[[nodiscard]] T dot(const AlignedTensor<T>& a, const AlignedTensor<T>& b)
{
    assert(a.size() == b.size());
    
    if constexpr (std::is_same_v<T, float>) {
        return dot_f32(a.data(), b.data(), a.size());
    } else {
        T sum{0};
        for (size_t i = 0; i < a.size(); ++i) {
            sum += a[i] * b[i];
        }
        return sum;
    }
}

// ============================================================================
// Type Aliases
// ============================================================================

using AlignedFloatTensor = AlignedTensor<float>;
using AlignedDoubleTensor = AlignedTensor<double>;

} // namespace simd
} // namespace micrograd
