#pragma once

#include <span>
#include <vector>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <algorithm>
#include <string>
#include <sstream>
#include <pipepye/core/types.hpp>

namespace pipepye::sparse {

template <typename T = scalar_t>
class VectorView {
public:
    using value_type = T;
    using size_type = index_t;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;
    using iterator = T*;
    using const_iterator = const T*;

    constexpr VectorView() noexcept : data_(nullptr), size_(0) {}
    constexpr VectorView(T* data, index_t size) noexcept : data_(data), size_(size) {}
    constexpr VectorView(std::span<T> s) noexcept : data_(s.data()), size_(static_cast<index_t>(s.size())) {}

    template <typename Alloc>
    constexpr VectorView(std::vector<std::remove_const_t<T>, Alloc>& vec) noexcept
        : data_(vec.data()), size_(static_cast<index_t>(vec.size())) {}

    template <typename Alloc>
    constexpr VectorView(const std::vector<std::remove_const_t<T>, Alloc>& vec) noexcept
        requires std::is_const_v<T>
        : data_(vec.data()), size_(static_cast<index_t>(vec.size())) {}

    [[nodiscard]] constexpr index_t size() const noexcept { return size_; }
    [[nodiscard]] constexpr bool empty() const noexcept { return size_ == 0; }
    [[nodiscard]] constexpr T* data() noexcept { return data_; }
    [[nodiscard]] constexpr const T* data() const noexcept { return data_; }

    [[nodiscard]] constexpr reference operator[](index_t i) noexcept { return data_[i]; }
    [[nodiscard]] constexpr const_reference operator[](index_t i) const noexcept { return data_[i]; }
    [[nodiscard]] constexpr reference operator()(index_t i) noexcept { return data_[i]; }
    [[nodiscard]] constexpr const_reference operator()(index_t i) const noexcept { return data_[i]; }

    [[nodiscard]] reference at(index_t i) {
        if (i < 0 || i >= size_) {
            throw std::out_of_range("VectorView::at index " + std::to_string(i) + " out of range [0, " + std::to_string(size_) + ")");
        }
        return data_[i];
    }

    [[nodiscard]] const_reference at(index_t i) const {
        if (i < 0 || i >= size_) {
            throw std::out_of_range("VectorView::at index " + std::to_string(i) + " out of range [0, " + std::to_string(size_) + ")");
        }
        return data_[i];
    }

    [[nodiscard]] constexpr iterator begin() noexcept { return data_; }
    [[nodiscard]] constexpr const_iterator begin() const noexcept { return data_; }
    [[nodiscard]] constexpr const_iterator cbegin() const noexcept { return data_; }
    [[nodiscard]] constexpr iterator end() noexcept { return data_ + size_; }
    [[nodiscard]] constexpr const_iterator end() const noexcept { return data_ + size_; }
    [[nodiscard]] constexpr const_iterator cend() const noexcept { return data_ + size_; }

    [[nodiscard]] VectorView<T> subvector(index_t offset, index_t count) const {
        if (offset < 0 || count < 0 || offset + count > size_) {
            throw std::out_of_range("Invalid subvector slice");
        }
        return VectorView<T>(data_ + offset, count);
    }

    // Mathematical Operations (CPU implementations)
    [[nodiscard]] scalar_t dot(VectorView<const T> other) const {
        if (size_ != other.size()) {
            throw std::invalid_argument("VectorView::dot dimension mismatch: " + std::to_string(size_) + " vs " + std::to_string(other.size()));
        }
        scalar_t sum = 0.0;
        for (index_t i = 0; i < size_; ++i) {
            sum += static_cast<scalar_t>(data_[i]) * static_cast<scalar_t>(other[i]);
        }
        return sum;
    }

    [[nodiscard]] scalar_t norm_1() const noexcept {
        scalar_t sum = 0.0;
        for (index_t i = 0; i < size_; ++i) {
            sum += std::abs(static_cast<scalar_t>(data_[i]));
        }
        return sum;
    }

    [[nodiscard]] scalar_t norm_2() const noexcept {
        scalar_t sum_sq = 0.0;
        for (index_t i = 0; i < size_; ++i) {
            scalar_t val = static_cast<scalar_t>(data_[i]);
            sum_sq += val * val;
        }
        return std::sqrt(sum_sq);
    }

    [[nodiscard]] scalar_t norm_inf() const noexcept {
        scalar_t max_val = 0.0;
        for (index_t i = 0; i < size_; ++i) {
            scalar_t val = std::abs(static_cast<scalar_t>(data_[i]));
            if (val > max_val) {
                max_val = val;
            }
        }
        return max_val;
    }

    [[nodiscard]] scalar_t norm_2_sq() const noexcept {
        scalar_t sum_sq = 0.0;
        for (index_t i = 0; i < size_; ++i) {
            scalar_t val = static_cast<scalar_t>(data_[i]);
            sum_sq += val * val;
        }
        return sum_sq;
    }

    [[nodiscard]] scalar_t sum() const noexcept {
        scalar_t s = 0.0;
        for (index_t i = 0; i < size_; ++i) {
            s += static_cast<scalar_t>(data_[i]);
        }
        return s;
    }

    [[nodiscard]] scalar_t mean() const {
        if (empty()) throw std::runtime_error("VectorView::mean called on empty vector");
        return sum() / static_cast<scalar_t>(size_);
    }

    [[nodiscard]] std::remove_const_t<T> min() const {
        if (empty()) throw std::runtime_error("VectorView::min called on empty vector");
        std::remove_const_t<T> m = data_[0];
        for (index_t i = 1; i < size_; ++i) {
            if (data_[i] < m) m = data_[i];
        }
        return m;
    }

    [[nodiscard]] std::remove_const_t<T> max() const {
        if (empty()) throw std::runtime_error("VectorView::max called on empty vector");
        std::remove_const_t<T> m = data_[0];
        for (index_t i = 1; i < size_; ++i) {
            if (data_[i] > m) m = data_[i];
        }
        return m;
    }

    [[nodiscard]] index_t argmin() const {
        if (empty()) throw std::runtime_error("VectorView::argmin called on empty vector");
        index_t idx = 0;
        for (index_t i = 1; i < size_; ++i) {
            if (data_[i] < data_[idx]) idx = i;
        }
        return idx;
    }

    [[nodiscard]] index_t argmax() const {
        if (empty()) throw std::runtime_error("VectorView::argmax called on empty vector");
        index_t idx = 0;
        for (index_t i = 1; i < size_; ++i) {
            if (data_[i] > data_[idx]) idx = i;
        }
        return idx;
    }

    [[nodiscard]] scalar_t abs_diff_inf(VectorView<const T> other) const {
        if (size_ != other.size()) {
            throw std::invalid_argument("VectorView::abs_diff_inf dimension mismatch");
        }
        scalar_t max_d = 0.0;
        for (index_t i = 0; i < size_; ++i) {
            scalar_t d = std::abs(static_cast<scalar_t>(data_[i]) - static_cast<scalar_t>(other[i]));
            if (d > max_d) max_d = d;
        }
        return max_d;
    }

    [[nodiscard]] scalar_t abs_diff_2(VectorView<const T> other) const {
        if (size_ != other.size()) {
            throw std::invalid_argument("VectorView::abs_diff_2 dimension mismatch");
        }
        scalar_t sum_sq = 0.0;
        for (index_t i = 0; i < size_; ++i) {
            scalar_t d = static_cast<scalar_t>(data_[i]) - static_cast<scalar_t>(other[i]);
            sum_sq += d * d;
        }
        return std::sqrt(sum_sq);
    }

    // In-place vector modifications (only available for non-const T)
    void fill(T val) requires (!std::is_const_v<T>) {
        std::fill(begin(), end(), val);
    }

    void set_zero() requires (!std::is_const_v<T>) {
        fill(static_cast<T>(0));
    }

    void copy_from(VectorView<const T> src) requires (!std::is_const_v<T>) {
        if (size_ != src.size()) {
            throw std::invalid_argument("VectorView::copy_from dimension mismatch: " +
                                        std::to_string(size_) + " vs " + std::to_string(src.size()));
        }
        std::copy(src.begin(), src.end(), begin());
    }

    void scale(T alpha) requires (!std::is_const_v<T>) {
        for (index_t i = 0; i < size_; ++i) {
            data_[i] *= alpha;
        }
    }

    /// @brief Computes y = alpha * x + y
    void axpy(T alpha, VectorView<const T> x) requires (!std::is_const_v<T>) {
        if (size_ != x.size()) {
            throw std::invalid_argument("VectorView::axpy dimension mismatch");
        }
        for (index_t i = 0; i < size_; ++i) {
            data_[i] += alpha * x[i];
        }
    }

    /// @brief Computes y = alpha * x + beta * y
    void axpby(T alpha, VectorView<const T> x, T beta) requires (!std::is_const_v<T>) {
        if (size_ != x.size()) {
            throw std::invalid_argument("VectorView::axpby dimension mismatch");
        }
        for (index_t i = 0; i < size_; ++i) {
            data_[i] = alpha * x[i] + beta * data_[i];
        }
    }

    /// @brief Element-wise projection onto bounds [lower, upper]
    void project_bounds(VectorView<const T> lower, VectorView<const T> upper) requires (!std::is_const_v<T>) {
        if (size_ != lower.size() || size_ != upper.size()) {
            throw std::invalid_argument("VectorView::project_bounds dimension mismatch");
        }
        for (index_t i = 0; i < size_; ++i) {
            if (data_[i] < lower[i]) data_[i] = lower[i];
            else if (data_[i] > upper[i]) data_[i] = upper[i];
        }
    }

    operator VectorView<const T>() const noexcept {
        return VectorView<const T>(data_, size_);
    }

private:
    T* data_{nullptr};
    index_t size_{0};
};

using ConstVectorView = VectorView<const scalar_t>;
using MutableVectorView = VectorView<scalar_t>;

/// @brief Owning contiguous dense vector container for optimization linear algebra
template <typename T = scalar_t>
class DenseVector {
public:
    using value_type = T;
    using size_type = index_t;

    DenseVector() = default;
    explicit DenseVector(index_t size, T init_val = static_cast<T>(0))
        : data_(size, init_val) {}

    DenseVector(std::initializer_list<T> list)
        : data_(list) {}

    DenseVector(const std::vector<T>& vec)
        : data_(vec) {}

    DenseVector(std::vector<T>&& vec) noexcept
        : data_(std::move(vec)) {}

    [[nodiscard]] index_t size() const noexcept { return static_cast<index_t>(data_.size()); }
    [[nodiscard]] bool empty() const noexcept { return data_.empty(); }
    [[nodiscard]] T* data() noexcept { return data_.data(); }
    [[nodiscard]] const T* data() const noexcept { return data_.data(); }

    void resize(index_t size, T init_val = static_cast<T>(0)) {
        data_.resize(size, init_val);
    }

    void reserve(index_t capacity) {
        data_.reserve(capacity);
    }

    void clear() noexcept {
        data_.clear();
    }

    void push_back(T val) {
        data_.push_back(val);
    }

    [[nodiscard]] T& operator[](index_t i) noexcept { return data_[i]; }
    [[nodiscard]] const T& operator[](index_t i) const noexcept { return data_[i]; }
    [[nodiscard]] T& operator()(index_t i) noexcept { return data_[i]; }
    [[nodiscard]] const T& operator()(index_t i) const noexcept { return data_[i]; }

    [[nodiscard]] T& at(index_t i) { return data_.at(i); }
    [[nodiscard]] const T& at(index_t i) const { return data_.at(i); }

    [[nodiscard]] auto begin() noexcept { return data_.begin(); }
    [[nodiscard]] auto begin() const noexcept { return data_.begin(); }
    [[nodiscard]] auto cbegin() const noexcept { return data_.cbegin(); }
    [[nodiscard]] auto end() noexcept { return data_.end(); }
    [[nodiscard]] auto end() const noexcept { return data_.end(); }
    [[nodiscard]] auto cend() const noexcept { return data_.cend(); }

    [[nodiscard]] const std::vector<T>& std_vector() const noexcept { return data_; }
    [[nodiscard]] std::vector<T>& std_vector() noexcept { return data_; }

    [[nodiscard]] VectorView<T> view() noexcept {
        return VectorView<T>(data_.data(), size());
    }

    [[nodiscard]] VectorView<const T> view() const noexcept {
        return VectorView<const T>(data_.data(), size());
    }

    operator VectorView<T>() noexcept {
        return view();
    }

    operator VectorView<const T>() const noexcept {
        return view();
    }

    // Mathematical delegates to VectorView
    [[nodiscard]] scalar_t dot(VectorView<const T> other) const { return view().dot(other); }
    [[nodiscard]] scalar_t norm_1() const noexcept { return view().norm_1(); }
    [[nodiscard]] scalar_t norm_2() const noexcept { return view().norm_2(); }
    [[nodiscard]] scalar_t norm_2_sq() const noexcept { return view().norm_2_sq(); }
    [[nodiscard]] scalar_t norm_inf() const noexcept { return view().norm_inf(); }
    [[nodiscard]] scalar_t sum() const noexcept { return view().sum(); }
    [[nodiscard]] scalar_t mean() const { return view().mean(); }
    [[nodiscard]] T min() const { return view().min(); }
    [[nodiscard]] T max() const { return view().max(); }
    [[nodiscard]] index_t argmin() const { return view().argmin(); }
    [[nodiscard]] index_t argmax() const { return view().argmax(); }
    [[nodiscard]] scalar_t abs_diff_inf(VectorView<const T> other) const { return view().abs_diff_inf(other); }
    [[nodiscard]] scalar_t abs_diff_2(VectorView<const T> other) const { return view().abs_diff_2(other); }
    void fill(T val) { view().fill(val); }
    void set_zero() { view().set_zero(); }
    void copy_from(VectorView<const T> src) { view().copy_from(src); }
    void scale(T alpha) { view().scale(alpha); }
    void axpy(T alpha, VectorView<const T> x) { view().axpy(alpha, x); }
    void axpby(T alpha, VectorView<const T> x, T beta) { view().axpby(alpha, x, beta); }
    void project_bounds(VectorView<const T> lower, VectorView<const T> upper) {
        view().project_bounds(lower, upper);
    }

private:
    std::vector<T> data_;
};

using Vector = DenseVector<scalar_t>;

template <typename T>
inline std::ostream& operator<<(std::ostream& os, VectorView<T> vec) {
    os << "[";
    for (index_t i = 0; i < vec.size(); ++i) {
        if (i > 0) os << ", ";
        os << vec[i];
    }
    os << "]";
    return os;
}

template <typename T>
inline std::ostream& operator<<(std::ostream& os, const DenseVector<T>& vec) {
    return os << vec.view();
}

} // namespace pipepye::sparse
