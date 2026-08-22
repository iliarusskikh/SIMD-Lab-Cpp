#pragma once
// aligned_alloc.hpp — 32-byte aligned buffers for AVX load/store (optional exercise).
//
// Unaligned loadu/storeu is fine for learning; aligned memory teaches cache-line
// alignment and when _mm256_load_ps beats _mm256_loadu_ps.

#include <cstdlib>
#include <new>

namespace simd {

inline void* allocate_aligned(std::size_t bytes, std::size_t alignment = 32)
{
    // std::aligned_alloc requires size to be a multiple of alignment.
    const std::size_t rounded =
        (bytes + alignment - 1) / alignment * alignment;
#if defined(_MSC_VER)
    void* ptr = _aligned_malloc(rounded, alignment);
#else
    void* ptr = std::aligned_alloc(alignment, rounded);
#endif
    if (!ptr) {
        throw std::bad_alloc{};
    }
    return ptr;
}

inline void free_aligned(void* ptr) noexcept
{
#if defined(_MSC_VER)
    _aligned_free(ptr);
#else
    std::free(ptr);
#endif
}

// RAII wrapper — std::vector-compatible interface for float buffers.
class AlignedFloatBuffer {
public:
    explicit AlignedFloatBuffer(std::size_t count) : count_(count)
        , ptr_(static_cast<float*>(allocate_aligned(count * sizeof(float))))
    {}

    ~AlignedFloatBuffer() { free_aligned(ptr_); }

    AlignedFloatBuffer(const AlignedFloatBuffer&) = delete;
    AlignedFloatBuffer& operator=(const AlignedFloatBuffer&) = delete;

    AlignedFloatBuffer(AlignedFloatBuffer&& other) noexcept : count_(other.count_), ptr_(other.ptr_)
    {
        other.ptr_ = nullptr;
        other.count_ = 0;
    }

    float* data() noexcept { return ptr_; }
    const float* data() const noexcept { return ptr_; }
    std::size_t size() const noexcept { return count_; }

    float& operator[](std::size_t i) { return ptr_[i]; }
    const float& operator[](std::size_t i) const { return ptr_[i]; }

private:
    std::size_t count_;
    float* ptr_;
};

} // namespace simd
