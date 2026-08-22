// 06_int32_add.cpp — Integer vector addition (32-bit lanes).
//
// Quant tie-in: volume sums, order counts, bucket IDs, index arithmetic.
//
// Concepts:
//   - integer SIMD uses different intrinsics (__m256i / int32x4_t)
//   - exact arithmetic (no rounding) — easier correctness checks
//   - overflow: wrapping is well-defined for unsigned; signed overflow is UB in C++

#include "bench_common.hpp"

#include <cstdint>
#include <vector>

static SIMD_SCALAR_NO_VEC void i32_add_scalar(const std::int32_t* a, const std::int32_t* b, std::int32_t* out, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = a[i] + b[i];
    }
}

#if defined(SIMD_BACKEND_AVX2)
static void i32_add_simd(const std::int32_t* a, const std::int32_t* b, std::int32_t* out, std::size_t n)
{
    constexpr std::size_t step = 8;
    std::size_t i = 0;
    for (; i + step <= n; i += step) {
        const __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
        const __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i));
        const __m256i vr = _mm256_add_epi32(va, vb);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + i), vr);
    }
    for (; i < n; ++i) {
        out[i] = a[i] + b[i];
    }
}
#elif defined(SIMD_BACKEND_NEON)
static void i32_add_simd(const std::int32_t* a, const std::int32_t* b, std::int32_t* out, std::size_t n)
{
    constexpr std::size_t step = 4;
    std::size_t i = 0;
    for (; i + step <= n; i += step) {
        const int32x4_t va = vld1q_s32(a + i);
        const int32x4_t vb = vld1q_s32(b + i);
        const int32x4_t vr = vaddq_s32(va, vb);
        vst1q_s32(out + i, vr);
    }
    for (; i < n; ++i) {
        out[i] = a[i] + b[i];
    }
}
#else
static void i32_add_simd(const std::int32_t* a, const std::int32_t* b, std::int32_t* out, std::size_t n)
{
    i32_add_scalar(a, b, out, n);
}
#endif

int main()
{
    bench::print_header("06 — int32 Vector Add");

    const std::size_t n = bench::kDefaultSize;
    bench::print_size_info(n, "int32 elements", sizeof(std::int32_t));

    std::vector<std::int32_t> a(n), b(n), expected(n), result(n);

    for (std::size_t i = 0; i < n; ++i) {
        a[i] = static_cast<std::int32_t>((i % 1000) - 500);
        b[i] = static_cast<std::int32_t>((i % 777) - 300);
    }

    i32_add_scalar(a.data(), b.data(), expected.data(), n);

    const auto scalar_time = bench::time_microseconds([&] {
        i32_add_scalar(a.data(), b.data(), result.data(), n);
    });

    const auto simd_time = bench::time_microseconds([&] {
        i32_add_simd(a.data(), b.data(), result.data(), n);
    });

    const bool ok = bench::verify_int32_exact(expected, result);

    bench::print_timed("Scalar", scalar_time);
    bench::print_timed("SIMD", simd_time);
    bench::print_speedup("Scalar", scalar_time.median_us, "SIMD", simd_time.median_us);
    bench::print_pass_fail(ok);

    return ok ? 0 : 1;
}
