// 01_vecadd.cpp — Element-wise vector addition: scalar vs SIMD.
//
// Quant tie-in: batch PnL add, element-wise spread adjustment, combining arrays.
//
// Concepts:
//   - __m256 / float32x4_t registers hold multiple lanes
//   - load → compute → store loop with step = lane count
//   - remainder scalar loop for tail elements

#include "bench_common.hpp"

#include <vector>

// Scalar baseline — optnone / no-tree-vectorize so the compiler does not SIMD this.
static SIMD_SCALAR_NO_VEC void vecadd_scalar(const float* a, const float* b, float* out, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = a[i] + b[i];
    }
}

#if defined(SIMD_BACKEND_AVX2)
static void vecadd_simd(const float* a, const float* b, float* out, std::size_t n)
{
    constexpr std::size_t step = 8;
    std::size_t i = 0;
    for (; i + step <= n; i += step) {
        const __m256 va = _mm256_loadu_ps(a + i);
        const __m256 vb = _mm256_loadu_ps(b + i);
        const __m256 vr = _mm256_add_ps(va, vb);
        _mm256_storeu_ps(out + i, vr);
    }
    for (; i < n; ++i) {
        out[i] = a[i] + b[i];
    }
}
#elif defined(SIMD_BACKEND_NEON)
static void vecadd_simd(const float* a, const float* b, float* out, std::size_t n)
{
    constexpr std::size_t step = 4;
    std::size_t i = 0;
    for (; i + step <= n; i += step) {
        const float32x4_t va = vld1q_f32(a + i);
        const float32x4_t vb = vld1q_f32(b + i);
        const float32x4_t vr = vaddq_f32(va, vb);
        vst1q_f32(out + i, vr);
    }
    for (; i < n; ++i) {
        out[i] = a[i] + b[i];
    }
}
#else
static void vecadd_simd(const float* a, const float* b, float* out, std::size_t n)
{
    vecadd_scalar(a, b, out, n);
}
#endif

// PnL Add
int main()
{
    bench::print_header("01 — Vector Add");

    const std::size_t n = bench::kDefaultSize;
    bench::print_size_info(n);

    std::vector<float> a(n), b(n), expected(n), result(n);

    for (std::size_t i = 0; i < n; ++i) {
        a[i] = 1.234f + static_cast<float>(i % 17) * 0.001f;
        b[i] = 2.567f + static_cast<float>(i % 13) * 0.002f;
    }

    vecadd_scalar(a.data(), b.data(), expected.data(), n);

    const auto scalar_time = bench::time_microseconds([&] {
        vecadd_scalar(a.data(), b.data(), result.data(), n);
    });

    const auto simd_time = bench::time_microseconds([&] {
        vecadd_simd(a.data(), b.data(), result.data(), n);
    });

    const bool ok = bench::verify_float_ranges(expected, result);

    bench::print_timed("Scalar", scalar_time);
    bench::print_timed("SIMD", simd_time);
    bench::print_speedup("Scalar", scalar_time.median_us, "SIMD", simd_time.median_us);
    bench::print_pass_fail(ok);

    return ok ? 0 : 1;
}
