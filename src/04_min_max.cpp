// 04_min_max.cpp — Element-wise min, max, and clip (clamp to [lo, hi]).
//
// Quant tie-in: clip returns, enforce limits, find bounds in a window.
//
// Concepts:
//   - _mm256_min_ps / max_ps operate lane-wise without branches
//   - clip(x) = min(hi, max(lo, x)) — three SIMD ops, zero branches in hot loop
//   - horizontal min/max for array-wide extrema (reduction)

#include "bench_common.hpp"

#include <algorithm>
#include <limits>
#include <vector>

static SIMD_SCALAR_NO_VEC void clip_scalar(const float* in, float* out, float lo, float hi, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = std::min(hi, std::max(lo, in[i]));
    }
}

static SIMD_SCALAR_NO_VEC float min_scalar(const float* in, std::size_t n)
{
    float m = std::numeric_limits<float>::max();
    for (std::size_t i = 0; i < n; ++i) {
        m = std::min(m, in[i]);
    }
    return m;
}

#if defined(SIMD_BACKEND_AVX2)
static void clip_simd(const float* in, float* out, float lo, float hi, std::size_t n)
{
    constexpr std::size_t step = 8;
    const __m256 vlo = _mm256_set1_ps(lo);
    const __m256 vhi = _mm256_set1_ps(hi);
    std::size_t i = 0;
    for (; i + step <= n; i += step) {
        __m256 v = _mm256_loadu_ps(in + i);
        v = _mm256_max_ps(v, vlo);
        v = _mm256_min_ps(v, vhi);
        _mm256_storeu_ps(out + i, v);
    }
    for (; i < n; ++i) {
        out[i] = std::min(hi, std::max(lo, in[i]));
    }
}

static float hmin_avx(__m256 v)
{
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 m = _mm_min_ps(lo, hi);
    m = _mm_min_ps(m, _mm_shuffle_ps(m, m, _MM_SHUFFLE(2, 3, 0, 1)));
    m = _mm_min_ps(m, _mm_shuffle_ps(m, m, _MM_SHUFFLE(1, 0, 3, 2)));
    return _mm_cvtss_f32(m);
}

static float min_simd(const float* in, std::size_t n)
{
    constexpr std::size_t step = 8;
    __m256 acc = _mm256_set1_ps(std::numeric_limits<float>::max());
    std::size_t i = 0;
    for (; i + step <= n; i += step) {
        const __m256 v = _mm256_loadu_ps(in + i);
        acc = _mm256_min_ps(acc, v);
    }
    float m = hmin_avx(acc);
    for (; i < n; ++i) {
        m = std::min(m, in[i]);
    }
    return m;
}
#elif defined(SIMD_BACKEND_NEON)
static void clip_simd(const float* in, float* out, float lo, float hi, std::size_t n)
{
    constexpr std::size_t step = 4;
    const float32x4_t vlo = vdupq_n_f32(lo);
    const float32x4_t vhi = vdupq_n_f32(hi);
    std::size_t i = 0;
    for (; i + step <= n; i += step) {
        float32x4_t v = vld1q_f32(in + i);
        v = vmaxq_f32(v, vlo);
        v = vminq_f32(v, vhi);
        vst1q_f32(out + i, v);
    }
    for (; i < n; ++i) {
        out[i] = std::min(hi, std::max(lo, in[i]));
    }
}

static float hmin_neon(float32x4_t v)
{
    float32x2_t m = vmin_f32(vget_low_f32(v), vget_high_f32(v));
    m = vpmin_f32(m, m);
    return vget_lane_f32(m, 0);
}

static float min_simd(const float* in, std::size_t n)
{
    constexpr std::size_t step = 4;
    float32x4_t acc = vdupq_n_f32(std::numeric_limits<float>::max());
    std::size_t i = 0;
    for (; i + step <= n; i += step) {
        const float32x4_t v = vld1q_f32(in + i);
        acc = vminq_f32(acc, v);
    }
    float m = hmin_neon(acc);
    for (; i < n; ++i) {
        m = std::min(m, in[i]);
    }
    return m;
}
#else
static void clip_simd(const float* in, float* out, float lo, float hi, std::size_t n)
{
    clip_scalar(in, out, lo, hi, n);
}
static float min_simd(const float* in, std::size_t n)
{
    return min_scalar(in, n);
}
#endif

int main()
{
    bench::print_header("04 — Min / Max / Clip");

    const std::size_t n = bench::kDefaultSize;
    bench::print_size_info(n);

    constexpr float lo = -0.5f;
    constexpr float hi = 0.5f;

    std::vector<float> in(n), expected(n), result(n);
    for (std::size_t i = 0; i < n; ++i) {
        // Sawtooth pattern crossing clip bounds
        in[i] = static_cast<float>((i % 200) - 100) * 0.01f;
    }

    clip_scalar(in.data(), expected.data(), lo, hi, n);
    const float expected_min = min_scalar(in.data(), n);

    const auto clip_scalar_time = bench::time_microseconds([&] {
        clip_scalar(in.data(), result.data(), lo, hi, n);
    });
    const auto clip_simd_time = bench::time_microseconds([&] {
        clip_simd(in.data(), result.data(), lo, hi, n);
    });

    const auto min_scalar_time = bench::time_microseconds([&] {
        volatile float sink = min_scalar(in.data(), n);
        (void)sink;
    });
    const auto min_simd_time = bench::time_microseconds([&] {
        volatile float sink = min_simd(in.data(), n);
        (void)sink;
    });

    const bool clip_ok = bench::verify_float_exact(expected, result);
    const float simd_min = min_simd(in.data(), n);
    const bool min_ok = simd_min == expected_min;

    std::cout << "Clip:\n";
    bench::print_timed("  Scalar", clip_scalar_time);
    bench::print_timed("  SIMD", clip_simd_time);
    bench::print_speedup("Scalar", clip_scalar_time.median_us, "SIMD", clip_simd_time.median_us);

    std::cout << "\nArray min:\n";
    bench::print_timed("  Scalar", min_scalar_time);
    bench::print_timed("  SIMD", min_simd_time);
    std::cout << "Min value: " << simd_min << " (expected " << expected_min << ")\n";

    const bool ok = clip_ok && min_ok;
    bench::print_pass_fail(ok);

    return ok ? 0 : 1;
}
