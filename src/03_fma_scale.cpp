// 03_fma_scale.cpp — AXPY: y[i] += alpha * x[i]  (scaled vector accumulate).
//
// Quant tie-in: EWMA updates, exponential smoothing, gradient-style accumulations.
//
// Concepts:
//   - FMA (fused multiply-add): one instruction, one rounding
//   - _mm256_fmadd_ps / vfmaq_f32
//   - in-place update of y array

#include "bench_common.hpp"

#include <vector>

static SIMD_SCALAR_NO_VEC void axpy_scalar(float alpha, const float* x, float* y, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i) {
        y[i] += alpha * x[i];
    }
}

#if defined(SIMD_BACKEND_AVX2)
static void axpy_simd(float alpha, const float* x, float* y, std::size_t n)
{
    constexpr std::size_t step = 8;
    const __m256 va = _mm256_set1_ps(alpha);
    std::size_t i = 0;
    for (; i + step <= n; i += step) {
        const __m256 vx = _mm256_loadu_ps(x + i);
        __m256 vy = _mm256_loadu_ps(y + i);
        vy = _mm256_fmadd_ps(va, vx, vy);
        _mm256_storeu_ps(y + i, vy);
    }
    for (; i < n; ++i) {
        y[i] += alpha * x[i];
    }
}
#elif defined(SIMD_BACKEND_NEON)
static void axpy_simd(float alpha, const float* x, float* y, std::size_t n)
{
    constexpr std::size_t step = 4;
    const float32x4_t va = vdupq_n_f32(alpha);
    std::size_t i = 0;
    for (; i + step <= n; i += step) {
        const float32x4_t vx = vld1q_f32(x + i);
        float32x4_t vy = vld1q_f32(y + i);
        vy = vfmaq_f32(vy, va, vx); // vy += va * vx
        vst1q_f32(y + i, vy);
    }
    for (; i < n; ++i) {
        y[i] += alpha * x[i];
    }
}
#else
static void axpy_simd(float alpha, const float* x, float* y, std::size_t n)
{
    axpy_scalar(alpha, x, y, n);
}
#endif

int main()
{
    bench::print_header("03 — FMA Scale (AXPY)");

    const std::size_t n = bench::kDefaultSize;
    bench::print_size_info(n);

    constexpr float alpha = 0.375f;
    std::vector<float> x(n), y_orig(n), y_expected(n), y_scalar(n), y_simd(n);

    for (std::size_t i = 0; i < n; ++i) {
        x[i] = static_cast<float>((i % 50) + 1) * 0.01f;
        y_orig[i] = static_cast<float>((i % 30)) * 0.1f;
    }

    y_expected = y_orig;
    axpy_scalar(alpha, x.data(), y_expected.data(), n);

    // Correctness check once (before timed runs mutate buffers).
    y_simd = y_orig;
    axpy_simd(alpha, x.data(), y_simd.data(), n);
    const bool ok = bench::verify_float_ranges(y_expected, y_simd, 1e-3f);

    const auto scalar_time = bench::time_microseconds([&] {
        y_scalar = y_orig;
        axpy_scalar(alpha, x.data(), y_scalar.data(), n);
    });

    const auto simd_time = bench::time_microseconds([&] {
        y_simd = y_orig;
        axpy_simd(alpha, x.data(), y_simd.data(), n);
    });

    bench::print_timed("Scalar", scalar_time);
    bench::print_timed("SIMD", simd_time);
    bench::print_speedup("Scalar", scalar_time.median_us, "SIMD", simd_time.median_us);
    bench::print_pass_fail(ok);

    return ok ? 0 : 1;
}
