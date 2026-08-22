// 07_reduce_sum.cpp — Array sum via SIMD accumulation + horizontal reduction.
//
// Quant tie-in: batch mean return, variance first pass, total PnL aggregation.
//
// Concepts:
//   - same pattern as dot product but with ones instead of second array
//   - naive SIMD sum can reorder additions → slightly different float rounding
//   - for production: Kahan/compensated sum when precision matters (see theory notes)

#include "bench_common.hpp"

#include <vector>

static SIMD_SCALAR_NO_VEC float sum_scalar(const float* in, std::size_t n)
{
    float sum = 0.0f;
    for (std::size_t i = 0; i < n; ++i) {
        sum += in[i];
    }
    return sum;
}

#if defined(SIMD_BACKEND_AVX2)
static float hsum_avx(__m256 v)
{
    const __m128 lo = _mm256_castps256_ps128(v);
    const __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 sum128 = _mm_add_ps(lo, hi);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    return _mm_cvtss_f32(sum128);
}

static float sum_simd(const float* in, std::size_t n)
{
    constexpr std::size_t step = 8;
    __m256 acc = _mm256_setzero_ps();
    std::size_t i = 0;
    for (; i + step <= n; i += step) {
        const __m256 v = _mm256_loadu_ps(in + i);
        acc = _mm256_add_ps(acc, v);
    }
    float sum = hsum_avx(acc);
    for (; i < n; ++i) {
        sum += in[i];
    }
    return sum;
}
#elif defined(SIMD_BACKEND_NEON)
static float hsum_neon(float32x4_t v)
{
    float32x2_t sum2 = vadd_f32(vget_low_f32(v), vget_high_f32(v));
    sum2 = vpadd_f32(sum2, sum2);
    return vget_lane_f32(sum2, 0);
}

static float sum_simd(const float* in, std::size_t n)
{
    constexpr std::size_t step = 4;
    float32x4_t acc = vdupq_n_f32(0.0f);
    std::size_t i = 0;
    for (; i + step <= n; i += step) {
        const float32x4_t v = vld1q_f32(in + i);
        acc = vaddq_f32(acc, v);
    }
    float sum = hsum_neon(acc);
    for (; i < n; ++i) {
        sum += in[i];
    }
    return sum;
}
#else
static float sum_simd(const float* in, std::size_t n)
{
    return sum_scalar(in, n);
}
#endif

int main()
{
    bench::print_header("07 — Reduce Sum");

    const std::size_t n = bench::kDefaultSize;
    bench::print_size_info(n);

    std::vector<float> data(n);
    for (std::size_t i = 0; i < n; ++i) {
        data[i] = 0.001f * static_cast<float>((i % 100) + 1);
    }

    const float expected = sum_scalar(data.data(), n);

    float scalar_result = 0.0f;
    const auto scalar_time = bench::time_microseconds([&] {
        scalar_result = sum_scalar(data.data(), n);
    });

    float simd_result = 0.0f;
    const auto simd_time = bench::time_microseconds([&] {
        simd_result = sum_simd(data.data(), n);
    });

    // Parallel lane accumulation reorders float adds vs serial scalar.
    const bool ok = bench::verify_scalar_close(expected, scalar_result, 1e-5f)
                 && bench::verify_scalar_close(expected, simd_result, 0.02f);

    bench::print_timed("Scalar", scalar_time);
    bench::print_timed("SIMD", simd_time);
    bench::print_speedup("Scalar", scalar_time.median_us, "SIMD", simd_time.median_us);
    std::cout << "Sum: " << std::fixed << std::setprecision(4)
              << simd_result << " (scalar ref " << expected << ")\n";
    bench::print_pass_fail(ok);

    return ok ? 0 : 1;
}
