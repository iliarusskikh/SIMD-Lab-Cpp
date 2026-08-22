// 02_dot_product.cpp — Dot product with horizontal reduction.
//
// Quant tie-in: correlation numerator, portfolio weights · returns, similarity scores.
//
// Concepts:
//   - multiply lanes, accumulate into a running vector sum
//   - horizontal reduction: fold 8 (or 4) partial sums into one scalar
//   - remainder loop for tail elements

#include "bench_common.hpp"

#include <vector>

static SIMD_SCALAR_NO_VEC float dot_scalar(const float* a, const float* b, std::size_t n)
{
    float sum = 0.0f;
    for (std::size_t i = 0; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

#if defined(SIMD_BACKEND_AVX2)
static float hsum_avx(__m256 v)
{
    // Fold 256-bit register to 128, then 128 to scalar.
    const __m128 lo = _mm256_castps256_ps128(v);
    const __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 sum128 = _mm_add_ps(lo, hi);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    return _mm_cvtss_f32(sum128);
}

static float dot_simd(const float* a, const float* b, std::size_t n)
{
    constexpr std::size_t step = 8;
    __m256 acc = _mm256_setzero_ps();
    std::size_t i = 0;
    for (; i + step <= n; i += step) {
        const __m256 va = _mm256_loadu_ps(a + i);
        const __m256 vb = _mm256_loadu_ps(b + i);
        acc = _mm256_fmadd_ps(va, vb, acc);
    }
    float sum = hsum_avx(acc);
    for (; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}
#elif defined(SIMD_BACKEND_NEON)
static float hsum_neon(float32x4_t v)
{
    // Pairwise add lanes: [a,b,c,d] → [a+b, c+d, a+b, c+d] → scalar
    float32x2_t sum2 = vadd_f32(vget_low_f32(v), vget_high_f32(v));
    sum2 = vpadd_f32(sum2, sum2);
    return vget_lane_f32(sum2, 0);
}

static float dot_simd(const float* a, const float* b, std::size_t n)
{
    constexpr std::size_t step = 4;
    float32x4_t acc = vdupq_n_f32(0.0f);
    std::size_t i = 0;
    for (; i + step <= n; i += step) {
        const float32x4_t va = vld1q_f32(a + i);
        const float32x4_t vb = vld1q_f32(b + i);
        acc = vfmaq_f32(acc, va, vb); // acc += va * vb
    }
    float sum = hsum_neon(acc);
    for (; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}
#else
static float dot_simd(const float* a, const float* b, std::size_t n)
{
    return dot_scalar(a, b, n);
}
#endif

int main()
{
    bench::print_header("02 — Dot Product");

    const std::size_t n = bench::kDefaultSize;
    bench::print_size_info(n);

    std::vector<float> a(n), b(n);
    for (std::size_t i = 0; i < n; ++i) {
        a[i] = 0.01f * static_cast<float>((i % 100) + 1);
        b[i] = 0.02f * static_cast<float>((i % 97) + 1);
    }

    const float expected = dot_scalar(a.data(), b.data(), n);

    float scalar_result = 0.0f;
    const auto scalar_time = bench::time_microseconds([&] {
        scalar_result = dot_scalar(a.data(), b.data(), n);
    });

    float simd_result = 0.0f;
    const auto simd_time = bench::time_microseconds([&] {
        simd_result = dot_simd(a.data(), b.data(), n);
    });

    // Parallel lane accumulation reorders float adds vs serial scalar (~1–2% at 10M).
    const bool ok = bench::verify_scalar_close(expected, scalar_result, 1e-5f)
                 && bench::verify_scalar_close(expected, simd_result, 0.02f);

    bench::print_timed("Scalar", scalar_time);
    bench::print_timed("SIMD", simd_time);
    bench::print_speedup("Scalar", scalar_time.median_us, "SIMD", simd_time.median_us);
    std::cout << "Dot product: " << std::fixed << std::setprecision(4)
              << simd_result << " (expected ≈ " << expected << ")\n";
    bench::print_pass_fail(ok);

    return ok ? 0 : 1;
}
