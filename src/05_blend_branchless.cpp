// 05_blend_branchless.cpp — Branchless threshold mask and absolute value.
//
// Quant tie-in: stop-loss flags, signal masks, replace if-chains in validation loops.
//
// Concepts:
//   - compare produces a bitmask per lane (all-ones = true, all-zeros = false)
//   - blend selects lane values from two inputs based on mask
//   - abs via clear sign bit: and(not(sign_mask), value)

#include "bench_common.hpp"

#include <cmath>
#include <vector>

static SIMD_SCALAR_NO_VEC void threshold_flag_scalar(const float* in, float* out, float threshold, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i) {
        // 1.0 if above threshold, else 0.0 — branch in scalar version
        out[i] = (in[i] > threshold) ? 1.0f : 0.0f;
    }
}

static SIMD_SCALAR_NO_VEC void abs_scalar(const float* in, float* out, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = std::abs(in[i]);
    }
}

#if defined(SIMD_BACKEND_AVX2)
static void threshold_flag_simd(const float* in, float* out, float threshold, std::size_t n)
{
    constexpr std::size_t step = 8;
    const __m256 vthresh = _mm256_set1_ps(threshold);
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 zero = _mm256_setzero_ps();
    std::size_t i = 0;
    for (; i + step <= n; i += step) {
        const __m256 v = _mm256_loadu_ps(in + i);
        const __m256 mask = _mm256_cmp_ps(v, vthresh, _CMP_GT_OQ);
        const __m256 result = _mm256_blendv_ps(zero, one, mask);
        _mm256_storeu_ps(out + i, result);
    }
    for (; i < n; ++i) {
        out[i] = (in[i] > threshold) ? 1.0f : 0.0f;
    }
}

static void abs_simd(const float* in, float* out, std::size_t n)
{
    constexpr std::size_t step = 8;
    const __m256 sign_mask = _mm256_set1_ps(-0.0f); // bit pattern 0x80000000 per lane
    std::size_t i = 0;
    for (; i + step <= n; i += step) {
        __m256 v = _mm256_loadu_ps(in + i);
        v = _mm256_andnot_ps(sign_mask, v);
        _mm256_storeu_ps(out + i, v);
    }
    for (; i < n; ++i) {
        out[i] = std::abs(in[i]);
    }
}
#elif defined(SIMD_BACKEND_NEON)
static void threshold_flag_simd(const float* in, float* out, float threshold, std::size_t n)
{
    constexpr std::size_t step = 4;
    const float32x4_t vthresh = vdupq_n_f32(threshold);
    const float32x4_t one = vdupq_n_f32(1.0f);
    const float32x4_t zero = vdupq_n_f32(0.0f);
    std::size_t i = 0;
    for (; i + step <= n; i += step) {
        const float32x4_t v = vld1q_f32(in + i);
        const uint32x4_t mask = vcgtq_f32(v, vthresh);
        const float32x4_t result = vbslq_f32(mask, one, zero);
        vst1q_f32(out + i, result);
    }
    for (; i < n; ++i) {
        out[i] = (in[i] > threshold) ? 1.0f : 0.0f;
    }
}

static void abs_simd(const float* in, float* out, std::size_t n)
{
    constexpr std::size_t step = 4;
    std::size_t i = 0;
    for (; i + step <= n; i += step) {
        float32x4_t v = vld1q_f32(in + i);
        v = vabsq_f32(v);
        vst1q_f32(out + i, v);
    }
    for (; i < n; ++i) {
        out[i] = std::abs(in[i]);
    }
}
#else
static void threshold_flag_simd(const float* in, float* out, float threshold, std::size_t n)
{
    threshold_flag_scalar(in, out, threshold, n);
}
static void abs_simd(const float* in, float* out, std::size_t n)
{
    abs_scalar(in, out, n);
}
#endif

int main()
{
    bench::print_header("05 — Blend / Branchless");

    const std::size_t n = bench::kDefaultSize;
    bench::print_size_info(n);

    constexpr float threshold = 0.0f;
    std::vector<float> in(n), flag_expected(n), flag_result(n);
    std::vector<float> abs_expected(n), abs_result(n);

    for (std::size_t i = 0; i < n; ++i) {
        in[i] = static_cast<float>((i % 200) - 100) * 0.01f;
    }

    threshold_flag_scalar(in.data(), flag_expected.data(), threshold, n);
    abs_scalar(in.data(), abs_expected.data(), n);

    const auto flag_scalar_time = bench::time_microseconds([&] {
        threshold_flag_scalar(in.data(), flag_result.data(), threshold, n);
    });
    const auto flag_simd_time = bench::time_microseconds([&] {
        threshold_flag_simd(in.data(), flag_result.data(), threshold, n);
    });

    const auto abs_scalar_time = bench::time_microseconds([&] {
        abs_scalar(in.data(), abs_result.data(), n);
    });
    const auto abs_simd_time = bench::time_microseconds([&] {
        abs_simd(in.data(), abs_result.data(), n);
    });

    const bool flag_ok = bench::verify_float_exact(flag_expected, flag_result);
    const bool abs_ok = bench::verify_float_exact(abs_expected, abs_result);

    std::cout << "Threshold flag (x > 0):\n";
    bench::print_timed("  Scalar", flag_scalar_time);
    bench::print_timed("  SIMD", flag_simd_time);
    bench::print_speedup("Scalar", flag_scalar_time.median_us, "SIMD", flag_simd_time.median_us);

    std::cout << "\nAbsolute value:\n";
    bench::print_timed("  Scalar", abs_scalar_time);
    bench::print_timed("  SIMD", abs_simd_time);
    bench::print_speedup("Scalar", abs_scalar_time.median_us, "SIMD", abs_simd_time.median_us);

    const bool ok = flag_ok && abs_ok;
    bench::print_pass_fail(ok);

    return ok ? 0 : 1;
}
