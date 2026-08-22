#pragma once
// bench_common.hpp — shared timing, verification, and reporting for all exercises.
//
// Every exercise compares:
//   1. Scalar baseline (SIMD_SCALAR_NO_VEC — no compiler autovec)
//   2. Hand-written intrinsics (when SIMD backend is available)

#include "simd_detect.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <vector>

namespace bench {

// Not a multiple of 4 or 8 — exercises the scalar remainder (tail) loop.
inline constexpr std::size_t kDefaultSize = 10'000'007;
inline constexpr int kWarmupRuns = 3;
inline constexpr int kTimedRuns = 10;

struct TimedResult {
    double median_us = 0.0;
    double min_us = 0.0;
    double max_us = 0.0;
};

// Run fn several times after warmup; return median/min/max microseconds.
// Uses steady_clock: monotonic, intended for elapsed intervals. (high_resolution_clock
// is not required to be steady; on this Apple libc++ it aliases steady_clock.)
template<typename Fn> // Fn&& fn is a callable object, forward reference to avoid copying
TimedResult time_microseconds(Fn&& fn, int warmup = kWarmupRuns, int runs = kTimedRuns)
{
    if (runs <= 0) {
        return {};
    }

    for (int w = 0; w < warmup; ++w) {
        fn();
    }

    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(runs));

    for (int r = 0; r < runs; ++r) {
        const auto t0 = std::chrono::steady_clock::now();
        fn();
        const auto t1 = std::chrono::steady_clock::now();
        const double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
        samples.push_back(us);
    }

    std::sort(samples.begin(), samples.end());
    TimedResult out{};
    out.min_us = samples.front();
    out.max_us = samples.back();
    // Even count: pick upper-middle sample (index n/2). Fine for relative speedups.
    out.median_us = samples[samples.size() / 2];
    return out;
}

inline bool verify_float_ranges(const std::vector<float>& expected,const std::vector<float>& actual,float tolerance = 1e-4f)
{
    if (expected.size() != actual.size()) {
        return false;
    }
    for (std::size_t i = 0; i < expected.size(); ++i) {
        if (std::abs(expected[i] - actual[i]) > tolerance) {
            std::cerr << "  mismatch at index " << i << ": expected " << expected[i] << ", got " << actual[i] << '\n';
            return false;
        }
    }
    return true;
}

inline bool verify_float_exact(const std::vector<float>& a, const std::vector<float>& b)
{
    if (a.size() != b.size()) {
        return false;
    }
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (a[i] != b[i]) {
            std::cerr << "  mismatch at index " << i << ": " << a[i] << " vs " << b[i] << '\n';
            return false;
        }
    }
    return true;
}

// Reductions (dot, sum) reorder float additions → compare with relative tolerance.
inline bool verify_scalar_close(float expected, float actual, float rel_tol = 0.02f)
{
    const float scale = std::max(1.0f, std::abs(expected));
    return std::abs(expected - actual) <= rel_tol * scale;
}

inline bool verify_int32_exact(const std::vector<std::int32_t>& a, const std::vector<std::int32_t>& b)
{
    if (a.size() != b.size()) {
        return false;
    }
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (a[i] != b[i]) {
            std::cerr << "  mismatch at index " << i << ": " << a[i] << " vs " << b[i] << '\n';
            return false;
        }
    }
    return true;
}

inline void print_header(const char* exercise_name)
{
    std::cout << "\n=== " << exercise_name << " ===\n";
    std::cout << "Backend: " << simd::backend_name() << " (" << simd::float_lanes() << " floats/lane)\n";
#ifdef __clang__
    std::cout << "Compiler: Clang " << __clang_major__ << '.' << __clang_minor__ << " / C++" << __cplusplus << '\n';
#elif defined(__GNUC__)
    std::cout << "Compiler: GCC " << __GNUC__ << '.' << __GNUC_MINOR__ << " / C++" << __cplusplus << '\n';
#endif
}

inline void print_size_info(std::size_t count, const char* element_label = "elements", std::size_t bytes_per_element = sizeof(float))
{
    const double mib = static_cast<double>(count * bytes_per_element) / (1024.0 * 1024.0);
    std::cout << "Array size: " << count << ' ' << element_label << " (" << std::fixed << std::setprecision(2) << mib << " MiB)\n\n";
}

inline void print_timed(const char* label, const TimedResult& t)
{
    std::cout << std::setw(18) << std::left << label << "  median " << std::setw(8) << std::fixed << std::setprecision(0) << t.median_us << " us" << "  (min " << t.min_us << ", max " << t.max_us << ")\n";
}

inline void print_speedup(const char* baseline_label, double baseline_us, const char* fast_label, double fast_us)
{
    if (fast_us <= 0.0) {
        return;
    }
    const double speedup = baseline_us / fast_us;
    std::cout << "Speedup (" << baseline_label << " / " << fast_label << "): " << std::setprecision(2) << speedup << "x\n";
}

inline void print_pass_fail(bool ok)
{
    std::cout << (ok ? "Correctness: PASS\n" : "Correctness: FAIL\n");
}

} // namespace bench
