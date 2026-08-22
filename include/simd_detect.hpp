#pragma once
// simd_detect.hpp — compile-time SIMD backend selection (AVX2 / NEON / scalar).
//
// CMake builds with -march=native (or -mcpu=native on Apple Silicon) so the
// compiler defines __AVX2__ / __ARM_NEON as appropriate. Exercise code includes
// this header once instead of repeating #if chains.

#include <cstddef>

#if defined(__AVX2__)
    #include <immintrin.h>
    #define SIMD_BACKEND_AVX2 1
    #define SIMD_BACKEND_NAME "AVX2"
    #define SIMD_FLOAT_LANES 8
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
    #include <arm_neon.h>
    #define SIMD_BACKEND_NEON 1
    #define SIMD_BACKEND_NAME "NEON"
    #define SIMD_FLOAT_LANES 4
#else
    #define SIMD_BACKEND_SCALAR 1
    #define SIMD_BACKEND_NAME "scalar"
    #define SIMD_FLOAT_LANES 1
#endif

// Prevent the compiler from auto-vectorizing scalar baseline loops.
#if defined(__clang__)
    #define SIMD_SCALAR_NO_VEC __attribute__((optnone))
#elif defined(__GNUC__)
    #define SIMD_SCALAR_NO_VEC __attribute__((optimize("O2", "no-tree-vectorize")))
#else
    #define SIMD_SCALAR_NO_VEC
#endif

namespace simd {

inline constexpr const char* backend_name() noexcept
{
#ifdef SIMD_BACKEND_NAME
    return SIMD_BACKEND_NAME;
#else
    return "unknown";
#endif
}

inline constexpr std::size_t float_lanes() noexcept
{
    return SIMD_FLOAT_LANES;
}

} // namespace simd
