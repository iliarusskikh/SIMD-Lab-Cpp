# SIMD Lab — Plan

**Status:** v1 complete (exercises 01–07, harness, theory, benches).

**Scope lock:** Single-threaded C++17 lab. Scalar baselines (no auto-vectorize) vs hand-written AVX2/NEON intrinsics. Correctness checks + median micro-benchmarks. No BLAS/Eigen, no multithreading, no AVX-512/SVE.

Docs: [README](README.md) · [theory knowledge notes](theory/index.html)

---

## Goal

Build working knowledge of SIMD for numeric hot paths and batch validation:

- Load / compute / store loops with remainder tails
- FMA, reductions, branch-free masks, integer lanes
- Honest timing (`steady_clock`, median) and verify-before-speedup
- Quant-shaped kernels (dot, AXPY, clip, flags) without claiming a trading system

---

## Locked decisions (v1)

| Decision | Choice |
|----------|--------|
| Dialect | C++17 |
| ISAs | AVX2 (+ FMA) on x86; NEON on Apple Silicon / aarch64 |
| Baseline | `SIMD_SCALAR_NO_VEC` (no compiler autovec on scalar) |
| Memory | Unaligned `loadu` / `storeu` + `std::vector` |
| Size | `kDefaultSize = 10'000'007` (exercises remainder path) |
| Timing | `steady_clock`, 3 warmup + 10 timed, median |
| Build | CMake + CTest; native `-O3` ISA flags |
| Theory | HTML knowledge notes (same theme as other labs) |

---

## Roadmap

| Phase | Goal | Status |
|-------|------|--------|
| 0 | README, PLAN, `.gitignore`, CMake, shared headers | ✅ |
| 1 | Exercises 01–07 (vecadd → reduce sum) | ✅ |
| 2 | Theory notes + intrinsics catalog | ✅ |
| 3 | Recorded machine results in README | ✅ |
| 4+ | Validate kernel / AoS→SoA / autovec column / CI | Optional |

---

## What shipped (v1)

| Target | Kernel |
|--------|--------|
| `01_vecadd` | Element-wise add |
| `02_dot_product` | Dot + horizontal reduce |
| `03_fma_scale` | AXPY via FMA |
| `04_min_max` | Clip + array min |
| `05_blend_branchless` | Threshold mask + abs |
| `06_int32_add` | int32 vector add |
| `07_reduce_sum` | Array sum reduce |

Shared: `bench_common.hpp`, `simd_detect.hpp`, `aligned_alloc.hpp` (optional aligned buffers).

---

## Out of scope (deliberate)

| Cut | Why |
|-----|-----|
| AVX-512 / SVE | Different programming model; keep ISA surface small |
| Eigen / BLAS wrappers | Lab is about intrinsics, not library APIs |
| Multithreading | Orthogonal; compose later if needed |
| Gather / scatter, AoS→SoA exercise | Valuable follow-on, not required for v1 |
| Production profilers | `chrono` harness is enough for teaching |

---

## Reading order

1. [Theory — Foundations](theory/Theory_Foundations.html)
2. [Theory — Loop patterns](theory/Theory_Intrinsics.html)
3. Code: `src/01_vecadd.cpp` → `07_reduce_sum.cpp`
4. [Intrinsics catalog](theory/Theory_Intrinsics_Catalog.html) as lookup
5. [Quant applications](theory/Theory_Quant_Applications.html) · [Benchmarking](theory/Theory_Benchmarking.html)

---

## Optional later

- `08_abs_diff_validate` — batch tolerance check (theory already describes the pattern)
- Demo using `AlignedFloatBuffer` + aligned loads
- Compiler-autovec third column in benches
- GitHub Actions: configure + `ctest`
