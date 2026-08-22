# SIMD-Lab-Cpp

Hands-on **C++17 SIMD lab**: scalar baselines vs hand-written AVX2/NEON intrinsics, with correctness checks and micro-benchmarks. Progressive exercises from vector add through reductions, FMA, and branch-free kernels.

Background for **batch validation** and **numeric hot paths** in quant work. **No third-party libraries** beyond the C++ standard library (CMake to build).

| Doc | Purpose |
|-----|---------|
| [`PLAN.md`](PLAN.md) | Scope lock, roadmap, reading order |
| [`theory/index.html`](theory/index.html) | Knowledge notes (SIMD concepts used in this tree) |
| [`LICENSE`](LICENSE) | MIT |

---

## What it does

Each exercise implements the same kernel two ways — a true scalar baseline (compiler auto-vectorization disabled) and hand-written AVX2 or NEON intrinsics — then verifies results and reports median runtime and speedup.

```text
  std::vector / arrays
         │
         ├─► scalar (SIMD_SCALAR_NO_VEC) ──► expected
         └─► AVX2 / NEON intrinsics ──────► result
                                              │
                                    verify + time (bench_common.hpp)
```

---

## Scope (v1)

| In | Out |
|----|-----|
| AVX2 (x86) and NEON (Apple Silicon / aarch64) | AVX-512, SVE |
| Seven progressive kernels + shared harness | Full BLAS / Eigen wrappers |
| Correctness checks + median micro-benchmarks | Production profiling (`perf`, Instruments) |
| Unaligned `loadu`/`storeu` paths | Gather/scatter, AoS↔SoA exercises (later) |
| CMake + CTest | Multithreading |

---

## Quick start

```bash
cmake -S . -B build
cmake --build build
./build/01_vecadd
```

Run all exercises (also via CTest):

```bash
ctest --test-dir build --output-on-failure
# or
./scripts/run_all.sh
```

---

## Exercises

| # | Target | File | Concept |
|---|--------|------|---------|
| 01 | `01_vecadd` | `src/01_vecadd.cpp` | load / add / store, remainder loop |
| 02 | `02_dot_product` | `src/02_dot_product.cpp` | FMA accumulate + horizontal reduce |
| 03 | `03_fma_scale` | `src/03_fma_scale.cpp` | AXPY: `y += α·x` |
| 04 | `04_min_max` | `src/04_min_max.cpp` | lane-wise min/max, clip |
| 05 | `05_blend_branchless` | `src/05_blend_branchless.cpp` | compare masks, blend, abs |
| 06 | `06_int32_add` | `src/06_int32_add.cpp` | integer SIMD |
| 07 | `07_reduce_sum` | `src/07_reduce_sum.cpp` | array sum reduction |

Each binary prints median timing (scalar vs SIMD), speedup, and PASS/FAIL correctness.

### Sample results (this machine)

Captured with the default harness (`n = 10'000'007`, 3 warmup + 10 timed, median). Backend: **NEON** (4 float lanes), Apple Silicon (`arm64`), Clang, `-O3 -mcpu=native`. Scalar uses `SIMD_SCALAR_NO_VEC`. Numbers vary by load and thermal state — re-run locally.

| Exercise | Kernel | Scalar (median µs) | SIMD (median µs) | Speedup |
|----------|--------|--------------------:|------------------:|--------:|
| 01 | vecadd | 14798 | 1609 | 9.2× |
| 02 | dot product | 13559 | 3375 | 4.0× |
| 03 | AXPY (FMA) | 14906 | 2774 | 5.4× |
| 04 | clip | 21674 | 1367 | 15.9× |
| 04 | array min | 17381 | 1892 | 9.2× |
| 05 | threshold flag | 26375 | 1132 | 23.3× |
| 05 | abs | 16224 | 1202 | 13.5× |
| 06 | int32 add | 14552 | 1982 | 7.3× |
| 07 | reduce sum | 10352 | 2398 | 4.3× |

High speedups on clip/flags partly reflect expensive scalar baselines (`optnone` / no autovec) vs tight NEON loops — useful for ranking kernels, not as absolute “production” claims.

---

## Layout

```
include/
  bench_common.hpp    # timing, verify, reporting
  simd_detect.hpp     # AVX2 / NEON / scalar dispatch
  aligned_alloc.hpp   # optional 32-byte aligned buffers
src/                  # one exercise per file
theory/               # knowledge notes — open theory/index.html
scripts/run_all.sh
CMakeLists.txt
```

---

## Theory notes

Open [`theory/index.html`](theory/index.html) in a browser: foundations, loop patterns, reductions/FMA, branchless/integer, benchmarking, quant applications, plus an [intrinsics catalog](theory/Theory_Intrinsics_Catalog.html) and AVX2↔NEON quick map.

Suggested order: [Foundations](theory/Theory_Foundations.html) → [Loop patterns](theory/Theory_Intrinsics.html) → exercises `01`…`07` → [Intrinsics catalog](theory/Theory_Intrinsics_Catalog.html) as a lookup while reading code.

---

## Build flags

CMake sets `-O3` and native ISA flags:

- **x86_64:** `-march=native -mfma` (AVX2 + FMA when the CPU supports them)
- **Apple Silicon / aarch64:** `-mcpu=native` (NEON)

Scalar baselines use `SIMD_SCALAR_NO_VEC` so the compiler does not auto-vectorize them — speedup compares intrinsics vs true scalar.

---

## Requirements

- C++17 compiler (Clang or GCC)
- CMake 3.16+

---

## License

See [LICENSE](LICENSE).
