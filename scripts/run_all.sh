#!/usr/bin/env bash
# Run all SIMD exercises after building.
set -euo pipefail
cd "$(dirname "$0")/.."
BUILD="${1:-build}"

if [[ ! -d "$BUILD" ]]; then
  echo "Build dir '$BUILD' missing. Run: cmake -S . -B $BUILD && cmake --build $BUILD"
  exit 1
fi

for ex in 01_vecadd 02_dot_product 03_fma_scale 04_min_max \
          05_blend_branchless 06_int32_add 07_reduce_sum; do
  echo "──────── $ex ────────"
  "$BUILD/$ex"
done
