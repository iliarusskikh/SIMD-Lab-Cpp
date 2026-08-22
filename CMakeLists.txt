cmake_minimum_required(VERSION 3.16)
# SIMD Lab — C++17 scalar baselines vs hand-written AVX2/NEON intrinsics.
#
# Targets (one per exercise):
#   01_vecadd, 02_dot_product, 03_fma_scale, 04_min_max,
#   05_blend_branchless, 06_int32_add, 07_reduce_sum
#
# Build:
#   cmake -S . -B build && cmake --build build
#   ./build/01_vecadd

project(SIMDLabCpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(simd_common INTERFACE)
target_include_directories(simd_common INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(simd_common INTERFACE -Wall -Wextra -Wpedantic)

# Native ISA flags — AVX2+FMA on x86, NEON on Apple Silicon / aarch64.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm|aarch64|ARM64")
    set(SIMD_ISA_FLAGS -mcpu=native)
else()
    set(SIMD_ISA_FLAGS -march=native -mfma)
endif()

function(add_simd_exercise target_name source_file)
    add_executable(${target_name} ${source_file})
    target_link_libraries(${target_name} PRIVATE simd_common)
    target_compile_options(${target_name} PRIVATE -O3 ${SIMD_ISA_FLAGS})
endfunction()

add_simd_exercise(01_vecadd            src/01_vecadd.cpp)
add_simd_exercise(02_dot_product       src/02_dot_product.cpp)
add_simd_exercise(03_fma_scale         src/03_fma_scale.cpp)
add_simd_exercise(04_min_max           src/04_min_max.cpp)
add_simd_exercise(05_blend_branchless  src/05_blend_branchless.cpp)
add_simd_exercise(06_int32_add         src/06_int32_add.cpp)
add_simd_exercise(07_reduce_sum        src/07_reduce_sum.cpp)

enable_testing()
add_test(NAME 01_vecadd           COMMAND 01_vecadd)
add_test(NAME 02_dot_product      COMMAND 02_dot_product)
add_test(NAME 03_fma_scale        COMMAND 03_fma_scale)
add_test(NAME 04_min_max          COMMAND 04_min_max)
add_test(NAME 05_blend_branchless COMMAND 05_blend_branchless)
add_test(NAME 06_int32_add        COMMAND 06_int32_add)
add_test(NAME 07_reduce_sum       COMMAND 07_reduce_sum)
