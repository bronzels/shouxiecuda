#ifndef SIMD_HPP
#define SIMD_HPP

#include <stdlib.h>

template <typename T>
void sum_array_cpu_simd_avx2(T *h_a, T *h_b, T *h_c, size_t size);

template <typename T>
void sum_array_cpu_simd_avx512(T *h_a, T *h_b, T *h_c, size_t size);

#endif //SIMD_HPP