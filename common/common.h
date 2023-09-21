#ifndef COMMON_H
#define COMMON_H

#include <stdio.h>
#include <time.h>
#include <stdlib.h>

//compare array
void compare_arrays_i(int * a, int * b, int size);
void compare_arrays_f(float * a, float * b, int size, float precision);

//sum array
void sum_array_cpu_i(int *a, int *b, int *c, int size);
void sum_array_cpu_f(float *a, float *b, float *c, int size);

//print array
void print_array_i(int * input, const int array_size);
void print_array_f(float * input, const int array_size);

enum INIT_PARAM{
    INIT_ZERO,INIT_RANDOM,INIT_ONE,INIT_ONE_TO_TEN,INIT_FOR_SPARSE_METRICS,INIT_0_TO_X
};

void initialize_i(int * input, const int array_size,
                  enum INIT_PARAM PARAM, int x);
void initialize_f(float * input, const int array_size,
                  enum INIT_PARAM PARAM);


#endif // !COMMON_H