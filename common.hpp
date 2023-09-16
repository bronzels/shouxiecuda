#ifndef COMMON_CPPH
#define COMMON_CPPH

#include <stdio.h>
#include <time.h>
#include <random>
#include <stdlib.h>
#include <fstream>

//compare two arrays
template <class T>
void compare_arrays(T * a, T * b, int size, T precision);
template <class T>
void compare_arrays(T * a, T * b, int size);

//matrix transpose in CPU
template <class T>
void mat_transpose_cpu(T * mat, T * transpose, int nx, int ny);

template <typename T>
void sum_arry_cpu(T *a, T *b, T *c, int size);

/*
template <typename T>
std::ostream& operator<< (std::ostream& out, const std::vector<T>& v);
//template <int> std::ostream& operator<< (std::ostream& out, const std::vector<int>& v);
*/
template <typename T> void print_vec(const std::vector<T> &v, int start, int end, char *sep, int tablen);

template <typename T>
void add_bias_(std::vector<T> &v, const T &a);

#define HANDLE_NULL( a ){if (a == NULL) { \
                            printf( "Host memory failed in %s at line %d\n", \
                                    __FILE__, __LINE__ ); \
                            exit( EXIT_FAILURE );}}
enum INIT_PARAM{
	INIT_ZERO,INIT_RANDOM,INIT_ONE,INIT_ONE_TO_TEN,INIT_FOR_SPARSE_METRICS,INIT_0_TO_X,INIT_RANDOM_KEY
};

//simple initialization
void initialize(int * input, const unsigned int array_size,
	INIT_PARAM PARAM = INIT_ONE_TO_TEN, int x = 0);

void initialize(float * input, const unsigned int array_size,
	INIT_PARAM PARAM = INIT_ONE_TO_TEN);

void launch_dummmy_kernel();

//compare two matrics
void compare_matrixes(int *a, int *b, int m, int n);
void compare_matrixes(float *a, float *b, int m, int n, float precision);

//reduction in cpu
int reduction_cpu(int * input, const int size);

//compare results
void compare_results(int gpu_result, int cpu_result);
void compare_results(float gpu_result, float cpu_result, float precision);

//print array
void print_array(int * input, const int array_size);
void print_array(float * input, const int array_size);

//print matrix
void print_matrix(int * matrix, int nx, int ny);

void print_matrix(float * matrix, int nx, int ny);

//get matrix
int* get_matrix(int rows, int columns);


//print_time_using_host_clock
void print_time_using_host_clock(clock_t start, clock_t end);

void printData(char *msg, int *in, const int size);

void sum_array_cpu(float* a, float* b, float *c, int size);

void print_arrays_toafile(int*, int , char* );

void print_arrays_toafile_side_by_side(float*,float*,int,char*);

void print_arrays_toafile_side_by_side(int*, int*, int, char*);

#endif // !COMMON_H