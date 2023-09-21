#ifndef CUDA_COMMON_H
#define CUDA_COMMON_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/replace.h>
#include <thrust/gather.h>

#include "cublas_v2.h"

#include <stdio.h>
#include <iostream>
#include <assert.h>

#define vectorPtr(x) thrust::raw_pointer_cast(x.data())

void ShowCudaGpuInfo();

template <class T>
void thrust_transpose(T *h_mat_array, T *h_trans_array, int nx, int ny);
template <class T>
void thrust_transpose(thrust::device_vector<T> &v_mat_d, thrust::device_vector<T> &v_out_d, int nx, int ny);
void cublas_transpose(float *h_mat_array, float *h_trans_array, int nx, int ny);

#ifndef cublasSafeCall
#define cublasSafeCall(err) __cublasSafeCall(err, __FILE__, __LINE__)
#endif

inline void __cublasSafeCall(cublasStatus_t err, const char *file, const int line) {
    if( CUBLAS_STATUS_SUCCESS != err) {
        fprintf(stderr, "CUBLAS error in file '%s', line %d\nerror %d \nterminating!\n", file, line, err);
        cudaDeviceReset();assert(0);
    }
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

__global__ void scan_efficient_1G(int * input, int* auxiliry_array, int input_size);
__global__ void scan_summation(int * input, int * auxiliry_array, int input_size);

#endif // !CUDA_COMMON_H

//void query_device();