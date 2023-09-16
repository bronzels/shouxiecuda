#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "helper_cuda.h"
#include "helper_functions.h"

#include "common.hpp"

#include <stdio.h>
#include <ctime>
#include <random>
using namespace std;

__global__ void stream_test(int* in, int* out, int size)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid < size)
    {
        for (int i = 0; i < 25; i ++)
        {
            out[gid] = in[gid] + (in[gid] - 1) * (gid % 10);
        }
    }
}

int async()
{
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    StopWatchInterface *timer = NULL;
    sdkCreateTimer(&timer);
    sdkResetTimer(&timer);
    float gpu_time = 0.0f;
    sdkStartTimer(&timer);
    cudaEventRecord(start, 0);

    int size = 1 << 26;
    int byte_size = size * sizeof(int);

    //host pointers
    int *h_in, *h_ref, *h_in2, *h_ref2;

    //allocate memory for host pointers
    cudaMallocHost((void **)&h_in, byte_size);
    cudaMallocHost((void **)&h_ref, byte_size);
    cudaMallocHost((void **)&h_in2, byte_size);
    cudaMallocHost((void **)&h_ref2, byte_size);

    initialize(h_in, size, INIT_ONE_TO_TEN);
    initialize(h_in2, size, INIT_ONE_TO_TEN);

    //device pointers
    int *d_in, *d_out, *d_in2, *d_out2;

    //allocate memory for host pointers
    cudaMalloc((void **)&d_in, byte_size);
    cudaMalloc((void **)&d_out, byte_size);
    cudaMalloc((void **)&d_in2, byte_size);
    cudaMalloc((void **)&d_out2, byte_size);

    cudaStream_t str, str2;
    cudaStreamCreate(&str);
    cudaStreamCreate(&str2);

    //luanching the grid
    dim3 block(128);
    dim3 grid((size + block.x - 1) / block.x);

    cudaMemcpyAsync(d_in, h_in, byte_size, cudaMemcpyHostToDevice, str);
    stream_test <<< grid, block>>> (d_in, d_out, size);
    cudaMemcpyAsync(h_ref, d_out, byte_size, cudaMemcpyDeviceToHost, str);

    cudaMemcpyAsync(d_in2, h_in2, byte_size, cudaMemcpyHostToDevice, str2);
    stream_test <<< grid, block>>> (d_in2, d_out2, size);
    cudaMemcpyAsync(h_ref2, d_out2, byte_size, cudaMemcpyDeviceToHost, str2);

    cudaStreamSynchronize(str);
    cudaStreamDestroy(str);

    cudaStreamSynchronize(str2);
    cudaStreamDestroy(str2);

    cudaEventRecord(stop, 0);
    sdkStopTimer(&timer);
    unsigned long int counter = 0;
    while(cudaEventQuery(stop) == cudaErrorNotReady)
    {
        counter ++;
    }
    checkCudaErrors(cudaEventElapsedTime(&gpu_time, start, stop));
    printf("async time spent executing by the GPU: %.6f\n", gpu_time);

    compare_arrays(h_ref, h_ref2, size);

    cudaFreeHost(h_in);
    cudaFreeHost(h_in2);
    cudaFreeHost(h_ref);
    cudaFreeHost(h_ref2);
    cudaFree(d_in);
    cudaFree(d_in2);
    cudaFree(d_out);
    cudaFree(d_out2);

    checkCudaErrors(cudaDeviceReset());
    return 0;
}

int sync()
{
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    StopWatchInterface *timer = NULL;
    sdkCreateTimer(&timer);
    sdkResetTimer(&timer);
    float gpu_time = 0.0f;
    sdkStartTimer(&timer);
    cudaEventRecord(start, 0);

    int size = 1 << 27;
    int byte_size = size * sizeof(int);

    //host pointers
    int *h_in, *h_ref;

    //allocate memory for host pointers
    h_in = (int *)malloc(byte_size);
    h_ref = (int *)malloc(byte_size);

    initialize(h_in, size, INIT_ONE_TO_TEN);

    //device pointers
    int *d_in, *d_out;

    //allocate memory for host pointers
    checkCudaErrors(cudaMalloc((void **)&d_in, byte_size));
    checkCudaErrors(cudaMalloc((void **)&d_out, byte_size));

    //luanching the grid
    dim3 block(128);
    dim3 grid((size + block.x - 1) / block.x);

    checkCudaErrors(cudaMemcpy(d_in, h_in, byte_size, cudaMemcpyHostToDevice));
    stream_test <<< grid, block>>> (d_in, d_out, size);
    checkCudaErrors(cudaMemcpy(h_ref, d_out, byte_size, cudaMemcpyDeviceToHost));

    cudaEventRecord(stop, 0);
    sdkStopTimer(&timer);
    unsigned long int counter = 0;
    while(cudaEventQuery(stop) == cudaErrorNotReady)
    {
        counter ++;
    }
    checkCudaErrors(cudaEventElapsedTime(&gpu_time, start, stop));
    printf("sync time spent executing by the GPU: %.6f\n", gpu_time);

    free(h_in);
    free(h_ref);
    checkCudaErrors(cudaFree(d_in));
    (cudaFree(d_out));

    checkCudaErrors(cudaDeviceReset());
    return 0;
}

/*
int size = 1 << 19
async time spent executing by the GPU: 1.675520
Arrays are same
sync time spent executing by the GPU: 1.949344

int size = 1 << 22
async time spent executing by the GPU: 12.006400
Arrays are same
sync time spent executing by the GPU: 11.176992


int size = 1 << 27
async time spent executing by the GPU: 366.351166
Arrays are same
sync time spent executing by the GPU: 337.174377

 */

int main(int argc, char** argv) {
    async();
    sync();
}