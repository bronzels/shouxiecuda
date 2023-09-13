#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "helper_cuda.h"
#include "helper_functions.h"

#include <stdio.h>
#include <ctime>
#include <random>
#include <time.h>
using namespace std;

#define RADIUS 3
#define BLOCK_SIZE 16

__global__ void stencil(int n, int *in, int *out)
{
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (globalIdx < RADIUS || globalIdx > n - RADIUS - 1)
        return;
    int value = 0;
    for ( int offset = - RADIUS; offset <= RADIUS; offset++)
        value += in[globalIdx + offset];
    out[globalIdx] = value;
}


__global__ void stencil_shm(int n, int *in, int *out)
{
    __shared__ int shared[BLOCK_SIZE + 2 * RADIUS];
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    shared[globalIdx] = in[globalIdx];
    if (globalIdx < RADIUS || globalIdx > n - RADIUS - 1)
        return;
    __syncthreads();
    int value = 0;
    for ( int offset = - RADIUS; offset <= RADIUS; offset++)
        value += shared[globalIdx + offset];
    out[globalIdx] = value;
}

void init_rand_i(int* iPtr, long size)
{
    uniform_int_distribution<int> u(-100, 100);
    default_random_engine e(time(NULL));
    for(size_t i = 0; i < size; ++i)
        *(iPtr + i) = u(e);
}

int main()
{
    int N = BLOCK_SIZE + 2 * RADIUS;
    long size_bytes = N * sizeof(int);
    int *in, *out;
    int *devIn, *devOut;
    in = (int *)malloc(size_bytes);
    out = (int *)malloc(size_bytes);
    cudaMalloc(&devIn, size_bytes);
    cudaMalloc(&devOut, size_bytes);

    init_rand_i(in, N);
    printf("in\n");
    for(size_t i = 0; i < N; ++i)
    {
        printf("%d,", *(in + i));
        if( i > 0 && i % 16 ==0)
            printf("\n");
    }
    printf("\n");
    cudaMemcpy(devIn, in, size_bytes, cudaMemcpyHostToDevice);

    for (int i = 0; i < 2; i++) {
        checkCudaErrors(cudaDeviceSynchronize());
        memset(out, 0, size_bytes);

        cudaEvent_t start, stop;
        checkCudaErrors(cudaEventCreate(&start));
        checkCudaErrors(cudaEventCreate(&stop));
        StopWatchInterface *timer = NULL;
        sdkCreateTimer(&timer);
        sdkResetTimer(&timer);
        float gpu_time = 0.0f;
        /*
        clock_t start, end;
        double totaltime;
        start = clock();
        */
        sdkStartTimer(&timer);
        cudaEventRecord(start, 0);
        printf(i == 0 ? "---stencil\n" : "---stencil-shm\n");
        if (i == 0)
            stencil <<< (N + 31) / 32, 32>>> (N, devIn, devOut);
        else
            stencil_shm <<< (N + 31) / 32, 32>>> (N, devIn, devOut);
        //checkCudaErrors(cudaDeviceSynchronize());
        cudaEventRecord(stop, 0);
        sdkStopTimer(&timer);
        unsigned long int counter = 0;
        while(cudaEventQuery(stop) == cudaErrorNotReady)
        {
            counter ++;
        }
        checkCudaErrors(cudaEventElapsedTime(&gpu_time, start, stop));
        printf("time spent executing by the GPU: %.6f\n", gpu_time);
        printf("time spent by CPU in CUDA calls: %.6f\n", sdkGetTimerValue(&timer));
        printf("CPU executed %lu iterations while waiting for GPU to finish\n", counter);
        /*
        end = clock();
        totaltime = (double)(end - start);///CLOCKS_PER_SEC*1000;
        printf("start:%d\n", start);
        printf("end:%d\n", end);
        printf("totaltime:%f\n", totaltime);
        //start:550000
        //end:550000
        //totaltime:0.000000
        */
        /*
15,-74,7,-13,87,66,98,99,93,-27,-9,-75,-28,-79,29,26,4,
91,-87,-53,61,-71,
---stencil
time spent executing by the GPU: 0.03
time spent by CPU in CUDA calls: 0.03
CPU executed 20 iterations while waiting for GPU to finish
out
0,0,0,186,270,437,403,407,245,151,-26,-96,-163,-132,-32,-44,-69,
71,-29,0,0,0,
---stencil-shm
time spent executing by the GPU: 0.01
time spent by CPU in CUDA calls: 0.01
CPU executed 50 iterations while waiting for GPU to finish
out
0,0,0,186,270,437,403,407,245,151,-26,-96,-163,-132,-32,-44,-69,
71,-29,0,0,0,
        */

        cudaMemcpy(out, devOut, size_bytes, cudaMemcpyDeviceToHost);
        printf("out\n");
        for(size_t i = 0; i < N; ++i)
        {
            printf("%d,", *(out + i));
            if( i > 0 && i % 16 ==0)
                printf("\n");
        }
        printf("\n");
    }

    free(in);
    free(out);
    cudaFree(devIn);
    cudaFree(devOut);
    cudaDeviceReset();
}
