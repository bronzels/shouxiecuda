#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <ctime>
#include <random>
using namespace std;

__global__ void vecAdd(int n, float *a, float *b, float *c)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n)
    {
        c[i] = a[i] + b[i];
    }
}

void init_rand_f(float* fPtr, long size)
{
    uniform_real_distribution<float> u(-1, 1);
    default_random_engine e(time(NULL));
    for(size_t i = 0; i < size; ++i)
        *(fPtr + i) = u(e);
}

int main()
{
    int N = 1024;
    long size_bytes = N * sizeof(float);
    float *a, *b, *c;
    float *devA, *devB, *devC;
    a = (float *)malloc(size_bytes);
    b = (float *)malloc(size_bytes);
    c = (float *)malloc(size_bytes);
    cudaMalloc(&devA, size_bytes);
    cudaMalloc(&devB, size_bytes);
    cudaMalloc(&devC, size_bytes);

    memset(c, 0, size_bytes);
    init_rand_f(a, N);
    init_rand_f(b, N);
    cudaMemcpy(devA, a, size_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(devB, b, size_bytes, cudaMemcpyHostToDevice);

    vecAdd <<< (N + 255) / 256, 256>>> (N, devA, devB, devC);
    cudaDeviceSynchronize();

    cudaMemcpy(c, devC, size_bytes, cudaMemcpyDeviceToHost);
    for(size_t i = 0; i < N; ++i)
    {
        printf("%f,", *(c + i));
        if( i % 64 ==0)
            printf("\n");
    }

    free(a);
    free(b);
    free(c);
    cudaFree(devA);
    cudaFree(devB);
    cudaFree(devC);
    cudaDeviceReset();
}
