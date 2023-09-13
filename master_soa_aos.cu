#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "helper_cuda.h"
#include "helper_functions.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define LEN 1<<22

struct testStruct {
    int x;
    int y;
};

struct structArray {
    int x[LEN];
    int y[LEN];
};

__global__ void test_aos(testStruct * in, testStruct * result, const int size)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gid < size)
    {
        testStruct temp = in[gid];
        temp.x += 5;
        temp.y += 10;
        result[gid] = temp;
    }
}

__global__ void test_soa(structArray * data, structArray * result, const int size)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gid < size)
    {
        float tmpx = data->x[gid];
        float tmpy = data->y[gid];

        tmpx += 5;
        tmpy += 10;
        result->x[gid] = tmpx;
        result->y[gid] = tmpy;
    }
}

int testAOS()
{
    printf(" testing AOS \n");

    int array_size = LEN;
    int byte_size = sizeof(structArray);
    int block_size = 128;

    testStruct * h_in, *h_ref;
    h_in = (testStruct*)malloc(byte_size);
    h_ref = (testStruct*)malloc(byte_size);

    for (int i = 0; i < array_size; i++)
    {
        h_in[i].x = 1;
        h_in[i].y = 2;
    }

    testStruct *d_in, *d_results;
    cudaMalloc((void **)&d_in, byte_size);
    cudaMalloc((void **)&d_results, byte_size);

    cudaMemcpy(d_in, h_in, byte_size, cudaMemcpyHostToDevice);

    dim3 block(block_size);
    dim3 grid((array_size + block.x - 1) / block.x);

    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    StopWatchInterface *timer = NULL;
    sdkCreateTimer(&timer);
    sdkResetTimer(&timer);
    float gpu_time = 0.0f;
    sdkStartTimer(&timer);
    cudaEventRecord(start, 0);

    test_aos <<<grid, block>>> (d_in, d_results, array_size);

    cudaEventRecord(stop, 0);
    sdkStopTimer(&timer);
    unsigned long int counter = 0;
    while(cudaEventQuery(stop) == cudaErrorNotReady)
    {
        counter ++;
    }
    checkCudaErrors(cudaEventElapsedTime(&gpu_time, start, stop));
    printf("time spent executing by the GPU: %.6f\n", gpu_time);

    cudaDeviceSynchronize();

    cudaMemcpy(&h_ref, &d_results, byte_size, cudaMemcpyDeviceToHost);

    cudaFree(d_results);
    cudaFree(d_in);
    free(h_in);
    free(h_ref);

    cudaDeviceReset();
}

int testSOA()
{
    printf(" testing SOA \n");

    int array_size = LEN;
    int byte_size = sizeof(structArray);
    int block_size = 128;

    structArray * h_in, *h_ref;
    h_in = (structArray*)malloc(byte_size);
    h_ref = (structArray*)malloc(byte_size);

    for (int i = 0; i < array_size; i++)
    {
        h_in->x[i] = 1;
        h_in->y[i] = 2;
    }

    structArray *d_in, *d_results;
    cudaMalloc((void **)&d_in, byte_size);
    cudaMalloc((void **)&d_results, byte_size);

    cudaMemcpy(d_in, h_in, byte_size, cudaMemcpyHostToDevice);

    dim3 block(block_size);
    dim3 grid((array_size + block.x - 1) / block.x);

    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    StopWatchInterface *timer = NULL;
    sdkCreateTimer(&timer);
    sdkResetTimer(&timer);
    float gpu_time = 0.0f;
    sdkStartTimer(&timer);
    cudaEventRecord(start, 0);

    test_soa <<<grid, block>>> (d_in, d_results, array_size);

    cudaEventRecord(stop, 0);
    sdkStopTimer(&timer);
    unsigned long int counter = 0;
    while(cudaEventQuery(stop) == cudaErrorNotReady)
    {
        counter ++;
    }
    checkCudaErrors(cudaEventElapsedTime(&gpu_time, start, stop));
    printf("time spent executing by the GPU: %.6f\n", gpu_time);

    cudaDeviceSynchronize();

    cudaMemcpy(&h_ref, &d_results, byte_size, cudaMemcpyDeviceToHost);

    cudaFree(d_results);
    cudaFree(d_in);
    free(h_in);
    free(h_ref);

    cudaDeviceReset();
}

int main(int argc, char** argv)
{
	int kernel_ind = 0;

	if (argc > 1)
	{
		kernel_ind = atoi(argv[1]);
	}

	if (kernel_ind == 0)
	{
		testAOS();
	}
	else
	{
		testSOA();
	}

	return EXIT_SUCCESS;
}
/*
aos
Memory Throughput [%]	92.01
L1/TEX Cache Throughput [%]	32.10
L2 Cache Throughput [%]	45.82
DRAM Throughput [%]	92.01
Memory Throughput [Gbyte/second]	317.40
Mem Busy [%]	45.82
L1/TEX Hit Rate [%]	49.61
Max Bandwidth [%]	92.01
L2 Hit Rate [%]	50.42
Mem Pipes Busy [%]	17.33

soa
Memory Throughput [%]	92.38
L1/TEX Cache Throughput [%]	32.22   (-100.00%)
L2 Cache Throughput [%]	45.64
DRAM Throughput [%]	92.38
Memory Throughput [Gbyte/second]	318.79
L1/TEX Hit Rate [%]	0
L2 Hit Rate [%]	50.02
Mem Busy [%]	45.64
Max Bandwidth [%]	92.38
Mem Pipes Busy [%]	17.40

testing AOS
time spent executing by the GPU: 0.221952

testing SOA
time spent executing by the GPU: 0.222656

 */
