#include <stdio.h>
#include <stdlib.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "common.h"
#include "cuda_common.cuh"

__global__ void blocking_nonblocking_testnull()
{
    int gid = blockDim.x * blockIdx.x + threadIdx.x;
    if (gid == 0)
    {
        for (size_t i = 0; i < 10000; i++)
        {
            printf("stream null \n");
        }
    }
}

__global__ void blocking_nonblocking_test1()
{
    int gid = blockDim.x * blockIdx.x + threadIdx.x;
    if (gid == 0)
    {
        for (size_t i = 0; i < 10000; i++)
        {
            printf("stream 1 \n");
        }
    }
}

__global__ void blocking_nonblocking_test2()
{
    int gid = blockDim.x * blockIdx.x + threadIdx.x;
    if (gid == 0)
    {
        for (size_t i = 0; i < 10000; i++)
        {
            printf("stream 2 \n");
        }
    }
}

__global__ void blocking_nonblocking_test3()
{
    int gid = blockDim.x * blockIdx.x + threadIdx.x;
    if (gid == 0)
    {
        for (size_t i = 0; i < 10000; i++)
        {
            printf("stream 3 \n");
        }
    }
}

__global__ void blocking_nonblocking_test4()
{
    int gid = blockDim.x * blockIdx.x + threadIdx.x;
    if (gid == 0)
    {
        for (size_t i = 0; i < 10000; i++)
        {
            printf("stream 4 \n");
        }
    }
}

int main(int argc, char ** argv)
{
	int size = 1 << 15;

    cudaStream_t stm1, stm2, stm3, stm4;

    cudaEvent_t event1;
    cudaEventCreate(&event1, cudaEventDisableTiming);

    cudaStreamCreateWithFlags(&stm1, cudaStreamNonBlocking);
    cudaStreamCreate(&stm2);
    cudaStreamCreateWithFlags(&stm3, cudaStreamNonBlocking);
    cudaStreamCreate(&stm4);

	dim3 block(128);
	dim3 grid(size / block.x);



    blocking_nonblocking_test1<<<grid, block, 0, stm1>>>();
    cudaEventRecord(event1, stm1);
    cudaStreamWaitEvent(stm3, event1, 0);

    blocking_nonblocking_test2<<<grid, block, 0, stm2>>>();
    blocking_nonblocking_testnull<<<grid, block>>>();
    blocking_nonblocking_test3<<<grid, block, 0, stm3>>>();
    blocking_nonblocking_test4<<<grid, block, 0, stm4>>>();

    cudaEventDestroy(event1);
/*
stream 2
stream 3
stream 1
stream 2
stream 3
stream 1
stream 2
stream 3
stream 1
stream 2
stream 3
stream 1
stream 2
stream 3
stream null
stream null
stream null
stream null
stream null
stream null
stream null
stream null
stream 4
stream 4
stream 4
stream 4
stream 4
stream 4
stream 4
stream 4
*/

/*cudaStreamWaitEvent(stm3, event1, 0);
stream 1
stream 2
stream 1
stream 2
stream 1
stream 2
stream 1
stream 2
stream 3
stream null
stream 3
stream null
stream 3
stream null
stream 3
stream null
stream 3
stream null
stream 3
stream null
stream 3
stream null
stream 3
stream null
stream 4
stream 4
stream 4
stream 4
stream 4
stream 4
stream 4
stream 4
*/
    cudaStreamDestroy(stm1);
    cudaStreamDestroy(stm2);
    cudaStreamDestroy(stm3);
    cudaStreamDestroy(stm4);

	gpuErrchk(cudaDeviceSynchronize());

	gpuErrchk(cudaDeviceReset());
	return 0;
}