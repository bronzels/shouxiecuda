#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include "helper_cuda.h"
#include "helper_functions.h"

#include "common.cpph"

__global__ void dynamic_parallelism_check(int size, int depth)
{
	printf(" Depth : %d - tid : %d \n", depth, threadIdx.x);

	if (size == 1)
		return;

	if (threadIdx.x == 0)
	{
		dynamic_parallelism_check <<<1, size / 2 >>> (size / 2, depth + 1);
	}
}

__global__ void dynamic_parallelism_check_ex2(int size, int depth)
{
    printf(" Depth: %d - blockIdx.x : %d - threadIdx.x : %d \n", depth, blockIdx.x, threadIdx.x);

    if (size == 1)
        return;

    if (threadIdx.x == 0)
    {
        dynamic_parallelism_check_ex2<<<1, size/2>>>(size/2, depth+1);
    }
}

__global__ void dynamic_parallelism_check_ex3(int size, int depth)
{
    printf(" Depth: %d - blockIdx.x : %d - threadIdx.x : %d \n", depth, blockIdx.x, threadIdx.x);

    if (size == 1)
        return;

    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        dynamic_parallelism_check_ex2<<<2, size/2>>>(size/2, depth+1);
    }
}

int main(int argc, char** argv)
{
    //dynamic_parallelism_check<<<1,16>>>(16, 0);
    dynamic_parallelism_check_ex2<<<2,8>>>(8, 0);

    cudaDeviceSynchronize();
    cudaDeviceReset();
    return 0;
}